 
from model.VSEinfty import VSEModel
from model.SGRAF import SGRAF
import torch 
import torch.nn.init 
from torch.nn.utils import clip_grad_norm_ 
from utils import cosine_similarity_matrix

import logging

logger = logging.getLogger(__name__)

class CRCL(object):
    def __init__(self, opt):
        self.opt = opt
        self.tau = self.opt.tau
        self.alpha = self.opt.alpha
        self.module_name = opt.module_name
        if self.module_name == 'VSEinfty':
            self.base_model = VSEModel(opt)
        else:
            self.base_model = SGRAF(opt)

        self.params = list(self.base_model.params) 

        self.optimizer = torch.optim.AdamW(self.params, lr=opt.learning_rate)
  
        self.move_gt = None 
        self.step = 0

    def train_start(self):
        self.base_model.train_start()

    def val_start(self):
        self.base_model.val_start()

    def state_dict(self):
        return self.base_model.state_dict()

    def load_state_dict(self, state_dict):
        self.base_model.load_state_dict(state_dict)

    def forward_sims(self, img_embs, cap_embs, caption_lengths):
        if self.module_name == 'VSEinfty':
            return cosine_similarity_matrix(img_embs, cap_embs)
        else:
            return self.base_model.forward_sim(img_embs, cap_embs, caption_lengths)

    def forward_emb(self, images, image_lengths, captions, caption_lengths):
        if self.module_name == 'VSEinfty':
            img_embs, cap_embs = self.base_model.forward_emb(images, captions, lengths=caption_lengths, image_lengths= image_lengths)
        else:
            img_embs, cap_embs, _  =  self.base_model.forward_emb(images, captions, caption_lengths)

        return img_embs, cap_embs
 
    def train_self(self,  images, img_lengths,  targets, cap_lengths,  text_ids, epoch=None, schedule=0):
        self.step += 1
        self.logger.update('step', self.step)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        k = images.size(0)
        y_hard = torch.eye(k).cuda()
        self.optimizer.zero_grad()
        cap_lengths = list(map(int, cap_lengths.numpy().tolist())) 
        img_embs, cap_embs = self.forward_emb(images, img_lengths, targets, cap_lengths)
        eps = 1e-10
        y_last = torch.Tensor(self.move_gt[text_ids]).cuda()
        sims = self.forward_sims(img_embs, cap_embs, cap_lengths)
        sims_ =  torch.exp(sims/self.tau)

        p1 = sims_ / (torch.sum(sims_, dim=1, keepdim=True) + eps)
        p2 = sims_.t() / (torch.sum(sims_.t(), dim=1, keepdim=True) + eps)
        p = ((p1.diag()+p2.diag())/2).clone().detach() # Prevent Nan values

        if self.opt.mc:
            # freeze
            if epoch < self.opt.warm_epoch:
                y = y_last
            else:
                y = self.alpha * y_last + (1 - self.alpha) * p 
        else:
            y = p 
    
        if self.opt.mc:
            if torch.isnan(y).any():
                print(y)
                exit(0)
            if self.opt.confident:
                y[y<0.1] = 0
        
        loss_positive = - torch.log(p1.diag())
        loss_positive += - torch.log(p2.diag())

        if epoch == self.opt.warm_epoch-1 and schedule == 0:
            self.move_gt[text_ids] = p.clone().detach().cpu().numpy()
        else:
            self.move_gt[text_ids] = y.clone().detach().cpu().numpy()
  
        loss_positive = - torch.log(p1.diag())
        loss_positive += - torch.log(p2.diag())

        loss_neg = (p1.tan()* (1 - y_hard)).sum(1) /(p1.tan().sum(1)**(1-y)) 
        loss_neg += (p2.tan()* (1 - y_hard)).sum(1) / (p2.tan().sum(1)**(1-y))
  
        loss1 = (y * loss_positive).mean()
        loss2 =  loss_neg.mean() * 5

        loss = loss1 + loss2
        self.logger.update('loss1', loss_positive.mean().item(), k)
        self.logger.update('loss2', loss_neg.mean().item(), k)
        self.logger.update('loss', loss.item(), k)

        loss.backward()
        if self.opt.grad_clip > 0:
            clip_grad_norm_(self.params, self.opt.grad_clip)
        self.optimizer.step()

    



