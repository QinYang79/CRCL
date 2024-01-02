"""Training script"""

import logging
import os
import torch
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorboard_logger as tb_logger
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import math
import data
import opts
from model.CRCL import CRCL
from evaluation import encode_data, encode_data_vse, i2t_vse, shard_attn_scores, i2t, t2i, AverageMeter, LogCollector, t2i_vse
from utils import cosine_similarity_matrix, save_checkpoint
from vocab import deserialize_vocab
from matplotlib import pyplot as plt
import warnings
from pylab import xticks,yticks,np
warnings.filterwarnings("ignore")

def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(opt, data_loader, model, epoch, schedule,val_loader,test_loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1
    end = time.time()
    logger.info("=> Training epoch: {0}".format(epoch))
    
    for i, train_data in enumerate(data_loader):
        model.train_start()
        # if i==2:
        #     break
        images, img_lengths, targets, cap_lengths,  text_ids = train_data
        if images.size(0) == 1:
            break
        data_time.update(time.time() - end)
        model.logger = train_logger
        # model.train_emb( images,targets, cap_lengths,  text_ids ,epoch=epoch)
        model.train_self(images,img_lengths, targets, cap_lengths, text_ids, epoch = epoch,schedule =schedule) 
     
        batch_time.update(time.time() - end)
        if model.step % opt.log_step == 0:
            logger.info(
                'Epoch: [{3}-{0}][{1}/{2}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f}) \t{loss}'.format(epoch, i, num_loader_iter,schedule,
                                                                                 batch_time=batch_time,
                                                                                 data_time=data_time,
                                                                                 loss=str(model.logger)))
        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.step)
        tb_logger.log_value('step', i, step=model.step)
        tb_logger.log_value('batch_time', batch_time.val, step=model.step)
        tb_logger.log_value('data_time', data_time.val, step=model.step)
        model.logger.tb_log(tb_logger, step=model.step)
        
        if (model.step+1) % opt.val_step== 0:
            validation(opt, val_loader, model, tag = f'val_per{opt.val_step}')
            validation(opt, test_loader, model, tag = f'test_per{opt.val_step}')


def validation(opt, val_loader, model, tag=''):
    # compute the encoding for all the validation images and captions
    if opt.data_name in ['cc152k_precomp', 'cc510k_precomp']:
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        logger.info(f"No dataset")
        return 0
    if 'val' not in tag:
        logger.info(f"=> Test")
    else:
        logger.info(f"=> Validation")
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        if opt.module_name == 'VSEinfty':
            img_embs, cap_embs = encode_data_vse(model.base_model, val_loader)
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
            start = time.time()
            sims = cosine_similarity_matrix(img_embs, cap_embs)
            end = time.time()
            logger.info("calculate similarity time:".format(end - start))

            # caption retrieval
            npts = img_embs.shape[0]
            # (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)
            (r1, r5, r10, medr, meanr) = i2t_vse(npts, sims, per=per_captions)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                        (r1, r5, r10, medr, meanr))
            # image retrieval
            # (r1i, r5i, r10i, medri, meanr) = t2i(img_embs, cap_embs, cap_lens, sims)
            (r1i, r5i, r10i, medri, meanr) = t2i_vse(npts, sims, per=per_captions)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                        (r1i, r5i, r10i, medri, meanr))
            # sum of recalls to be used for early stopping
            r_sum = r1 + r5 + r10 + r1i + r5i + r10i
            logger.info('Current rsum is {}'.format(r_sum))

        else:
            img_embs, cap_embs, cap_lens = encode_data(model.base_model, val_loader, opt.log_step)
            # clear duplicate 5*images and keep 1*images
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])

            # record computation time of validation
            start = time.time()
            sims = shard_attn_scores(model.base_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000)
            end = time.time()
            logger.info(f"calculate similarity time: {end - start}")
            (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims, per_captions, return_ranks=False)
            logger.info("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
            logger.info("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
            (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims, per_captions, return_ranks=False)
            logger.info("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
            logger.info("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
            r_sum = r1 + r5 + r10 + r1i + r5i + r10i
            logger.info("Sum of Recall: %.2f" % (r_sum))

    # sum of recalls to be used for early stopping

    tb_logger.log_value(f'{tag}_r1', r1, step=model.step)
    tb_logger.log_value(f'{tag}_r5', r5, step=model.step)
    tb_logger.log_value(f'{tag}_r10', r10, step=model.step)
    tb_logger.log_value(f'{tag}_medr', medr, step=model.step)
    tb_logger.log_value(f'{tag}_meanr', meanr, step=model.step)
    tb_logger.log_value(f'{tag}_r1i', r1i, step=model.step)
    tb_logger.log_value(f'{tag}_r5i', r5i, step=model.step)
    tb_logger.log_value(f'{tag}_r10i', r10i, step=model.step)
    tb_logger.log_value(f'{tag}_medri', medri, step=model.step)
    tb_logger.log_value(f'{tag}_meanr', meanr, step=model.step)
    tb_logger.log_value(f'{tag}_r_sum', r_sum, step=model.step)

    return r_sum
 

def init_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


if __name__ == '__main__':
    opt = opts.parse_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    tb_logger.configure(opt.log_dir, flush_secs=5)
    logger = init_logging(opt.log_dir + '/log.txt')
    logger.info(opt)
    logger.info(f"=> PID:{os.getpid()}, GUP:[{opt.gpu}], Noise ratio: {opt.noise_ratio}")
    logger.info(f"=> Log save path: '{opt.log_dir}'")
    logger.info(f"=> Checkpoint save path: '{opt.checkpoint_dir}'") 
    # Load Vocabulary
    logger.info(f"=> Load vocabulary from '{opt.vocab_path}'")
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)
    # Load data loaders
    logger.info(f"=> Load loaders from '{opt.data_path}/{opt.data_name}'")
    train_loader, val_loader, test_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size,
                                                             opt.workers, opt, logger=logger)
    # Construct the model (MRL-)
    logger.info(f"=> Similarity model is {opt.module_name}")
    data_length = train_loader.dataset.length

    start_epoch = 0
    start_schedule = 0
    real_correspondence = train_loader.dataset.real_correspondence
    best_rsum = 0
    model = CRCL(opt)
    model.move_gt = np.ones(data_length)
    model.loss = np.zeros(data_length)
    model.prob =  np.zeros(data_length)
    model.move_gt_sharpen = np.ones(data_length)
    model.sim =  np.zeros(data_length)
    model.pred_correspondence =  np.ones(data_length)
    clean_index = np.where(real_correspondence==1)[0]
    noisy_index = np.where(real_correspondence==0)[0]
 
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_schedule =   checkpoint['schedule']
            start_epoch = checkpoint['epoch'] + 1
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # step is used to show logs as the continuation of another training
            model.step = checkpoint['step']
            model.move_gt = checkpoint['move_gt']
            model.pred_correspondences = checkpoint['pred_correspondence']
            
            train_loader.dataset.pred_correspondences = model.pred_correspondences
      
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            logger.info("=> start epoch '{}' start schedule {}"
                        .format(start_epoch,start_schedule))
            # validation(opt, test_loader, model)
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    for i in range(start_schedule, len(opt.schedules)): 
        ee = opt.schedules[i] +  opt.warm_epoch
        # for epoch in range(start_epoch, opt.num_epochs):   3
        for epoch in range(start_epoch, ee):   
            adjust_learning_rate(opt, model.optimizer, epoch)
            start_time = datetime.now()   
            train(opt, train_loader, model, epoch, schedule = i, val_loader=val_loader,test_loader=test_loader)
            end_time = datetime.now()
            tb_logger.log_value('cost_time', int((end_time - start_time).seconds), step=epoch)
 
            rsum = validation(opt, val_loader, model, tag='val')
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)
 
            validation(opt, test_loader, model, tag='test')
 
            if epoch == ee-1:
                save_checkpoint({
                    'schedule': i,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_rsum': best_rsum,
                    'opt': opt,
                    'step': model.step,   
                    'move_gt': model.move_gt, 
                }, is_best, filename='checkpoint_{}_{}.pth.tar'.format(i,epoch), prefix=opt.checkpoint_dir + '/',ckpt=True)
            else:
                save_checkpoint({
                    'schedule': i,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_rsum': best_rsum,
                    'opt': opt,
                    'step': model.step, 
                    'move_gt': model.move_gt
                    }, is_best, filename='checkpoint_{}_{}.pth.tar'.format(i,epoch), prefix=opt.checkpoint_dir + '/',ckpt=False)

        logger.info(f"Best rSum: {best_rsum}")
        start_epoch = 0
        best_rsum = 0
        model1 = CRCL(opt)
        model1.move_gt =  model.move_gt 
        model1.step =   0 
        model = model1

 