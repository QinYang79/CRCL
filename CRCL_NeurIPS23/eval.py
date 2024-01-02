# coding=utf-8
import os
import torch
from evaluation import validation_SGR_or_SAF, validation_SGRAF, validation_VSEinfty
from data import get_test_loader
from model.CRCL import CRCL
from vocab import deserialize_vocab


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model_paths = ['/home/qinyang/ProjectsOfITM/RECO_bak/runs/NCR_logs/cc152k_SAF_0.0_0.1_weight_1/checkpoint_dir/model_best.pth.tar',
         '/home/qinyang/ProjectsOfITM/RECO_bak/runs/NCR_logs/cc152k_SGR_0.0_0.1_weight_1/checkpoint_dir/model_best.pth.tar']
    
    # model_paths = ['/home/qinyang/ProjectsOfITM/CRCL/runs/NCR_logs/f30k_VSEinfty_0.8/checkpoint_dir/model_best.pth.tar']
    data_path = '/home/qinyang/projects/data/cross_modal_data/data/data'
    vocab_path = '/home/qinyang/projects/data/cross_modal_data/data/vocab'
    data_path =None
    vocab_path = None
    # eavl SGRAF.
    avg_SGRAF = True

    module_name = torch.load(model_paths[0])['opt'].module_name
    print(f"==>Evaluate module is {module_name}")
  
    if module_name == 'VSEinfty':
        print(f"Load checkpoint from '{model_paths[0]}'")
        checkpoint = torch.load(model_paths[0])
        opt = checkpoint['opt']
        if vocab_path is not None:
            opt.vocab_path = vocab_path
        if data_path is not None:
            opt.data_path = data_path  
        print(
            f"Noise ratio is {opt.noise_ratio}, module is {opt.module_name}, best validation epoch is {checkpoint['epoch']} ({checkpoint['best_rsum']})")
        if vocab_path != None:
            opt.vocab_path = vocab_path
        if data_path != None:
            opt.data_path = data_path
        vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
        model = CRCL(opt)
        model.load_state_dict(checkpoint['model'])
        if 'coco' in opt.data_name:
            test_loader = get_test_loader('testall', opt.data_name, vocab, 100, 0, opt)
            validation_VSEinfty(opt, test_loader, model=model, fold=True)
            validation_VSEinfty(opt, test_loader, model=model, fold=False)
        else:
            test_loader = get_test_loader('test', opt.data_name, vocab, 100, 0, opt)
            validation_VSEinfty(opt, test_loader, model=model, fold=False)
    else:
        if avg_SGRAF and len(model_paths) == 2:
            print(f"Load checkpoint from '{model_paths}'")
            checkpoint0 = torch.load(model_paths[0])
            checkpoint1 = torch.load(model_paths[1])
            opt0 = checkpoint0['opt']
            opt1 = checkpoint1['opt']
            if vocab_path is not None:
                opt0.vocab_path = vocab_path
            if data_path is not None:
                opt0.data_path = data_path  
            print(
                f"Noise ratios are {opt0.noise_ratio} and {opt1.noise_ratio}, "
                f"modules are {opt0.module_name} and {opt1.module_name}, best validation epochs are {checkpoint0['epoch']}"
                f" ({checkpoint0['best_rsum']}) and {checkpoint1['epoch']} ({checkpoint1['best_rsum']})")
            vocab = deserialize_vocab(os.path.join(opt0.vocab_path, '%s_vocab.json' % opt0.data_name))
            model0 = CRCL(opt0)
            model1 = CRCL(opt1)

            model0.load_state_dict(checkpoint0['model'])
            model1.load_state_dict(checkpoint1['model'])

            if 'coco' in opt0.data_name:
                test_loader = get_test_loader('testall', opt0.data_name, vocab, 100, 0, opt0)
                print(f'=====>model {opt0.module_name} fold:True')
                validation_SGR_or_SAF(opt0, test_loader, model0, fold=True)
                print(f'=====>model {opt1.module_name} fold:True')
                validation_SGR_or_SAF(opt0, test_loader, model1, fold=True)
                print(f'=====>model SGRAF fold:True')
                validation_SGRAF(opt0, test_loader, models=[model0, model1], fold=True)

                print(f'=====>model {opt0.module_name} fold:False')
                validation_SGR_or_SAF(opt0, test_loader, model0, fold=False)
                print(f'=====>model {opt1.module_name} fold:False')
                validation_SGR_or_SAF(opt0, test_loader, model1, fold=False)
                print('=====>model SGRAF fold:False')
                validation_SGRAF(opt0, test_loader, [model0, model1], fold=False)
            else:
                test_loader = get_test_loader('test', opt0.data_name, vocab, 100, 0, opt0)
                print(f'=====>model {opt0.module_name} fold:False')
                validation_SGR_or_SAF(opt0, test_loader, model0, fold=False)
                print(f'=====>model {opt1.module_name} fold:False')
                validation_SGR_or_SAF(opt0, test_loader, model1, fold=False)
                print('=====>model SGRAF fold:False')
                validation_SGRAF(opt0, test_loader, models=[model0, model1], fold=False)   
        else:
            print(f"Load checkpoint from '{model_paths[0]}'")
            checkpoint = torch.load(model_paths[0])
            opt = checkpoint['opt']
            if vocab_path is not None:
                opt.vocab_path = vocab_path
            if data_path is not None:
                opt.data_path = data_path  
            print(
                f"Noise ratio is {opt.noise_ratio}, module is {opt.module_name}, best validation epoch is {checkpoint['epoch']} ({checkpoint['best_rsum']})")
            if vocab_path != None:
                opt.vocab_path = vocab_path
            if data_path != None:
                opt.data_path = data_path
            vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
            model = CRCL(opt)
            model.load_state_dict(checkpoint['model'])
            if 'coco' in opt.data_name:
                test_loader = get_test_loader('testall', opt.data_name, vocab, 100, 0, opt)
                validation_SGR_or_SAF(opt, test_loader, model=model, fold=True)
                validation_SGR_or_SAF(opt, test_loader, model=model, fold=False)
            else:
                test_loader = get_test_loader('test', opt.data_name, vocab, 100, 0, opt)
                validation_SGR_or_SAF(opt, test_loader, model=model, fold=False)
