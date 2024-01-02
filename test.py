# Model evaluation.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.backends.cudnn as cudnn
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
from model.stirer import STIRER
from model.crnn import CRNN
from torch.utils.data import ConcatDataset
from utils.loss import ImageLoss
from utils.text_focus_loss import TextFocusLoss
from utils.label_converter import get_charset, strLabelConverter, str_filt
from utils.ssim_psnr import calculate_psnr, SSIM
from dataset import LRSTRDataset, LRSTR_collect_fn
def eval(model, eval_dataset):
    print('----------------<Evaluation>----------------')
    model.eval()
    global_accs = []
    global_crnn_accs = []
    # for CE decoder
    decode_mapper = {}
    for i,c in enumerate(charset):
        decode_mapper[i+1] = c
    decode_mapper[0] = ''
    print(decode_mapper)
    # Evaluation
    for eval_dataset in eval_datasets:
        crnn_accs = []
        print("Evaluating", eval_dataset.dataset_name)
        eval_data_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=64,
            shuffle=False, num_workers=2,
            drop_last=False,collate_fn=collect_fn_eval)
        metrics_recorder = {}
        for it, batch in enumerate(eval_data_loader):
            img_hr, img_lr, label_tensors, length_tensors, label_strs, _ = batch
            img_hr = img_hr.cuda()
            img_lr = img_lr.cuda()
            label_tensors = label_tensors.cuda()
            length_tensors = length_tensors.cuda()
            with torch.no_grad():
                sr, logit, srs, logits = model(img_lr)
                logits.append(logit)
                srs.append(sr)
            for step in range(len(srs)):
                if not 'acc_'+str(step) in metrics_recorder.keys():
                    metrics_recorder['acc_'+str(step)] = []
                    metrics_recorder['psnr_'+str(step)] = []
                    metrics_recorder['ssim_'+str(step)] = []
                BS = img_hr.size(0)
                if args['sr']:
                    img_sr = srs[step]
                if args['rec']:
                    if step == len(srs)-1:
                        pred_strs = []
                        for i in range(logits[step].size(0)):
                            ids = logits[step][i].tolist()
                            pred_str = ''
                            for j in range(len(ids)):
                                if ids[j] == len(charset)+1:
                                    break
                                # print('HANDLING',ids[j],decode_mapper[ids[j]])
                                pred_str += decode_mapper[ids[j]]
                                # print("PP",pred_str)
                            pred_strs.append(pred_str)
                        # print('gt=',label_strs[i],'pred=',pred_str,'data=',logits[step][i])
                    else:
                        logit = logits[step].softmax(-1)
                        pred_strs = label_converter.decode(torch.argmax(logit,2).view(-1),torch.IntTensor([logit.size(1)]*BS))
                for i in range(BS):
                    if args['sr']:
                        sr = img_sr[i,...]
                        hr = img_hr[i,...]
                        psnr = calculate_psnr(sr.unsqueeze(0),hr.unsqueeze(0)).cpu()
                        ssim = calculate_ssim(sr.unsqueeze(0),hr.unsqueeze(0)).cpu()
                        metrics_recorder['psnr_'+str(step)].append(psnr)
                        metrics_recorder['ssim_'+str(step)].append(ssim)
                    if args['rec']:
                        pred_str = pred_strs[i]
                        # pred_str = str_filt(pred_str,'lower')
                        gt_str = label_strs[i]
                        # gt_str = str_filt(gt_str,'lower')
                        if pred_str == gt_str:
                            metrics_recorder['acc_'+str(step)].append(True)
                        else:
                            metrics_recorder['acc_'+str(step)].append(False)
            logits_last = crnn(srs[-1]).softmax(-1).transpose(0,1)
            pred_strs_crnn = label_converter.decode(torch.argmax(logits_last,2).view(-1),torch.IntTensor([logits_last.size(1)]*BS))
            for i in range(BS):
                pred_str = pred_strs_crnn[i]
                pred_str = str_filt(pred_str,'lower')
                gt_str = label_strs[i]
                gt_str = str_filt(gt_str,'lower')
                if pred_str == gt_str:
                    crnn_accs.append(True)
                else:
                    crnn_accs.append(False)
                global_crnn_accs.append(crnn_accs[-1])
            # os._exit(-1)
        for k in metrics_recorder.keys():
            if len(metrics_recorder[k]) == 0:
                metrics_recorder[k].append(-1)
        for step in range(len(srs)):
            print("STEP %d Acc %.2f PSNR %.2f SSIM %.2f"%(step,
            100.0*sum(metrics_recorder['acc_'+str(step)])/len(metrics_recorder['acc_'+str(step)]),
            sum(metrics_recorder['psnr_'+str(step)])/len(metrics_recorder['psnr_'+str(step)]),
            100.0*sum(metrics_recorder['ssim_'+str(step)])/len(metrics_recorder['ssim_'+str(step)])),flush=True)
        # add the last
        step = len(srs)-1
        global_accs.append(sum(metrics_recorder['acc_'+str(step)])/len(metrics_recorder['acc_'+str(step)]))
        print("CRNN Acc %.2f"%(100.0*sum(crnn_accs)/len(crnn_accs)))
    final_acc = sum(global_accs)/len(global_accs)
    print("Avg CRNN Acc %.2f"%(100.0*sum(global_crnn_accs)/len(global_crnn_accs)))
    model.train()
    return final_acc


args = {
    'exp_name': 'EVAL',
    'batch_size': 48,#128
    'multi_card': True,
    'train_dataset': [
        'dataset/textzoom/train1/',
        'dataset/textzoom/train2/',
    ],
    'eval_dataset': [
        'dataset/textzoom/test/easy/',
        'dataset/textzoom/test/medium/',
        'dataset/textzoom/test/hard/'
    ],
    'Epoch': 500,
    'alpha': 0.5,
    'print_iter': 10,
    'num_gpu': 2 ,
    'eval_iter': 100,
    'resume':'ckpt/STIRER_Final_Release.pth',
    'seed':3407,
    'sr':True,
    'rec':True,
    'multi_stage': False,
    'charset': 37,
    'ar': True
}
torch.manual_seed(args['seed']) #cpu
torch.cuda.manual_seed(args['seed']) #gpu
 
np.random.seed(args['seed']) #numpy
random.seed(args['seed']) # random and transforms
torch.backends.cudnn.deterministic=True #cudnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STIRER(upscale=2, img_size=(16, 64),
                   window_size=(16,2), img_range=1., depths=[4,5,6],
                   embed_dim=48, num_heads=[6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect',is_eval=False).to(device)
crnn = CRNN(32, 1, 37, 256).cuda()
crnn.load_state_dict(torch.load('ckpt/crnn.pth',map_location='cpu'),strict=True)
crnn.eval()
sr_param_list = []
for n,p in model.named_parameters():
    if 'upsample' in n or 'conv_after_body' in n:
        sr_param_list.append(id(p))
if args['resume'] != '':
    print('Loading parameters from',args['resume'])
    params = torch.load(args['resume'],map_location='cpu')
    # cleaned_params = {}
    # for q in params.keys():
    #     if not 'last_sr' in q:
    #         cleaned_params[q] = params[q]
    model.load_state_dict(params,strict=False)
if args['multi_card']:
    model = torch.nn.DataParallel(model, device_ids=range(args['num_gpu']))
# x = torch.randn(4,4,16,64).to(device)
# model(x)
# os._exit(2333)
train_dataset = ConcatDataset([LRSTRDataset(root,syn=False,max_len=20,train=True, args=args) for root in args['train_dataset']])
collect_fn = LRSTR_collect_fn(args=args)
collect_fn_eval = LRSTR_collect_fn(train=False, args=args)
eval_datasets = [LRSTRDataset(root,syn = False, args=args) for root in args['eval_dataset']]
train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True, num_workers=16,
            drop_last=True,collate_fn=collect_fn)
# loss_pixel = ImageLoss()
loss_pixel = TextFocusLoss(args)
loss_ctc =  nn.CTCLoss(blank=0, reduction='mean',zero_infinity=True)
base_params = filter(lambda p: id(p) not in sr_param_list,model.parameters())
sr_params = filter(lambda p: id(p) in sr_param_list,model.parameters())
charset = get_charset(args['charset'])
label_converter = strLabelConverter(get_charset(args['charset']))
calculate_ssim = SSIM()
max_acc = 0.0
eval(model, eval_datasets)
