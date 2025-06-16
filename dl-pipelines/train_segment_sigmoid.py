""" Segmentation experiment script

This script was designed to run on supercomputers/clusters.

v1: 2023-06-06 jakelee - fresh rewrite with cmutils

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""
import sys
import os
import os.path as op
import csv
import json
import argparse
import time
import random
import logging
from datetime       import datetime
from pathlib import Path
from collections import OrderedDict
from tqdm           import tqdm

from sklearn.metrics        import precision_recall_curve
from sklearn.metrics        import precision_recall_fscore_support as prfs
import matplotlib.pyplot    as plt
import numpy                as np

import torch
from torch.nn       import BCEWithLogitsLoss
from torch.optim    import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision    import transforms
from torchinfo import summary

from archs.unet import UNetLite, PaddedUNet, DeepPaddedUNet

import cmutils
import cmutils.pytorch as cmtorch

import wandb

# constant random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, x, y):
        bce_loss = self.bce_loss(x,y)
        focal_loss = self.alpha * (1-torch.exp(-bce_loss))**self.gamma * bce_loss
        return focal_loss

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, smooth=0.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth        

    def forward(self, inputs, targets):
        #flatten label and prediction tensors
        pos_pred = torch.sigmoid(inputs).view(-1)
        pos_labs = targets.view(-1)
        neg_pred = 1-pos_pred
        neg_labs = 1-pos_labs
        
        #True Positives, False Positives & False Negatives
        TP = (pos_pred * pos_labs).sum() + self.smooth
        FP = (pos_pred * neg_labs).sum()
        FN = (neg_pred * pos_labs).sum()
        
        tversky_loss  = 1.0 - (TP / (TP + self.alpha*FP + self.beta*FN))                                       
        focal_tversky = tversky_loss**self.gamma

        return focal_tversky
    
def get_augment(mean, std, crop, train=False, rotd=0.25, tdel=0.0125,
                ch4min=0, ch4max=4000, rgbmin=0, rgbmax=20):
    """Define dataset preprocessing and augmentation"""

    preproc = [
        cmtorch.ClampMethaneTile(ch4min=ch4min, ch4max=ch4max,
                                 rgbmin=rgbmin, rgbmax=rgbmax),
        transforms.Normalize(mean, std)
    ]
    
    augment = []
    if train:
        # rotation \pm rotd
        # rotd < 0.5 => 0% increase in nodata px for dense (256,256) img
        # (x,y) translation \pm (tdel,tdel)*img_size
        # tdel=0.0125 => \pm 6.4px translation for (512,512) img
        # tdel=0.0125 => \pm 3.2px translation for (256,256) img
        # tdel=0.0125 => \pm 1.6px translation for (128,128) img
        rotang = [(-rotd,rotd)]+[(ra-rotd,ra+rotd) for ra in [90,180,270]]
        rottrn = [transforms.RandomAffine(rang,translate=(tdel,tdel))
                  for rang in rotang]
        rottrn = transforms.RandomChoice(rottrn)

        augment += [
            #transforms.RandomChoice(rottrn),
            transforms.RandomApply(transforms=[rottrn],p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ]

    if crop:
        augment += [transforms.CenterCrop(crop)]

    preproc = transforms.Compose(preproc)
    augment = transforms.Compose(augment)

    return preproc, augment

def build_dataloader(csv_path, root='/', norm=None, train=True,
                     batch_size=8, crop=None, normmax=4000,
                     subsample=None, num_workers=4):
    """ Build a pytorch dataloader based on provided arguments
    
    These preprocessing steps include calculated mean and standard deviation
    values for the _training set_ of each tiled dataset.

    csv_path: str
        Path to csv path to be loaded for dataloader
    root: str
        Root path to be prepended to paths inside csv
    norm: str
        Key to define normalization statistics
    train: bool
        Whether to shuffle the dataset. Defaults to True.

    """

    # Load data CSV
    datarows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            datarows.append(row)
    
    # Calculate loss weights to deal with imbalanced dataset
    all_labels = [1 if int(r[1]) == 1 else 0 for r in datarows]
    n = len(all_labels)
    npos = sum(all_labels)
    nneg = n - npos
    if subsample in ('balance','pos_only'):
        all_labels = np.int32(all_labels)
        posidx, = np.where(all_labels==1)
        negidx, = np.where(all_labels!=1)
        nneg = npos if subsample=='balance' else max(10,len(negidx)//npos)
        useidx = np.r_[posidx,np.random.permutation(negidx)[:nneg]]
        datarows = [datarows[idx] for idx in useidx]
        
    all_labels = [1 if int(r[1]) == 1 else 0 for r in datarows]

    
    lab_counts = [npos, nneg]

    ch4min = 0
    ch4max = 4000
    
    # Define transforms and dataset class
    # Single channel configurations
    if norm == "COVID_QC":
        mean, std = cmutils.COVID_CH4_NORM
    elif norm == "CalCH4_v8":
        mean, std = cmutils.CALCH4_CH4_NORM
    elif norm == "Permian_QC":
        mean, std = cmutils.PERMIAN_CH4_NORM
    elif norm == "ANG_default":
        mean, std = cmutils.MULTI_CH4_NORM
    elif norm == 'UNIT':
        mean, std = 0, normmax
        ch4max = normmax
    elif norm == 'CENTER':
        mean, std = 0, 2*normmax
        ch4min = -normmax
        ch4max =  normmax
    else:
        raise Exception(f"Undefined normalization: {norm}")

    dataset = cmtorch.SegmentDatasetCH4(
        root,
        datarows,
        *get_augment(mean, std, crop, train=train, 
                     ch4min=ch4min, ch4max=ch4max)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )

    return dataloader, lab_counts

def ap_opt_f1b(labs,probs):
    pre_,rec_,thr_ = precision_recall_curve(labs,probs,pos_label=1)
    ap = abs((pre_[:-1]*np.diff(rec_)).sum())

    f1b_ = (pre_+rec_)[:-1]
    f1b_ = (pre_*rec_)[:-1]/np.where(f1b_!=0,f1b_,1)
    f1b_[~np.isfinite(f1b_)] = -np.inf 
    f1b_opt = f1b_.max()
    opt_thrs = np.unique(thr_[f1b_==f1b_opt])

    return ap,f1b_opt,opt_thrs

def tilewise_accuracy(lab_masks,seg_probs,prob_thr=None,cls_agg='max',
                      verbose=False,wandb_hists=True):
    """
    Summary: given an array of label images and matching array of
    pixelwise probability scores:
    - convert pixelwise labels + predictions to tilewise labels + predictions
    - evalutate prfs on resulting tilewise labels + predictions

    Both input arrays are 4-d arrays with matching dimensions = 
    [n_imgs,1,n_rows,n_cols]

    Arguments:
    - lab_masks: uint8 array of binary pixelwise labels \in {0,1}
    - seg_probs: float32 array of pixelwise probability scores \in [0,1]
    - prob_thr: minimum pixelwise probability for positive class membership
    - cls_agg: method to convert pixelwise class probabilities to tilewise class probability (max,median,mean)
    Output:
    - cls_out: tilewise classification metrics + metadata
    - seg_out: pixelwise segmentation metrics + metadata
    """

    if lab_masks.ndim==3:
        lab_masks = lab_masks[:,np.newaxis]
    if seg_probs.ndim==3:
        seg_probs = seg_probs[:,np.newaxis]
    
    assert lab_masks.shape == seg_probs.shape, f'lab_masks + seg_probs shape mismatch ({lab_masks.shape} vs. {seg_probs.shape})'
    assert lab_masks.shape[1] == 1, f'multi channel images of dim={lab_masks.shape[1]} not supported currently'

    hist_bins = np.linspace(0,1,50+1)
    n_img = lab_masks.shape[0]
    img_npix = np.prod(lab_masks.shape[-2:])
    cls_labs = np.zeros(n_img,dtype=np.uint8)
    cls_prob = np.ones(n_img,dtype=np.float32)*np.nan
    seg_ntp,seg_nfp,seg_nfn,seg_ntn = np.zeros([4,n_img],dtype=np.int32)
    seg_pos,seg_neg = np.ones([2,n_img],dtype=np.float32)*np.nan
    seg_ap,seg_thr = np.ones([2,n_img])*np.nan
    seg_prf = np.ones([n_img,3])*np.nan

    if cls_agg=='max':
        aggf = np.nanmax
    elif cls_agg=='mean':
        aggf = np.nanmean
    elif cls_agg=='median':
        aggf = np.nanmedian

    # tile label = positive iff lab_mask contains any positive labels
    cls_labs = lab_masks.any(axis=(3,2,1))
    
    for i in range(n_img):
        seg_labs = lab_masks[i].ravel()==1
        seg_prob = seg_probs[i].ravel()

        cls_prob[i] = aggf(seg_prob)
        
        if cls_labs[i]==1:
            seg_ap[i],f1b_opt,opt_thrs = ap_opt_f1b(seg_labs,seg_prob)
            
            if verbose or len(opt_thrs)>1:
                logging.info(f'pos image {i}/{n_img}: cls_prob={cls_prob[i]:8.6f}, seg_f1b={f1b_opt:8.6f} @ thr={opt_thrs}')

            seg_thr[i] = opt_thrs[-1]
            seg_pred = seg_prob>=seg_thr[i]
            seg_errs = seg_labs!=seg_pred
            seg_ntp[i] = (~seg_errs &  seg_labs).sum()
            seg_nfn[i] = ( seg_errs &  seg_labs).sum()
            seg_nfp[i] = ( seg_errs & ~seg_labs).sum()
            seg_ntn[i] = (~seg_errs & ~seg_labs).sum()            

            seg_prf[i] = prfs(seg_labs,seg_pred,pos_label=1,average='binary',
                              zero_division=np.nan)[:-1]

            seg_pos[i] = aggf(seg_prob[seg_labs])
            if not seg_labs.all():
                seg_neg[i] = aggf(seg_prob[~seg_labs])
        else:
            seg_neg[i] = aggf(seg_prob)
            #if verbose:
            #    logging.info(f'neg image {i}/{n_img}: cls_prob={cls_prob[i]:8.6f}')

    # todo: handle nfp/ntn for negative tiles
    if 0:
        neg_nfp,neg_ntn = np.ones([2,n_img])*np.nan
        pos_thr = np.nanmedian(seg_thr)
        neg_idx = np.where(cls_labs==0)[0]
        for i in neg_idx:
            neg_prob = seg_probs[i].ravel()
            neg_labs = lab_masks[i].ravel()==1
            neg_pred = neg_prob>=pos_thr
            neg_errs = neg_labs!=neg_pred            
            neg_nfp[i] = ( neg_errs & ~neg_labs).sum()
            neg_ntn[i] = (~neg_errs & ~neg_labs).sum()

    # classification metrics 
    pre_,rec_,thr_ = precision_recall_curve(cls_labs,cls_prob,
                                            pos_label=1)
    cls_thr = prob_thr
    if cls_thr is None:
        f1b_ = pre_+rec_
        f1b_ = 2*(pre_*rec_)/np.where(f1b_!=0,f1b_,1)
        cls_thr = thr_[np.argmax(f1b_)]
            
    cls_pred = cls_prob>=cls_thr
    cls_errs = cls_pred != cls_labs
    cls_ntp = (~cls_errs &  cls_labs).sum()
    cls_nfp = ( cls_errs & ~cls_labs).sum()
    cls_nfn = ( cls_errs &  cls_labs).sum()
    cls_ntn = (~cls_errs & ~cls_labs).sum()

    cls_ap = -np.sum(pre_[:-1]*np.diff(rec_))
    cls_prf = prfs(cls_labs,cls_pred,pos_label=1,average='binary',
                   zero_division=np.nan)[:-1]

    cls_pos = cls_prob[cls_labs]
    cls_neg = cls_prob[~cls_labs]
    cls_pos = cls_pos[np.isfinite(cls_pos)]
    cls_neg = cls_neg[np.isfinite(cls_neg)]
    cls_out = dict(ap=cls_ap,
                   prob_thr=cls_thr,
                   pre=cls_prf[0],
                   rec=cls_prf[1],
                   f1b=cls_prf[2],
                   ntp=cls_ntp,
                   nfp=cls_nfp,
                   nfn=cls_nfn,
                   ntn=cls_ntn,
                   npos=cls_ntp+cls_nfn,
                   nneg=cls_ntn+cls_nfp,
                   prob_pos=np.nanmedian(cls_pos),
                   prob_neg=np.nanmedian(cls_neg),
    )

    if wandb_hists:
        cls_out['hist_pos'] = wandb.Histogram(cls_pos)
        cls_out['hist_neg'] = wandb.Histogram(cls_neg)

    
    # segmentation metrics = median results **on positive tiles only**
    seg_prf = np.nanmedian(seg_prf,axis=0)
    seg_pos = seg_pos[np.isfinite(seg_pos)]
    seg_neg = seg_neg[np.isfinite(seg_neg)]
    seg_out = dict(ap=np.nanmedian(seg_ap),
                   prob_thr=np.nanmedian(seg_thr),
                   pre=seg_prf[0],
                   rec=seg_prf[1],
                   f1b=seg_prf[2],
                   ntp=np.nansum(seg_ntp),
                   nfp=np.nansum(seg_nfp),
                   nfn=np.nansum(seg_nfn),
                   ntn=np.nansum(seg_ntn),
                   npos=np.nansum(seg_ntp+seg_nfn),
                   nneg=np.nansum(seg_ntn+seg_nfp),
                   prob_pos=np.nanmedian(seg_pos),
                   prob_neg=np.nanmedian(seg_neg),
    )
    
    if wandb_hists:
        seg_out['hist_pos'] = wandb.Histogram(seg_pos)
        seg_out['hist_neg'] = wandb.Histogram(seg_neg)
    
    return seg_out,cls_out

# helper class for combined wandb and cmdline logging
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model on tiled methane data.")

    parser.add_argument('traincsv',         help="Filepath of the training set CSV")
    parser.add_argument('valcsv',           help="Filepath of the validation set CSV")
    parser.add_argument('--dataroot',       help="Root directory for relative paths. Defaults to / for absolute paths.",
                                            type=str,
                                            default='/')
    parser.add_argument('--norm',           choices=["UNIT","CENTER","CalCH4_v8", "COVID_QC", "Permian_QC", "ANG_default"],
                                            default="UNIT",
                                            help="Method or campaign id to use for normalization (default=UNIT)")
    parser.add_argument('--norm-max',       type=float,
                                            help="Max value for UNIT+CENTER normalization (default=4000.0)",
                                            default=4000.0)
    parser.add_argument('--pool',           choices=["max","average"],
                                            default="average",
                                            help="Use max or average pooling in Unet model (default=average)")
    parser.add_argument('--lr',             type=float,
                                            help="Learning rate (default=0.003)",
                                            default=0.003)
    parser.add_argument('--cls-weight',     type=float,
                                            default=0.5,
                                            help="Multitask Tilewise (cls) vs Pixelwise (seg) loss weight (default 0.5)")
    parser.add_argument('--pos-weight',     type=float,
                                            help="Positive class weight (default=n_neg/n_pos tiles)")
    parser.add_argument('--epochs',         type=int,
                                            default=200,
                                            help="Epochs for training (default=200)")
    parser.add_argument('--crop',           type=int,
                                            default=256,
                                            help="Center crop dims (default=256)")
    parser.add_argument('--batch',          type=int,
                                            default=8,
                                            help="Batch size for model training (default=8)")
    parser.add_argument('--outroot',        default="train_out/",
                                            help="Root output directory path (default=./train_out/)")
    parser.add_argument('--cls-agg',        default="max",
                                            help="Aggregate pixelwise seg probs -> tilewise class prob / tile using (max | mean) pooling (default=max)")    
    parser.add_argument('--warmup',         type=int,
                                            default=1,
                                            help="Number of initial warmup epochs to exclude from logs (default=1)")
    parser.add_argument('--gpu',            type=int,
                                            default=0,
                                            help="Specify GPU index to use")
    parser.add_argument('--upsample-pad',   action='store_true',
                                            help="Pad upsampling layers instead of cropping")
    parser.add_argument('--use-tversky',    action='store_true',
                                            help="Use tversky-based loss")
    parser.add_argument('--use-full',       action='store_true',
                                            help="Use full (rather than lite) paddedunet")
    parser.add_argument('--use-deep',       action='store_true',
                                            help="Use deep (rather than lite) paddedunet")
    parser.add_argument('--use-sched',      action='store_true',
                                            help="Use scheduler")
    parser.add_argument('--weight-file',    help="Restore model weights from existing .pt file")    
    parser.add_argument('--verbose',        action='store_true',
                                            help="verbose output")
    
    args = parser.parse_args()

    # SETUP ####################################################################

    # Set up output directories and files
    # Set up output directories and files
    traincsv_parts = Path(args.traincsv).parts
    if len(traincsv_parts) >= 2:
        trainid = traincsv_parts[-2] + '_' + Path(traincsv_parts[-1]).stem
    else:
        trainid = traincsv_parts[-1].stem

    # define wandb projname
    projname  = f"{trainid}_sigmoid"
    if args.upsample_pad:
        projname += "_fwdpad"
    
    if args.use_deep:
        projname += "_unetdeep"
    elif args.use_full:
        projname += "_unetfull"
    else:
        projname += "_unetlite"

    if args.pool != 'max':
        projname += f'_{args.pool}pool'
        
    projname += f"_ch4_crop{args.crop}_clsagg{args.cls_agg}"

    # add timestamp to local expname
    expname = f"seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{projname}"
    
    cmutils.check_mkdir(args.outroot)
    outdir = op.join(args.outroot, expname)
    cmutils.check_mkdir(outdir)
    cmutils.check_mkdir(op.join(outdir, 'weights'))

    # Training progress CSV files headers and paths
    # TODO: Given time, switch to tensorboard
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(op.join(outdir, 'out.log')),
            #logging.StreamHandler(),
            TqdmLoggingHandler()
        ]
    )

    
    batch_losses = [["epoch", "batch", "loss"]]
    train_epoch_losses = [["epoch", "mean train loss"]]
    val_epoch_losses = [["epoch", "mean val loss"]]

    outbatchcsv = op.join(outdir, "batch_losses.csv")
    outepochcsv = op.join(outdir, "epoch_losses.csv")
    outvalcsv = op.join(outdir, "val_losses.csv")

    # Device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

 
    # DATA #####################################################################

    # Get dataloaders and loss weights
    train_loader, lab_counts = build_dataloader(args.traincsv,
                                                root=args.dataroot,
                                                norm=args.norm,
                                                train=True,
                                                batch_size=args.batch,
                                                crop=args.crop,
                                                normmax=args.norm_max)

    val_loader, _ = build_dataloader(args.valcsv,
                                     root=args.dataroot,
                                     norm=args.norm,
                                     train=False,
                                     batch_size=args.batch,
                                     crop=args.crop,
                                     normmax=args.norm_max)


    # MODEL ####################################################################

    ## Load Model
    in_ch = 1 # single channel CMF input
    unetkws = dict(in_ch=in_ch,
                   num_classes=1, # plume=positive, everything else=negative
                   upsample_pad=args.upsample_pad,
                   pool=args.pool)
    if args.use_deep:
        model = DeepPaddedUNet(**unetkws).to(device)
    elif args.use_full:
        model = PaddedUNet(**unetkws).to(device)
    else:
        model = UNetLite(**unetkws).to(device)

    if args.weight_file and op.exists(args.weight_file):
        logging.info(f"loading weight_file={args.weight_file}")
        model.load_state_dict(torch.load(args.weight_file,map_location=device))
        
    model_summary = summary(model, input_size=(args.batch, in_ch, args.crop, args.crop))
    loss_weight = args.cls_weight # cls_loss vs. seg_loss convex mixing term

    assert (loss_weight>=0.0) and (loss_weight <= 1.0)
    
    if args.pos_weight:
        pos_cls = args.pos_weight
    else:
        pos_cls = lab_counts[1]/lab_counts[0] # nneg/npos frequency
    pos_seg = max(1.25,pos_cls/10.0)
    
    if args.use_tversky:
        seg_scalef = 1.0
        alpha, gamma = 1.0, 2.0 # focal loss balance/exponent
        logging.info(f"cls_loss=TverskyLoss(alpha={alpha},beta={pos_cls},gamma=1.0)")
        logging.info(f"seg_loss=FocalTverskyLoss(alpha={alpha},beta={pos_seg},gamma={gamma})")
        logging.info(f"loss={loss_weight}*cls_loss + (1-{loss_weight})*seg_loss")
    else:
        seg_scalef = args.crop ** 2
        alpha, gamma = 0.25, 2.0 # focal loss balance/exponent
        logging.info(f"cls_loss=BCEWithLogitsLoss(pos_weight={pos_cls})")
        logging.info(f"seg_loss=FocalLoss(alpha={alpha},gamma={gamma},pos_weight={pos_seg})")
        logging.info(f"loss={loss_weight}*cls_loss + (1-{loss_weight})*seg_loss")

    
    # dump args + model summary
    argsdict = vars(args)
    argsdict['__file__'] = __file__
    argsdict['expname'] = expname
    argsdict['lab_counts'] = lab_counts
    argsdict['alpha'] = alpha
    argsdict['gamma'] = gamma
    argsdict['loss_weight'] = loss_weight
    argsdict['pos_cls'] = pos_cls
    argsdict['pos_seg'] = pos_seg
    argsdict['seg_scalef'] = seg_scalef

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=projname,

        # track hyperparameters and run metadata
        config=argsdict,
    )    

    #  TODO (BDB, 11/27/23): this may not work for parallel wandb runs 
    argsdict['wandb_run_id'] = wandb.run.id
    argsdict['wandb_run_name'] = wandb.run.name

    # create a symlink from [outdir] to [wandb.run.name] 
    os.symlink(expname,op.join(args.outroot,wandb.run.name))
    
    argsdict['model_summary'] = f'{model_summary}'
    with open(op.join(outdir,'argparse_args.json'), 'w') as fid:
        json.dump(argsdict, fid)

    msumstr=f"Experiment {expname} model summary:\n{model_summary}"
    with open(op.join(outdir,'model_summary.txt'), 'w') as fid:
        print(msumstr,file=fid)

    logging.info(msumstr)

    ## Set up optimizer
    logging.info(f"Training with Adam(lr={args.lr})")
    optimizer = Adam(model.parameters(), lr=args.lr)

    ## Scheduler
    if args.use_sched:
        scheduler = StepLR(optimizer, args.lr, args.epochs)
        
    ## Loss Functions
    if args.use_tversky:
         # cls_loss = pos-weighted tversky loss (w/o focal)
        cls_loss = FocalTverskyLoss(alpha=alpha, beta=pos_cls, gamma=1.0)
        seg_loss = FocalTverskyLoss(alpha=alpha, beta=pos_seg, gamma=gamma)        
    else:
        pos_cls = torch.as_tensor([pos_cls],device=device)
        pos_seg = torch.as_tensor([pos_seg],device=device)
        cls_loss = BCEWithLogitsLoss(pos_weight=pos_cls)
        seg_loss = FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_seg)

    # metrics/outputs to push to wandb / epoch
    log_metrics = ['ap','f1b','pre','rec']
    log_probs = ['prob_pos','prob_neg','prob_thr']
    log_hists = ['hist_pos','hist_neg']

    ap_best = -np.inf
    # TRAIN ####################################################################
    for epoch in range(-abs(args.warmup),args.epochs):
        model.train()
            
        start = time.time()
        epoch_loss = 0
        epoch_loss_seg = 0
        epoch_loss_cls = 0
        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch}: batch 0 loss nan",
                          total=len(train_loader))
        
        train_out_cls,train_tgt_cls = [],[]
        for iter, batch in enumerate(train_pbar):
            inputs = batch['x'].to(device)
            targets = batch['y'].to(device)

            # Standard training without SAM
            optimizer.zero_grad()
            
            outputs = model(inputs)

            nbatch = targets.size(dim=0)

            out_cls = torch.amax(outputs.view(nbatch,-1),dim=1)
            tgt_cls = torch.amax(targets.view(nbatch,-1),dim=1)

            # multitask cls/seg loss if loss weight \notin {0.0,1.0} 
            loss_cls = loss_seg = 0.0
            if loss_weight>0.0:
                loss_cls = cls_loss(out_cls, tgt_cls)
                
            if loss_weight<1.0:
                loss_seg = seg_loss(outputs, targets) * seg_scalef
            
            loss = loss_weight*loss_cls + (1-loss_weight)*loss_seg
            loss.mean().backward()
            
            optimizer.step()

            with torch.no_grad():
                # Keeping track of losses
                epoch_loss += loss.cpu().item()
                epoch_loss_seg += loss_seg
                epoch_loss_cls += loss_cls
                #logging.info(f"epoch {epoch}, batch {iter}/{len(train_loader)}, train loss {loss.cpu()}")
                train_pbar.set_description(f"Epoch {epoch}: batch {iter+1} loss {loss:10.6}")
                if epoch>=0: # skip warmup epochs
                    batch_losses.append([epoch, iter, loss])
                    if args.use_sched:
                        #scheduler(epoch)
                        scheduler.step()
                        
                    train_out_cls += out_cls.cpu().tolist()
                    train_tgt_cls += tgt_cls.cpu().tolist()
                    
        # End of training for epoch
        epoch_ctime = time.time()-start
        logging.info(f"Epoch {epoch}: compute time {epoch_ctime} seconds")

        epoch_loss = epoch_loss / len(train_loader)
        if loss_weight>0.0:
            epoch_loss_cls = epoch_loss_cls.cpu().item() / len(train_loader)
        if loss_weight<1.0:
            epoch_loss_seg = epoch_loss_seg.cpu().item() / len(train_loader)
        logging.info(f"Epoch {epoch}: train loss {epoch_loss}")

        if epoch < 0:
            continue
        
        train_epoch_losses.append([epoch, epoch_loss])
        
        # Validation at each epoch
        model.eval()
        
        val_epoch_loss = 0
        val_epoch_loss_cls = 0
        val_epoch_loss_seg = 0
        val_labs,val_prob = [],[]
        val_pbar = tqdm(val_loader,
                        desc=f"Epoch {epoch}: computing val loss",
                        total=len(val_loader))
        for iter, batch in enumerate(val_pbar):
            inputs = batch['x'].to(device)
            targets = batch['y'].to(device)

            with torch.no_grad():
                # Note that using eval model disables aux returns
                outputs = model(inputs)
                nbatch = targets.size(dim=0)

                loss_cls = loss_seg = 0.0
                if loss_weight>0.0:
                    # convert labels/preds from pixelwise to tilewise / sample
                    out_cls = torch.amax(outputs.view(nbatch,-1),dim=1)
                    tgt_cls = torch.amax(targets.view(nbatch,-1),dim=1)
                    loss_cls = cls_loss(out_cls, tgt_cls)

                if loss_weight<1.0:
                    loss_seg = seg_loss(outputs, targets) * seg_scalef
                    
                loss = loss_weight*loss_cls + (1.0-loss_weight)*loss_seg
                
                batch_prob  = torch.sigmoid(outputs)
                val_labs   += targets.cpu().tolist()
                val_prob   += batch_prob.cpu().tolist()
    
                val_epoch_loss += loss
                val_epoch_loss_cls += loss_cls
                val_epoch_loss_seg += loss_seg

        # End of validation for epoch
        val_epoch_loss = val_epoch_loss.cpu().item() / len(val_loader)
        if loss_weight>0.0:
            val_epoch_loss_cls = val_epoch_loss_cls.cpu().item() / len(val_loader)
        if loss_weight<1.0:
            val_epoch_loss_seg = val_epoch_loss_seg.cpu().item() / len(val_loader)
        logging.info(f"Epoch {epoch}: val loss {val_epoch_loss}")
        val_epoch_losses.append([epoch, val_epoch_loss])


        # Calculate test set metrics
        val_labs,val_prob = np.float32(val_labs),np.float32(val_prob)
        val_seg,val_cls = tilewise_accuracy(val_labs, val_prob,
                                            cls_agg=args.cls_agg,
                                            verbose=args.verbose)
        # val_ap,val_thr,val_prfs = val_tiles['cls_ap'],val_tiles['cls_thr'],val_tiles['cls_prfs']
        # logging.info(f"Epoch {epoch}: tilewise val: AP={val_ap}, thr={val_thr}")
        # logging.info(f"Epoch {epoch}: tilewise val: F1={val_prfs[2]}, pre={val_prfs[0]}, rec={val_prfs[1]}")
        
        logdata = OrderedDict()
        logdata['epoch_loss/loss_train'] = epoch_loss
        logdata['epoch_loss/loss_train_seg'] = epoch_loss_seg
        logdata['epoch_loss/loss_train_cls'] = epoch_loss_cls
        logdata['epoch_loss/loss_val'] = val_epoch_loss
        logdata['epoch_loss/loss_val_seg'] = val_epoch_loss_seg
        logdata['epoch_loss/loss_val_cls'] = val_epoch_loss_cls
        
        for key in log_metrics:
            logdata['val_metrics/'+key+'_cls'] = val_cls[key]
            logdata['val_metrics/'+key+'_seg'] = val_seg[key]
            key_mean = val_cls[key]+val_seg[key]
            if key_mean!=0:
                key_mean = 2*(val_cls[key]*val_seg[key])/key_mean
            logdata['val_metrics/'+key+'_mean'] = key_mean
        for key in log_probs:
            logdata['val_probs/'+key+'_cls'] = val_cls[key]
            logdata['val_probs/'+key+'_seg'] = val_seg[key]
        for key in log_hists:
            logdata['val_hists/'+key+'_cls'] = val_cls[key]
            logdata['val_hists/'+key+'_seg'] = val_seg[key]

        wandb.log(logdata)

        # Save weights for best ap_mean score
        ap_mean = logdata['val_metrics/ap_mean']
        if (epoch + 1) > 5 and ap_mean > ap_best:
            logging.info(f"Epoch {epoch}: new best ap_mean {ap_mean}")
            weightpath = op.join(outdir, 'weights', f"apbest_{expname}_weights.pt")
            torch.save(model.state_dict(), weightpath)
            ap_best = ap_mean
        
        # Save weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            weightpath = op.join(outdir, 'weights', f"{epoch}_{expname}_weights.pt")
            torch.save(model.state_dict(), weightpath)
            
        # Update loss curve plot with this epoch
        # TODO: tensorboard pls
        fig, ax = plt.subplots()
        ax.plot(range(epoch+1), np.array(train_epoch_losses[1:])[:,1], label='Train')
        ax.plot(range(epoch+1), np.array(val_epoch_losses[1:])[:,1], label='Val')
        ax.grid()
        ax.legend()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("CE Loss")
        ax.set_title(f"{expname} Loss Curve")
        fig.savefig(op.join(outdir, 'loss_curve.png'), dpi=300)
        plt.close(fig)


    wandb.finish()
    # EVALUATION ###############################################################

    # Write out all losses
    with open(outbatchcsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(batch_losses)
    
    with open(outepochcsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train_epoch_losses)
    
    with open(outvalcsv, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(val_epoch_losses)

    model.eval()

    ## Evaluation dataloader, training set performance
    dataloader, _ = build_dataloader(args.traincsv,
                                     root=args.dataroot,
                                     norm=args.norm,
                                     train=False,
                                     batch_size=args.batch,
                                     crop=args.crop,
                                     normmax=args.norm_max)

    # Pred training data set
    train_path = []
    train_labs,train_prob = [],[]
    for iter, batch in tqdm(enumerate(dataloader),
                            desc="Computing train preds"):
        inputs = batch['x'].to(device)
        targets = batch['y'].to(device)

        with torch.no_grad():
            train_path += batch['xpath']
            batch_prob  = torch.sigmoid(model(inputs))
            train_labs += targets.cpu().tolist()
            train_prob += batch_prob.cpu().tolist()
            
    # Calculate training set metrics
    train_labs,train_prob = np.float32(train_labs),np.float32(train_prob)    
    train_seg,train_cls = tilewise_accuracy(train_labs, train_prob,
                                            cls_agg=args.cls_agg,
                                            verbose=args.verbose)
    
    logging.info(f"Epoch {epoch} tile train (n={len(train_labs)}):")
    
    # Write tilewise class predictions to CSV
    with open(op.join(outdir, 'train_tile_predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'prob'])
        for path, labimg, probimg in zip(train_path, train_labs, train_prob):
            writer.writerow([path, int(np.nanmax(labimg)), np.nanmax(probimg)])

    
    #train_ap,train_thr,train_prfs = train_tiles['cls_ap'],train_tiles['cls_thr'],train_tiles['cls_prfs']    
    #logging.info(f"Epoch {epoch} tile train (n={len(train_labs)}): AP={train_ap}, thr={train_thr}")
    #logging.info(f"Epoch {epoch} tile train (n={len(train_labs)}): F1={train_prfs[2]}, pre={train_prfs[0]}, rec={train_prfs[1]}")

    logdata = OrderedDict()
    for key in log_metrics:
        logdata['train_metrics/cls_'+key] = train_cls[key]
        logdata['train_metrics/seg_'+key] = train_seg[key]
    for key in log_probs:
        logdata['train_probs/cls_'+key] = train_cls[key]
        logdata['train_probs/seg_'+key] = train_seg[key]

    for key,val in logdata.items():
        logging.info(f'{key}: {val}')

    # Test set dataloader
    dataloader, _ = build_dataloader(args.valcsv,
                                     root=args.dataroot,
                                     norm=args.norm,
                                     train=False,
                                     batch_size=args.batch,
                                     crop=args.crop,
                                     normmax=args.norm_max)

    # Pred test data set
    test_path = []
    test_labs,test_prob = [],[]
    for iter, batch in tqdm(enumerate(dataloader),
                            desc="Computing val preds"):
        inputs = batch['x'].to(device)
        targets = batch['y'].to(device)

        with torch.no_grad():
            test_path  += batch['xpath']                        
            batch_prob  = torch.sigmoid(model(inputs))
            test_labs  += targets.cpu().tolist()
            test_prob  += batch_prob.cpu().tolist()

    logging.info(f"Epoch {epoch} tile test (n={len(test_labs)}):")
            
    # Calculate test set metrics wrt optimal threshold wrt training data
    test_labs,test_prob = np.float32(test_labs),np.float32(test_prob)
    test_seg,test_cls = tilewise_accuracy(test_labs, test_prob,
                                          prob_thr=train_cls['prob_thr'], 
                                          cls_agg=args.cls_agg,
                                          verbose=args.verbose)

    # Write tilewise class predictions to CSV
    with open(op.join(outdir, 'test_tile_predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'prob'])
        for path, labimg, probimg in zip(test_path, test_labs, test_prob):
            writer.writerow([path, int(np.nanmax(labimg)), np.nanmax(probimg)])

    logdata = OrderedDict()
    for key in log_metrics:
        logdata['test_metrics/cls_'+key] = test_cls[key]
        logdata['test_metrics/seg_'+key] = test_seg[key]
    for key in log_metrics:
        logdata['test_probs/cls_'+key] = test_cls[key]
        logdata['test_probs/seg_'+key] = test_seg[key]

    for key,val in logdata.items():
        logging.info(f'{key}: {val}')
