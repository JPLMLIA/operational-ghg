""" Segmentation experiment script

This script was designed to run on supercomputers/clusters.

v1: 2023-06-06 jakelee - fresh rewrite with cmutils

Jake Lee, jakelee, jake.h.lee@jpl.nasa.gov
"""
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

import numpy                as np

import torch
from torch.nn       import BCEWithLogitsLoss
from torch.optim    import Adam
from torchinfo import summary

import cmutils
from unet import DeepPaddedUNet
from data import build_dataloader
from metrics import tilewise_accuracy

import wandb

# constant random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class FocalLoss(torch.nn.Module):
    """
    A PyTorch module implementing the Focal Loss function.

    The Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing training on hard negatives. It modifies
    the standard binary cross-entropy loss with a modulating factor.

    Parameters:
    - alpha (float): Balancing factor for positive/negative examples, defaults to 0.25.
    - gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted, defaults to 2.0.
    - pos_weight (Tensor, optional): A weight for positive examples in BCEWithLogitsLoss.

    Methods:
    - forward(x, y): Computes the focal loss between predictions `x` and targets `y`.

    Returns:
    - Tensor: The computed focal loss value as a tensor.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.bce_loss = BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, x, y):
        bce_loss = self.bce_loss(x,y)
        focal_loss = self.alpha * (1-torch.exp(-bce_loss))**self.gamma * bce_loss
        return focal_loss


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
    parser.add_argument('--epochs',         type=int,
                                            default=200,
                                            help="Epochs for training (default=200)")
    parser.add_argument('--batch',          type=int,
                                            default=8,
                                            help="Batch size for model training (default=8)")
    parser.add_argument('--outroot',        default="train_out/",
                                            help="Root output directory path (default=./train_out/)")
    parser.add_argument('--warmup',         type=int,
                                            default=1,
                                            help="Number of initial warmup epochs to exclude from logs (default=1)")
    parser.add_argument('--gpu',            type=int,
                                            default=0,
                                            help="Specify GPU index to use")
    parser.add_argument('--weight-file',    help="Restore model weights from existing .pt file")    
    parser.add_argument('--verbose',        action='store_true',
                                            help="verbose output")

    args = parser.parse_args()

    # SETUP ####################################################################

    # Set up output directories and files
    traincsv_parts = Path(args.traincsv).parts
    if len(traincsv_parts) >= 2:
        trainid = traincsv_parts[-2] + '_' + Path(traincsv_parts[-1]).stem
    else:
        trainid = traincsv_parts[-1].stem

    # define wandb projname
    projname  = f"{trainid}_sigmoid_unetdeep_{args.pool}pool_ch4_crop256_clsaggmax"

    # add timestamp to local expname
    expname = f"seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{projname}"

    cmutils.check_mkdir(args.outroot)
    outdir = op.join(args.outroot, expname)
    cmutils.check_mkdir(outdir)
    cmutils.check_mkdir(op.join(outdir, 'weights'))

    # Training progress CSV files headers and paths
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
                                                train=True,
                                                batch_size=args.batch,
                                                normmax=args.norm_max)

    val_loader, _ = build_dataloader(args.valcsv,
                                     root=args.dataroot,
                                     train=False,
                                     batch_size=args.batch,
                                     normmax=args.norm_max)


    # MODEL ####################################################################

    ## Load Model
    in_ch = 1 # single channel CMF input
    unetkws = dict(in_ch=in_ch,
                   num_classes=1, # plume=positive, everything else=negative
                   upsample_pad=False,
                   pool=args.pool)

    model = DeepPaddedUNet(**unetkws).to(device)

    if args.weight_file and op.exists(args.weight_file):
        logging.info(f"loading weight_file={args.weight_file}")
        model.load_state_dict(torch.load(args.weight_file,map_location=device))

    model_summary = summary(model, input_size=(args.batch, in_ch, 256, 256))
    loss_weight = args.cls_weight # cls_loss vs. seg_loss convex mixing term

    assert (loss_weight>=0.0) and (loss_weight <= 1.0)

    pos_cls = lab_counts[1]/lab_counts[0] # nneg/npos frequency
    pos_seg = max(1.25,pos_cls/10.0)

    seg_scalef = 256 ** 2
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

    ## Loss Functions
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
                                            cls_agg='max',
                                            verbose=args.verbose)
        
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
            
    wandb.finish()

    # EVALUATION ###############################################################
    model.eval()

    ## Evaluation dataloader, training set performance
    dataloader, _ = build_dataloader(args.traincsv,
                                     root=args.dataroot,
                                     train=False,
                                     batch_size=args.batch,
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
                                            cls_agg='max',
                                            verbose=args.verbose)
    
    logging.info(f"Epoch {epoch} tile train (n={len(train_labs)}):")
    
    # Write tilewise class predictions to CSV
    with open(op.join(outdir, 'train_tile_predictions.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'prob'])
        for path, labimg, probimg in zip(train_path, train_labs, train_prob):
            writer.writerow([path, int(np.nanmax(labimg)), np.nanmax(probimg)])

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
                                     train=False,
                                     batch_size=args.batch,
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
                                          cls_agg='max',
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
