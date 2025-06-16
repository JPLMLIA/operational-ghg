import sys
import os
import os.path as op
import argparse
import logging
from pathlib import Path


import matplotlib.pyplot    as pl
import numpy                as np

import torch
from torchvision    import transforms

import rasterio
from archs.unet import PaddedUNet as UCNet
from archs.unet import DeepPaddedUNet as DeepUCNet

import cmutils
import cmutils.pytorch as cmtorch

def extrema(a):
    return np.nanmin(a),np.nanmax(a)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a flightline saliency map with a pre-trained UCnet model."
    )

    parser.add_argument('flightline',       help="Filepaths to flightline ENVI IMG.",
                                            type=str)
    parser.add_argument('--norm',           choices=["UNIT"],
                                            default="UNIT",
                                            help="Normalization method for preprocessing")
    parser.add_argument('--norm-max',       type=float,
                                            help="Max value for UNIT+CENTER normalization",
                                            default=4000.0)    
    parser.add_argument('--clobber',        action="store_true",
                                            help='Overwrite existing outputs')
    parser.add_argument('--band', '-n',     help="Band to read if multiband",
                                            default=4,
                                            type=int)
    parser.add_argument('--salmin',         help="Min salience to plot",
                                            default=0.5,
                                            type=float)
    parser.add_argument('--salmax',         help="Max salience to plot",
                                            default=1.0,
                                            type=float)
    parser.add_argument('--weights', '-w',  help="Weight file to use for prediction.")
    parser.add_argument('--arch', '-a',     help="Arch to use for prediction.",
                                            choices=["UCNet", "DeepUCNet"],
                                            default='DeepUCNet')
    parser.add_argument('--pool',           help="Does model use max or average pooling?",
                                            choices=["max", "average"],
                                            default='max')
    parser.add_argument('--upsample-pad',   action='store_true',
                                            help="Does model pad upsampling layers instead of cropping?")
    parser.add_argument('--gpus', '-g',     help="GPU devices for inference. -1 for CPU.",
                                            nargs='+',
                                            default=[-1],
                                            type=int)
    parser.add_argument('--outroot', '-o',   help="Output directory for generated salience maps.",
                                            default="pred_out",
                                            type=str)

    args = parser.parse_args()

    # SETUP ####################################################################

    # Set up output directories and files
    #expname = f"UCNetpred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.norm}"

    #outdir = op.join(args.outroot, expname)
    outdir = op.join(args.outroot, op.split(op.splitext(args.weights)[0])[1])
    #cmutils.check_mkdir(args.outroot)
    cmutils.check_mkdir(outdir)

    outf = op.join(outdir,f"{Path(args.flightline).stem}_ucnetsaliency.img")
    figf = op.join(outdir,f"{Path(args.flightline).stem}_ucnetsaliency.png")

    if op.exists(outf) and not args.clobber:
        print(f'{outf} exists, exiting')
        sys.exit(1)

    # TODO: Given time, switch to tensorboard
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(op.join(outdir, 'out.log')),
            logging.StreamHandler()
        ]
    )


    # MODEL ####################################################################

    logging.info("Model Initialization...")
    weightpath = args.weights
    if op.isfile(weightpath):
        logging.info(f"Found {weightpath}.")
    else:
        logging.info(f"Model not found at {weightpath}, exiting.")
        sys.exit(1)


    logging.info("Initializing pytorch device.")
    if args.gpus == [-1]:
        # CPU
        device = torch.device('cpu')
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # apple m1/m2
            device = torch.device("mps")
        else:
            if not torch.cuda.is_available():
                logging.error("CUDA not found, exiting.")
                sys.exit(1)
        
            # Set first device
            device = torch.device(f"cuda:{args.gpus[0]}")
    logging.info(f"Using device: {device}")


    unetkws = dict(in_ch=1,
                   num_classes=1,
                   upsample_pad=args.upsample_pad,
                   pool=args.pool)
    
    ulayer = 'up1'
    logging.info("Loading model")
    if args.arch == "UCNet":
        model = UCNet(**unetkws)
        dlayer = 'down4'
        div = 16 #  TODO (BDB, 04/03/24): double check div=16 with full model 
    elif args.arch == "DeepUCNet":
        model = DeepUCNet(**unetkws)
        dlayer = 'down5'
        div = 32
    model.load_state_dict(torch.load(weightpath,map_location=device))

    model = model.to(device)
    
    model.eval()


    # DATA #####################################################################

    if args.norm=='UNIT':
        transform = transforms.Compose([
            cmtorch.ClampScaleMethaneTile(ch4max=args.norm_max)
        ])
    else:
        raise Exception("Only UNIT normalization available currently")

    logging.info(f"Reading and preprocessing {args.flightline}")
    dataset = rasterio.open(args.flightline)
    cmf = dataset.read(args.band)
    msk = cmf!=-9999
    if dataset.count==4:
        rgb = np.clip(dataset.read([1,2,3])/15,0,1)
        rgb = np.uint8(rgb*255).transpose([1,2,0])
        rgb = np.dstack([rgb,msk*255])
    else:
        rgb = np.zeros(list(cmf.shape)+[4],dtype=np.uint8)
        rgb[...,-1] = 255

    logging.info(f'CMF shape: {dataset.shape}')
    logging.info(f'CMF original (min,max): {cmf[msk].min(),cmf.max()}')

    #  TODO (BDB, 04/03/24): infer receptive field dims instead of hard coding it! 
    dim = 256

    tdiv = dim//div
    div_ax0 = div - cmf.shape[0] % div
    div_ax1 = div - cmf.shape[1] % div

    pad_ax0 = (div_ax0 // 2, (div_ax0 // 2) + (div_ax0 % 2))
    pad_ax1 = (div_ax1 // 2, (div_ax1 // 2) + (div_ax1 % 2))

    x = np.pad(cmf, (pad_ax0, pad_ax1), constant_values=0)
    x = np.expand_dims(x, axis=[0,1])
    x = torch.tensor(x, dtype=torch.float).to(device)
    x = transform(x)
    logging.info(f'CMF clipped  (min,max): {x.min(),x.max()}')

    # INFERENCE ################################################################

    logging.info("Computing pixelwise salience")
    activation = {}
    with torch.no_grad():
        sal = model(x)
        sal = torch.nn.functional.sigmoid(sal)
        sal = sal.cpu().detach().numpy().squeeze()
        
    sal = sal[pad_ax0[0]:-pad_ax0[1],pad_ax1[0]:-pad_ax1[1]]

    logging.info(f'Salience (min,max): {sal[msk].min(),sal.max()}')  

    sal[~msk] = -9999

    logging.info("Writing salience image")
    with rasterio.Env():
        profile = dataset.profile

        if 'blockysize' in profile:
            del profile['blockysize']
        if 'interleave' in profile:
            del profile['interleave']

        profile.update(
            #driver='GTiff',compress='lzw',
            dtype=rasterio.float32,
            count=1
        )
        
        logging.info(f"Saving to {outf}")
        with rasterio.open(outf, 'w', **profile) as dst:
            dst.write(sal.astype(rasterio.float32), 1)

    # CMF / SALIENCE QUICKLOOK #################################################
    logging.info(f"Rendering CMF + Salience quicklook")
    cmfmin = 0
    cmfmax = 1500
    
    salmin = args.salmin
    salmax = args.salmax

    interpolation='nearest'    

    salpx = sal[msk]
    pxmin,pxmax = extrema(salpx)
    if pxmax < salmin:
        salmin = pxmin
        salmin = pxmax
    
    cmf[~msk | (cmf<=cmfmin)] = np.nan
    sal[~msk | (sal<salmin)] = np.nan
    nrows,ncols = cmf.shape[0],cmf.shape[1]
    if nrows > ncols:
        rgb = rgb.transpose(1,0,2)
        cmf = cmf.T
        sal = sal.T
        #rcen = rcen[:,[1,0]]
    #censz = min(nrows,ncols)*0.1

    aspect = cmf.shape[1]/cmf.shape[0]

    figrows,figcols,figscale=2,1,5
    figsize=(aspect*figcols*figscale,figrows*figscale*1.05)
    fig,ax = pl.subplots(figrows,figcols,figsize=figsize,
           	         sharex=True,sharey=True)

    ax[0].imshow(rgb,interpolation=interpolation)
    ax[0].imshow(cmf,vmin=cmfmin,vmax=cmfmax,cmap='YlOrRd',
                 interpolation=interpolation)

    ax[1].imshow(rgb,interpolation=interpolation)
    ax[1].imshow(sal,vmin=salmin,vmax=salmax,cmap='RdYlBu_r',
                 interpolation=interpolation)
    

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    pl.tight_layout()
    pl.subplots_adjust(bottom=0.01,top=0.99,
                       left=0.01,right=0.99,
                       hspace=0.01,wspace=0.01)
    logging.info(f"Saving quicklook to {figf}")
    pl.savefig(figf)
    pl.close(fig)
            
    logging.info("Done!")
