#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docstring for imagesampler.py
Docstrings: http://www.python.org/dev/peps/pep-0257/
"""
from __future__ import absolute_import, division, print_function
from warnings import warn

import sys, os, numbers, shutil

from glob import glob

import matplotlib as mpl
mpl.use('agg')


#import cartopy.feature as cfeature

sys.path.append(os.path.join(os.getenv('earthimage'),'src'))

from imageutil import *
from earthimage import *
from sentinelimage import SentinelImage, EDDY, NOTEDDY

from cmfimage import CMFImage, posrgb, negrgb, bgrgb, locrgb
from cmfimage import POINTSRC, DIFFSRC, FALSESRC, LOCSRC

import pylab as pl
randperm = np.random.permutation

from skimage.transform import downscale_local_mean as _downscale
from skimage.morphology import remove_small_objects

try:
    from scipy.ndimage import find_objects as _findobj
    from scipy.ndimage import center_of_mass as _cmass
except:
    from scipy.ndimage.measurements import find_objects as _findobj
    from scipy.ndimage.measurements import center_of_mass as _cmass
    
import fiona

from collections import OrderedDict

basedir = pathsplit(__file__)[0]
configdir = pathjoin(basedir,'config')
gconfigf = pathjoin(configdir,'defaults.cfg')

STRATIFY_TILEDIM = 64 # tile image into dxd tiles 

# max % nodata pixels for a labeled negative/random bg tile to be accepted
MAX_NODATA_NEG = 0.33

# max % nodata pixels for a labeled positive tile to be accepted
# note: use a large value to permit labeled samples near image boundaries
MAX_NODATA_POS = 0.80 

def stratified_split(y,**kwargs):
    try:
        from sklearn.model_selection import train_test_split as _trtesplit
    except:
        from sklearn.cross_validation import train_test_split as _trtesplit

    print(f'stratified_split on {len(y)} labels, {len(set(y))} unique classes')
    defaults = (('train_size',0.5),('stratify',y),('random_state',42))
    for key,val in defaults:
        kwargs.setdefault(key,val)

    inds = np.asarray(np.arange(len(y)),dtype=np.int32)
    tridx,teidx = _trtesplit(inds,**kwargs)
    return tridx,teidx

def group_split(y,**kwargs):
    from sklearn.model_selection import GroupShuffleSplit
    defaults = (('train_size',0.5),('random_state',42))
    for key,val in defaults:
        kwargs.setdefault(key,val)
    
    gss = GroupShuffleSplit(n_splits=1, **kwargs)
    inds = np.asarray(np.arange(len(y)),dtype=np.int32)
    tridx,teidx = next(gss.split(inds,**kwargs))
    return tridx,teidx

def bwdist(bwimg,metric='chessboard',**kwargs):
    if metric=='euclidean':
        from scipy.ndimage.morphology import distance_transform_edt as _bwdist
    elif metric in ('chessboard','taxicab'):
        from scipy.ndimage.morphology import distance_transform_cdt as _bwdist
    kwargs.setdefault('return_distances',True)
    kwargs.setdefault('return_indices',False)
    return _bwdist(bwimg,**kwargs)
    
# tile filter functions
def tile_data_dense(label_map,max_nodata,**kwargs):
    # true if tile contains more than p% valid (not nan/nodata) pixels
    nodata_percent = np.count_nonzero(label_map==NODATA)/float(label_map.size)
    return nodata_percent < max_nodata

def tile_some_pos(label_map,labels_pos,**kwargs):
    # true if tile is partially labeled positive (labels=positive labels)
    max_nodata = kwargs.get('max_nodata',MAX_NODATA_POS)
    if not tile_data_dense(label_map,max_nodata,**kwargs):
        return False
    mask_pos = np.isin(label_map,labels_pos)
    return mask_pos[label_map!=NODATA].any()

def tile_none_pos(label_map,labels_pos,**kwargs):
    # true if tile is fully labeled negative or bg (labels=negative labels)
    # label_pos = any label that is not (negative|unlabeled|nodata)
    #mask_notpos = np.isin(label_map,labels_neg+[NOLABEL])
    max_nodata = kwargs.get('max_nodata',MAX_NODATA_NEG)
    if not tile_data_dense(label_map,max_nodata,**kwargs):
        return False
    mask_pos = np.isin(label_map,labels_pos)
    return not (mask_pos[label_map!=NODATA].any())

def find_pairs(imgpaths, labpaths, imgsuffix='_img', labsuffix='_mask.png'):
    # provided imgfile in imgdir, labfile in labdir, assumes matches are:
    #   pathjoin(imgdir,imgfile) ->
    #   pathjoin(labdir,imgfile+labsuffix)

    img_names,lab_names = [],[]

    imgmatch = set([])
    for labpath in labpaths:
        labimgfile = pathsplit(labpath)[1].replace(labsuffix,imgsuffix)
        if labimgfile in imgmatch:
            # duplicate label image filenames in different label directories
            continue
        for imgidx,imgpath in enumerate(imgpaths):
            imgfile = pathsplit(imgpath)[1]
            if imgfile in imgmatch:
                # duplicate image filenames in different image directories
                continue                
            if imgfile == labimgfile:
                img_names.append(imgpath)
                lab_names.append(labpath)
                imgmatch.add(imgfile)
                imgpaths = np.delete(imgpaths,imgidx)
                break
        
    return img_names,lab_names

def labels2centroids(label_map,labels,labrad=3,**kwargs):
    """
    labels2centroids(label_map,labels,**kwargs)

    Summary: converts label_map for specified labels into centroid_map,
    with each labeled connected component represented as a 3x3 crosshair

    Arguments:
    - label_map: label image containing labels
    - labels: labels to convert to centroids

    Keyword Arguments:
    - None

    Output:
    - centroid_map: new label map w/ labeled regions for each labeled
    connected component with a label value in provided labels list
    """

    from skimage.morphology import diamond
    # ijlab = pixel offsets to label for crosshair @ roi center of mass
    # diamond permits 8-conn translation in the N/S/E/W + NE/NW/SE/SW
    ijlab = np.c_[np.where(diamond(labrad))]-labrad
    centroid_map = np.ones_like(label_map)*NOLABEL
    data_mask = label_map!=NODATA
    centroid_map[~data_mask] = NODATA
    for labval in labels:
        labrois = imlabel(label_map==labval)
        roiobjs = _findobj(labrois)
        for roiidx,roiobj in enumerate(roiobjs):
            # get center of mass of labeled pixels in roiobj bbox
            roimsk = labrois[roiobj]==(roiidx+1)
            imin,jmin = roiobj[0].start,roiobj[1].start
            idif,jdif = roiobj[0].stop-imin,roiobj[1].stop-jmin
            
            iroi,jroi = np.int32(_cmass(roimsk))
            iroi,jroi = iroi+imin,jroi+jmin
            # label centroid + ijlab pixel offsets within image extent
            iroi,jroi = iroi+ijlab[:,0],jroi+ijlab[:,1]
            ijmsk = (iroi>=0) & (iroi<label_map.shape[0]) & \
                    (jroi>=0) & (jroi<label_map.shape[1]) 
            centroid_map[iroi[ijmsk],jroi[ijmsk]] = labval

    # replace pixel rois with labels in labels with labeled centroids
    label_new = label_map.copy()
    label_mask = np.isin(label_map,labels) | np.isin(centroid_map,labels)
    label_mask = label_mask & data_mask
    label_new[label_mask] = centroid_map[label_mask]
    
    return label_new

class CoordSampler():
    """
    CoordSampler(object)
    """
    def __init__(self,data,label_map,labels,min_area=2,**kwargs):
        self.data = data
        self.labels = labels
        self.label_map = label_map.copy()
        self.mask = self.label_map!=NODATA

        # label_mask tracks locations of labels to be sampled
        # statis mask, never updated
        self.label_mask = np.isin(self.label_map,labels)

        # coord_mask tracks which locations have been sampled already
        # updated on each sample() call
        self.coord_mask = self.label_mask & self.mask

        #if all([key in kwargs for key in ('imgbase','name')]):
        #    outimgf = kwargs['imgbase']+'_'+kwargs['name']+'_mask.png'
        #    imsave(outimgf,np.uint8(self.coord_mask*255))
        
        self.min_area = min_area
            
        self.ncoord = np.count_nonzero(self.coord_mask)
        self.ndat = np.count_nonzero(self.mask)
        self.roi_map = []

    def sample(self,n_samples,replace=False,stratify='tiles',
               stratify_tiledim=128,**kwargs):
        """
        sample(self,n_samples)
        
        Summary: collects n_samples 2d points from coord_mask, excluding
          previously sampled locations to sample w/o replacement 
        
        Arguments:
        - n_samples: number of samples
        Keyword Arguments:
        - replace: sample with replacement
        - stratify: partition image by "rois" (labeled connected components) or  by "tiles" of dims stratify_tiledim^2
        - stratify_tiledim: pixel size of square tiles used to partition image when stratify='tiles'
        
        Output:
        - output
        """
        if n_samples<=0:
            return np.empty((0,), dtype=np.uint32), \
                np.empty((0,), dtype=np.uint32)
        
        sample_size = n_samples
        if stratify in ('rois','tiles') and self.min_area > 1:
            # remove any connected components smaller than min_area
            print(f'removing small objects (area<{self.min_area}px) in coord_mask')
            print(f'{np.count_nonzero(self.coord_mask)} coords before')
            self.coord_mask = remove_small_objects(np.array(self.coord_mask,dtype=np.bool_),
                                                   connectivity=2,
                                                   min_size=self.min_area)
            self.coord_mask = self.coord_mask!=0
            print(f'{np.count_nonzero(self.coord_mask)} coords after')
            self.ncoord = np.count_nonzero(self.coord_mask)
        
        # use coord mask to exclude previous samples / nodata regions
        y_idx,x_idx = np.nonzero(self.coord_mask)            
        if len(y_idx) == 0:
            return y_idx,x_idx
        idx_labs = self.label_map[y_idx,x_idx]
        
        sample_size = min(sample_size,self.ncoord)        
        rand_idx = randperm(len(x_idx))
        y_idx,x_idx = y_idx[rand_idx],x_idx[rand_idx]
        
        idx_labs = idx_labs[rand_idx]
        print(f'sampling {sample_size} of {self.ncoord} coordinates (stratify={stratify})')        
        def stratified_sample(roi_labs,sample_size):
            try:
                rand_idx,extra_idx = stratified_split(roi_labs,train_size=0.5)
                n_extra=sample_size-len(rand_idx)
                if n_extra>0:
                    rand_idx = np.r_[rand_idx,extra_idx[:n_extra]]
            except:
                nlabs = len(roi_labs)
                ulabs,labcounts = np.unique(roi_labs,return_counts=True)
                num_rois = len(ulabs)
                print(f'stratifying w {num_rois} rois in self.coord_mask')                
                roi_prob = np.zeros(nlabs)
                for ul,nl in zip(ulabs,labcounts):
                    roi_prob[roi_labs==ul] = nlabs/float(nl)
                roi_prob /= roi_prob.sum()
                rand_idx = np.random.choice(nlabs,size=sample_size,p=roi_prob,
                                            replace=(sample_size>nlabs))                    
            return rand_idx

        if stratify=='rois':
            if len(self.roi_map) == 0:
                self.roi_map = imlabel(self.coord_mask)
                self.roi_map[~self.mask] = 0
                print(f'{len(np.unique(self.tile_map))-1} {stratify_tiledim}^2 tiles in self.tile_map')
                
            roi_labs = self.roi_map[y_idx,x_idx]
            rand_idx = stratified_sample(roi_labs,sample_size)
        elif stratify=='tiles':
            if len(self.roi_map) == 0:
                print(f'computing roi_map with tiledim={stratify_tiledim}')
                self.roi_map,_ = tile_classmap(np.uint8(self.coord_mask),
                                                stratify_tiledim)
                self.roi_map[~self.mask] = 0
                print(f'{len(np.unique(self.roi_map))-1} {stratify_tiledim}^2 tiles in self.roi_map')

            tile_labs = self.roi_map[y_idx,x_idx]
            rand_idx = stratified_sample(tile_labs,sample_size)
        else:
            rand_idx = np.random.choice(y_idx.size, size=sample_size,
                                        replace=False)

        if len(rand_idx) == 0: # crop if our rand index is too long
            return [],[]
        
        rand_idx = rand_idx[:min(len(rand_idx),sample_size)]
        y_samp,x_samp = y_idx[rand_idx], x_idx[rand_idx]

        samp_labs = self.label_map[y_samp,x_samp]
        if not replace:
            # remove y_samp,x_samp from sample mask to exclude from
            # future sample calls
            self.coord_mask[y_samp,x_samp] = 0
            self.ncoord = np.count_nonzero(self.coord_mask)

        # if self.quantize>1:
        #     return np.int32(y_samp*self.quantize), \
        #         np.int32(x_samp*self.quantize)

        return y_samp,x_samp

    def replace(self,coords):
        # previously sampled points = (coord_mask==0) & (mask==1)
        # replaces coords in coord_mask to permit sampling with replacement 
        if len(coords)==0:
            return
        idx = np.asarray(coords)
        assert(idx.ndim==2 and idx.shape[1]==2)
        # if self.quantize>1:
        #     idx = np.unique(idx//self.quantize,axis=0)
        self.coord_mask[idx[:,0],idx[:,1]] = 1
        self.ncoord = np.count_nonzero(self.coord_mask)

    def exclude(self,coords):
        if len(coords)==0:
            return
        idx = np.asarray(coords)
        assert(idx.ndim==2 and idx.shape[1]==2)
        self.coord_mask[idx[:,0],idx[:,1]] = 0
        self.ncoord = np.count_nonzero(self.coord_mask)

    def empty(self):
        return (self.ncoord==0) or (not self.coord_mask.any())
        
class TileSampler(CoordSampler):
    """
    TileSampler(CoordSampler)
    """
    def __init__(self,data,label_map,labels,**kwargs):
        super(TileSampler,self).__init__(data,label_map,labels=labels,**kwargs)
        self.tile_mask = np.zeros_like(self.label_map,dtype=np.int32)
        
    def sample(self,ntiles,tiledim,keep_function,
               max_overlap=0,replace_rejected=False,
               plot_rejected=False,max_trials=10,
               max_patience=4,verbose=False,**kwargs):
        """
        sample(self,image,ntiles,keep_function=tile_dense)
        
        Summary: sample 2d tiles from image with optional rejection function
        
        Arguments:
        - image: EarthImage instance to sample
        - ntiles: number of tiles to sample
        - keep_function: tile keep function
        
        
        Keyword Arguments:
        - max_overlap: keep tiles that partially overlap prior tile samples
        - replace_rejected: resample rejected tiles (default=False)
        
        Output:
        - output
        """
        try:
            if len(tiledim)==1:
                tiledim = (tiledim[0],tiledim[0])
        except:
            tiledim = (tiledim,tiledim)

        if max_overlap==0:
            tile_coords = np.c_[np.where(self.tile_mask==1)]
            self.exclude(tile_coords)
            
        tlabs = np.zeros([ntiles,tiledim[0],tiledim[1]],
                         dtype=self.label_map.dtype)
        twins = np.zeros([ntiles,4],dtype=np.int32)

        nkeep,npatience,nlast,nseen = 0, 1, -1, 0
        rejected = set([])

        # coordsampler instance
        cs = super(TileSampler,self)
        
        # accept/reject tile
        for bi in range(max_trials):
            # num samples / trial 
            nsamp = min(2*ntiles,self.ncoord)

            if nkeep==ntiles or self.empty():
                print(f'  batch {bi} exit nkeep ({nkeep}) + {ntiles} + nreject ({len(rejected)}), patience={npatience}/{max_patience}')
                break
            
            print(f'coordsampler batch={bi}: sampling {nsamp} of {self.ncoord} coords')
            y_samp, x_samp = cs.sample(nsamp,**kwargs)
            coords = list(zip(y_samp,x_samp))
            print(f'coordsampler batch={bi}: sampled {len(coords)} of {nsamp} requested coords')
            for nseen,(row,col) in enumerate(coords):                
                msg = f'nseen ({nseen}) = nkeep ({nkeep}) + nreject ({len(rejected)}), patience={npatience}/{max_patience}'                    

                # center tile on col,row,width,height
                twin = (max(0,col-tiledim[1]//2),
                        max(0,row-tiledim[0]//2),
                        tiledim[1],tiledim[0])
                tmsk = extract_window(self.tile_mask,twin,NOLABEL)
                if tmsk.shape != tiledim:
                    rejected.add((row,col))
                    if verbose:
                        print(f'  tile rejected @ {row,col}: shape={tmsk.shape} != tiledim={tiledim}')
                    continue
                
                # tover = overlap wrt any existing tiles in tilemap
                tover = np.count_nonzero(tmsk)/tmsk.size
                if tover > max_overlap:
                    if verbose:
                        print(f'  tile rejected @ {row,col}: tover={100*tover:.2f}>max_overlap={100*max_overlap:.2f}')
                    rejected.add((row,col))
                    continue

                # tlab = labeled pixels within window
                tlab = extract_window(self.label_map,twin,NODATA)
                tnodata = np.count_nonzero(tlab==NODATA)/tmsk.size
                if tnodata > 0.9: 
                    rejected.add((row,col))
                    if verbose:
                        print(f'  tile rejected @ {row,col}: > 90% NODATA')
                    continue
                # keep track of samples we've kept or skipped
                tkeep = keep_function(tlab)
                if not tkeep:
                    if verbose:
                        print(f'  tile rejected @ {row,col}: tkeep={tkeep}')
                    if plot_rejected:
                        #tdat = self.read_data(w=twin)
                        tdat = extract_window(self.read_data(),twin,NODATA)
                        #rejfigf = 'rejected_tile_row{row}_col{col}.png'
                        figkw = dict(figsize=(3*7.5,1*7.5+0.25),
                                     sharex=True,sharey=True)
                        fig,ax = pl.subplots(3,1,**figkw)
                        plotkw=dict(vmin=0,vmax=1,cmap='RdYlBu_r')
                        ax[0].imshow(tdat,**plotkw)
                        ax[1].imshow(tlab,**plotkw)
                        ax[2].imshow(tmsk,**plotkw)
                        ax[0].set_title(f'rejected tile@{row,col} (keep={tkeep}, overlap={tover})')
                        ax[1].set_title('tile labels')
                        ax[2].set_title('tile mask')
                        pl.tight_layout()
                        #pl.savefig(rejfigf)
                        pl.show()
                        pl.close(fig)
                        
                    rejected.add((row,col))
                    continue
                
                if verbose:
                    print(f'  tile accepted @ {row,col}: tkeep={tkeep}')
                    
                twins[nkeep] = twin
                tlabs[nkeep] = tlab
                nkeep += 1

                # add tile (left,right,bottom,top) region to tile mask
                tl,tr = twin[0],min(self.tile_mask.shape[1],twin[0]+twin[2])
                tb,tt = twin[1],min(self.tile_mask.shape[0],twin[1]+twin[3])
                self.tile_mask[tb:tt,tl:tr] = 1

                # crop tile mask bounds to exclude coords that will be rejected
                if max_overlap>0:
                    rh = int(np.ceil(max_overlap*(tt-tb)/2))
                    rw = int(np.ceil(max_overlap*(tr-tl)/2))
                    tl,tr = tl+rw,tr-rw
                    tb,tt = tb+rh,tt-rh

                # exclude (cropped) tile coords from future batches 
                if tb<tt and tl<tr:
                    self.coord_mask[tb:tt,tl:tr] = 0
                    self.ncoord = np.count_nonzero(self.coord_mask)

                # NOTE (BDB, 06/28/23): this check must occur at loop end
                # since self.empty()==True if all labels are sampled in first
                # iteration of first batch
                if nkeep==ntiles or self.empty():
                    print(f' {nseen} of {len(coords)} coords exit '+msg)
                    break                
                    
            # end sample loop
            if nseen < len(coords):
                # replace any points we didn't consider
                self.replace(coords[nseen:])

            if replace_rejected and len(rejected)!=0:
                # reuse rejected points (necessary if reject_function changes)
                self.replace(list(rejected))
            else:
                # otherwise don't sample these coords again
                self.exclude(list(rejected))

            # keep track if we didn't add any new tiles in this batch
            if nkeep>0 and nkeep != nlast:
                nlast = nkeep
                npatience = 1 # reset patience
            else:
                npatience += 1

            print(f'  nkeep ({nkeep}) + nreject ({len(rejected)}) = nseen ({nseen}), patience={npatience}/{max_patience}')
            if npatience==max_patience:
                print(f'max_patience={max_patience} reached')
                break
        # end batch loop
        
        # truncate tile lists to nkeep max
        if nkeep < ntiles:
            twins = twins[:nkeep]
            tlabs = tlabs[:nkeep]

        return twins, tlabs
            
    def collect(self,image,tilewins):
        nbands = image.count
        tiledim = twins[0][2:]
        tiles = np.zeros([len(tilewins),tiledim[0],tiledim[1],nbands],
                         dtype=image.dtype)
        for ti,twin in enumerate(tilewins):
            tiles[ti] = np.atleast_3d(image.read_data(w=twin))
            
        return tiles

class BinaryTileSampler():
    """
    BinaryTileSampler(object)
    """
    def __init__(self,data,label_map,labels_pos,labels_neg,**kwargs):
        poskw = dict(name='positive',**kwargs)
        negkw = dict(name='negative',**kwargs)
        self.labels_pos = labels_pos
        self.labels_neg = labels_neg        
        self.ts_pos = TileSampler(data,label_map,labels=labels_pos,**poskw)
        self.ts_neg = TileSampler(data,label_map,labels=labels_neg,**negkw)

        self.label_map = self.ts_pos.label_map
        self.mask = self.ts_pos.mask
        self.npos = self.ts_pos.ncoord
        self.nneg = self.ts_neg.ncoord
        self.ndat = self.ts_pos.ndat

    def sample(self,ntiles,tiledim,
               pos_keep_fn,
               neg_keep_fn,
               max_overlap_pos=0,
               max_overlap_neg=0,
               min_percent_pos=0,
               max_percent_pos=100,
               **kwargs):
        assert len(tiledim)==2
            
        pos_windows = np.empty([0,4],dtype=np.int32)
        neg_windows = np.empty([0,4],dtype=np.int32)
        pos_pixlabs = np.empty([0,tiledim[0],tiledim[1]],dtype=np.int32)
        neg_pixlabs = np.empty([0,tiledim[0],tiledim[1]],dtype=np.int32)
        max_samples = ntiles

        # balance positive/negative samples, favoring positives
        pos_samples = max_samples
        if self.nneg != 0:
            pos_ratio = float(self.npos)/max(1,self.npos+self.nneg)
            pos_percent = min(max_percent_pos,100*pos_ratio)
            pos_percent = max(min_percent_pos,pos_percent)
            pos_samples = int(np.ceil(max_samples*(pos_percent/100.0)))
        pos_samples = int(min(self.npos,pos_samples))
        neg_samples = int(min(self.nneg,max_samples-pos_samples))
            
        if pos_samples!=0:
            print(f'\nsampling up to {pos_samples} positive tiles (max_overlap_pos={max_overlap_pos})')                    
            pos_windows, pos_pixlabs = self.ts_pos.sample(pos_samples,tiledim,
                                                          keep_function=pos_keep_fn,
                                                          max_overlap=max_overlap_pos,
                                                          **kwargs)

            # prevent pos/neg tile overlap
            self.ts_neg.exclude(np.c_[np.where(self.ts_pos.tile_mask==1)])
            self.ts_neg.tile_mask[self.ts_pos.tile_mask==1] = 1
            #imsave('pos_tiles.png',np.uint8(self.ts_pos.tile_mask*255))
        
        if neg_samples!=0:
            print(f'\nsampling up to {neg_samples} negative tiles (max_overlap_neg={max_overlap_neg})')            
            neg_windows, neg_pixlabs = self.ts_neg.sample(neg_samples,tiledim,
                                                          keep_function=neg_keep_fn,
                                                          max_overlap=max_overlap_neg,
                                                          **kwargs)
            #imsave('neg_tiles.png',np.uint8(self.ts_neg.tile_mask*255))

            
        # note: always perform this masking whether neg_samples>0 or not
        self.ts_neg.tile_mask[self.ts_pos.tile_mask==1] = NOLABEL

        tile_mask_pos = self.ts_pos.tile_mask
        tile_mask_neg = self.ts_neg.tile_mask
        mask_overlap = (tile_mask_pos==1) & (tile_mask_neg==1)
        if mask_overlap.any():
            pos_overlap = np.count_nonzero(np.isin(self.label_map[mask_overlap],self.labels_pos))
            print(f'WARNING: {np.count_nonzero(mask_overlap)} overlapping pixels (pos_overlap={pos_overlap}) in positive/negative tiles')
            if pos_overlap!=0:
                input()
            
        self.tile_mask = tile_mask_pos - tile_mask_neg        
        
        # [ntile x 4] array tile window bounds
        tile_windows = np.r_[pos_windows,neg_windows]

        # [ntile x tile_dim x tile_dim] array of tile label_map subimages
        tile_pixlabs = np.r_[pos_pixlabs,neg_pixlabs]

        # [ntile x 1] binary vector of image labels (pos=1,neg=0)
        pos_count = len(pos_windows)
        neg_count = len(neg_windows)
        
        pos_imglabs =  np.ones(pos_count,dtype=np.int8)
        neg_imglabs = -np.ones(neg_count,dtype=np.int8)
        tile_imglabs = np.r_[pos_imglabs,neg_imglabs]
        
        return tile_windows, tile_pixlabs, tile_imglabs

    def collect(self,image,twins):
        return self.ts_pos.collect(image,twins)

class ROISampler():
    """
    ROISampler(object)
    """
    def __init__(self,data,label_map,labels,**kwargs):
        self.data = data
        self.label_map = label_map.copy()
        self.mask = ~np.isin(self.label_map,[NOLABEL,NODATA])
        self.roi_labs = imlabel(np.isin(self.label_map,labels) & self.mask)
        self.roi_mask = np.zeros_like(self.roi_labs,dtype=np.bool_)
        self.roi_bbox = _findobj(self.roi_labs)
        self.roi_ulab = np.unique(self.roi_labs)[1:]
        self.nroi = len(self.roi_ulab)
        self.ndat = np.count_nonzero(self.mask)
        
    def sample(self,**kwargs):
        y_samp,x_samp = np.zeros([2,self.nroi])
        for roi_idx,(roi_lab,roi_bbox) in enumerate(zip(self.roi_ulab,
                                                        self.roi_bbox)):
            roi_mask = self.roi_labs[roi_bbox]==roi_lab
            imin,jmin = roi_bbox[0].start,roi_bbox[1].start
            roi_ij = np.int32(_cmass(roi_mask))+[imin,jmin]
            y_samp[roi_idx] = roi_ij[0]
            x_samp[roi_idx] = roi_ij[1]
            
        return y_samp,x_samp

def tile_coverage(tile_mask,labs_mask):
    tile_labs_cover = tile_mask[labs_mask] 
    labs_count = np.count_nonzero(labs_mask)
    labs_intile = np.count_nonzero(tile_labs_cover)
    labs_missed = labs_count - labs_intile
    return labs_count, labs_intile, labs_missed
    
def sample_image(config,imgf,labf,rgbf=None):
    paths,labs,isbg = [],[],[] 

    print(f'imgf: {imgf}')
    print(f'labf: {labf}')
    print(f'rgbf: {rgbf}')

    
    imgbase = basename(imgf)
    tilebase = imgbase+'_tilemap'

    image_class = config['image_class']

    if image_class=='SentinelImage':
        img = SentinelImage(imgf,labf)
        
    elif image_class=='CMFImage':
        img = CMFImage(imgf,labf=labf,rgbf=rgbf)               

    vmin, vmax = img.vmin, img.vmax

    nrows,ncols = img.shape[:2]
        
    labels_pos = img.labels_pos
    labels_neg = img.labels_neg
    labels_bg  = img.labels_bg

    assert([None not in [labels_pos,labels_neg,labels_bg]])

    labels_all  = labels_pos+labels_neg

    labmin = min(labels_all+[NOLABEL])
    labmax = max(labels_all+[NOLABEL])
    
    tiledim     = config['tile_dim']
    nlab_tiles  = config['nlab_tiles']
    nbg_tiles   = config['nbg_tiles']

    max_overlap_lab = config['max_overlap_lab']
    max_overlap_bg  = config['max_overlap_bg']
    min_percent_pos = config['min_percent_pos']
    max_percent_pos  = config['max_percent_pos']

    fig_tpose  = config.get('fig_tpose',False)
    
    # define figure transpose + output size with wrt figh * aspect ratio
    if fig_tpose and img.shape[0] > img.shape[1]:
        # image lines > samples -> transpose image x/y for figure
        img_aspect = img.shape[0]/img.shape[1]
        img_tpose = (1,0)
        rgb_tpose = (1,0,2)
        img_xlabel = 'Lines'
        img_ylabel = 'Samples'
    else:
        # image lines <= samples -> original image x/y for figure
        img_aspect = img.shape[1]/img.shape[0]
        img_tpose = (0,1)
        rgb_tpose = (0,1,2)
        img_xlabel = 'Samples'
        img_ylabel = 'Lines'
            
    mask_shpf   = config.get('mask_shp',None)
    pos_centroids  = config.get('pos_centroids',False)
    neg_centroids  = config.get('neg_centroids',False)

    labels_centroids = []
    if pos_centroids:
        labels_centroids += list(labels_pos)
    if neg_centroids:
        labels_centroids += list(labels_neg)

    # make sure tiledim = tuple([height,width])
    try:
        if len(tiledim)==1:
            tiledim = (tiledim[0],tiledim[0])
        elif len(tiledim)==2:
            tiledim = tuple(tiledim)
        else:
            assert len(tiledim)==2
    except:
        # tiledim scalar
        tiledim = (tiledim,tiledim)

    require_square_tiles = False
    if require_square_tiles:
        tiledim = (max(tiledim),max(tiledim))

    out_path = config['out_path']
    tileimgf = pathjoin(out_path,tilebase+'.png')
    tilefigf = pathjoin(out_path,imgbase+'_tiles.pdf')                         
    
    pos_keep_fn = lambda label_map,**kwargs: tile_some_pos(label_map,labels_pos,**kwargs)
    neg_keep_fn = lambda label_map,**kwargs: tile_none_pos(label_map,labels_pos,**kwargs)
    bg_keep_fn  = lambda label_map,**kwargs: tile_none_pos(label_map,labels_pos,**kwargs)

    base_sampler_params = dict(imgbase=imgbase,outpath=out_path,
                               stratify="tiles",
                               stratify_tiledim=STRATIFY_TILEDIM)
    
    lab_sampler_params = dict(max_overlap_pos=max_overlap_lab,
                              max_overlap_neg=max_overlap_lab,
                              min_percent_pos=min_percent_pos,
                              max_percent_pos=max_percent_pos,
                              **base_sampler_params)

    bg_sampler_params = dict(name='bg', max_overlap=max_overlap_bg,
                             min_area=1, **base_sampler_params)

    # optional land/ocean mask
    if mask_shpf is not None:
        with fiona.open(mask_shpf,"r",
                        crs=img.ds.crs,
                        driver=img.ds.driver) as shapefile:
            mask_geom = [feature["geometry"] for feature in shapefile]

        img_data, mask_transform = rio.mask.mask(img.ds,
                                                 mask_geom,
                                                 invert=False,
                                                 crop=False)
        img_data = img_data.transpose((1,2,0))
        if image_class=='SentinelImage':
            img.mask = img.mask & maskbounds2(~np.isin(img_data,[0,img.nodata,NODATA])) 
            img_data[~img.mask] = NODATA
            img.data = img_data

    # grab data + mask
    img_data = img.read_data().squeeze()
    img_mask = img.read_mask()
    nmask = np.count_nonzero(img_mask)

    # skip this image if it contains no valid pixels
    if nmask==0:
        return dict(paths=paths,labs=labs,isbg=isbg)

    # make sure label_map is the right dtype + nodata pixels are masked
    if img.label_map.dtype != np.int32:
        img.label_map = np.int32(img.label_map)
        img.label_map[~img_mask] = NODATA

    img.label_map_orig = img.label_map.copy()
    # reduce specified label rois to centroids if necessary
    if labels_centroids != []:
        img.label_map = labels2centroids(img.label_map,labels_centroids)

    npos = np.count_nonzero(np.isin(img.label_map,labels_pos))
    nneg = np.count_nonzero(np.isin(img.label_map,labels_neg))
    nlab = npos+nneg
    nunlab = np.count_nonzero(img.label_map==NOLABEL)
    nnodata = np.count_nonzero(img.label_map==NODATA)

    if nlab==0 and nbg_tiles==0:
        # skip this image if it contains no positive or negative labels
        print(f'no labeled pixels to sample and no bg tiles requested, skipping')
        return dict(paths=paths,labs=labs,isbg=isbg)

    print(f'tiledim: {tiledim}')
    print(f'labels_pos {labels_pos} labels_neg {labels_neg} labels_bg {labels_bg}')    
    print(f'img.label_map shape={img.label_map.shape}, values={list(np.unique(img.label_map))}')
    print(f'img.label_map labeled={nlab} (pos={npos}, neg={nneg}), unlabeled={nunlab}, nodata={nnodata}')

    save_debug_images = False
    if save_debug_images:
        debugfigf = pathjoin(out_path,f'{imgbase}_data_labs.pdf')
        figkw = dict(figsize=(15,2*5+0.25),sharex=True,sharey=True)
        fig,ax = pl.subplots(2,1,**figkw)
        ax[0].imshow(img_mask.transpose(img_tpose),cmap='RdYlBu_r',interpolation='nearest')
        ax[1].imshow(img.label_map.transpose(img_tpose),
                     vmin=labmin-1,vmax=labmax,cmap='RdYlBu_r',interpolation='nearest')
        ax[0].set_title(f'data_mask: {nmask} data, {nnodata} nodata')
        ax[1].set_title(f'label_map: {npos} pos, {nneg} neg')
        pl.tight_layout()
        pl.savefig(debugfigf)
        pl.close(fig)
        print(f'saved {debugfigf}')

    # sample positive / negative class tiles
    labts = BinaryTileSampler(img_data,img.label_map,labels_pos,labels_neg,
                              **lab_sampler_params)
    tile_windows, tile_pixlabs, tile_imglabs = \
        labts.sample(nlab_tiles,tiledim,pos_keep_fn,neg_keep_fn,
                     **lab_sampler_params)

    pos_labs_mask = labts.ts_pos.label_mask==1
    neg_labs_mask = labts.ts_neg.label_mask==1
    pos_tile_mask = labts.ts_pos.tile_mask==1
    neg_tile_mask = labts.ts_neg.tile_mask==1

    tile_npos = np.count_nonzero(tile_imglabs)
    tile_nneg = len(tile_imglabs)-tile_npos
    
    if pos_labs_mask.any():
        pos_count, pos_intile, pos_missed = tile_coverage(pos_tile_mask,pos_labs_mask)
        print(f'pos count, intile, missed: {pos_count, pos_intile, pos_missed}')

    if neg_labs_mask.any():
        neg_count, neg_intile, neg_missed = tile_coverage(neg_tile_mask,neg_labs_mask)
        print(f'neg count, intile, missed: {neg_count, neg_intile, neg_missed}')

    tile_nbg = 0
    bg_tile_mask  = np.zeros_like(pos_tile_mask)

    # sample background tiles if necessary
    if nbg_tiles > 0:
        print(f'\nsampling {nbg_tiles} background tiles (max_overlap_bg={max_overlap_bg})')
        # generate a mask where we have no labels and set background labels 1
        # make sure we don't sample any regions in existing tile_mask
        # make sure to exclude nodata pixels in bg sampling
        bg_label_map = img.label_map.copy()
        bg_label_map[~img_mask] = NODATA
        bgts = TileSampler(img_data,bg_label_map,labels=labels_bg,
                           **bg_sampler_params)

        # mask all labeled tiles, exclude positive tile pixels from sampling
        bgts.exclude(np.c_[np.where(pos_tile_mask)])
        bgts.tile_mask[pos_tile_mask | neg_tile_mask] = 1
        #imsave('bg_tiles.png',np.uint8(bg_tile_mask*255))

        # sample background tiles, reset pos|neg tiles in bg_tile mask to NOLABEL afterwards
        bgtile_windows, bgtile_pixlabs = \
            bgts.sample(nbg_tiles,tiledim,bg_keep_fn,**bg_sampler_params)
        bgts.tile_mask[pos_tile_mask | neg_tile_mask] = NOLABEL

        # add bgtile_imglabs similar to BinaryTileSampler()
        bgtile_imglabs = np.zeros(len(bgtile_windows),dtype=np.int8)

        tile_nbg = len(bgtile_imglabs)
        
        if len(bgtile_windows) > 0:
            # concat windows, invert labels
            tile_windows = np.r_[tile_windows,bgtile_windows]
            tile_pixlabs = np.r_[tile_pixlabs,bgtile_pixlabs]
            tile_imglabs = np.r_[tile_imglabs,bgtile_imglabs]
            bg_tile_mask = bgts.tile_mask==1

        #imsave('bg_tiles.png',np.uint8(bg_tile_mask*255))

    if bg_tile_mask.any():
        bg_labs_mask = img_mask & ~(pos_labs_mask|neg_labs_mask)
        bg_count, bg_intile, bg_missed = tile_coverage(bg_tile_mask,bg_labs_mask)                                                
        print(f'bg count, intile, missed: {bg_count, bg_intile, bg_missed}')

    print(f'tile imglabs: {tile_npos} positive, {tile_nneg} negative, {tile_nbg} background')
    print(f'tile windows:\n{tile_windows[:,[0,1]].T}')
        
    # dump geotiffs to imgbase/(pos|neg|bg)
    for tw,tlab,tpixlab in zip(tile_windows,tile_imglabs,tile_pixlabs):
        twstr = f'tile{tw[2]}x{tw[3]}+{tw[0]}+{tw[1]}'
        tposm = np.isin(tpixlab,labels_pos)
        ntpos = np.count_nonzero(tposm)
        if tlab in (1,-1): # tile is not a bg tile
            labcat = 'neg' if tlab==-1 else 'pos'
            if labcat == 'neg' and ntpos!=0:
                print(f'WARNING: negative tile@{tw} contains {ntpos} positive labels! Skipping.')
                continue                
            timglab = tlab
            tbgmask = 0

        else: # bg tile
            labcat  = 'bg'
            if ntpos != 0:
                print(f'WARNING: bg tile@{tw} contains {ntpos} positive labels! Skipping.')
                continue
            timglab = 0 # bg tiles = negative class
            tbgmask = 1

        # tcatdir \in {pos,neg,bg}
        tcatdir = pathjoin(imgbase,labcat)
        if not pathexists(pathjoin(out_path,tcatdir)):
            os.makedirs(pathjoin(out_path,tcatdir))
            
        # NOTE: USE RELATIVE PATH for csv files
        timgf = pathjoin(tcatdir,imgbase+'_'+twstr+'.tif')

        # update csv output lists
        paths.append(timgf)
        labs.append(f'{timglab:.0f}')
        isbg.append(f'{tbgmask:.0f}')        

        # use full path for actual out_tiff dump
        out_tiff = pathjoin(out_path,timgf)
        img.write_window(out_tiff,tw,driver='GTiff',dtype='float32')

        if timglab in labels_centroids:
            # extract label tile from original label mask, not centroid map
            tpixlab = extract_window(img.label_map_orig,tw)
            
        #write png of label mask after removing nodata values
        tpixmsk = tpixlab != NODATA
        tpixlab[~tpixmsk] = NOLABEL

        out_labf = pathjoin(out_path,tcatdir,imgbase+'_'+twstr+'.png')
        imsave(out_labf,np.uint8(tpixlab))

        if rgbf:
            trgb = extract_window(img.read_rgb(),tw)
            out_rgbf = pathjoin(out_path,tcatdir,imgbase+'_'+twstr+'_rgb.png')
            imsave(out_rgbf,trgb)

    # valid data = white
    img_tiles_rgb = np.zeros(list(img.shape[:2])+[3],dtype=np.uint8)

    # fill unlabeled + nodata regions
    img_tiles_rgb[img_mask] = nolabelrgb
    img_tiles_rgb[~img_mask] = nodatargb

    img_tiles_rgb[pos_tile_mask] = posrgb
    img_tiles_rgb[neg_tile_mask] = negrgb
    img_tiles_rgb[bg_tile_mask] =  bgrgb

    imsave(tileimgf,img_tiles_rgb)

    rgbshow = np.uint8(img.read_rgb().transpose(rgb_tpose))
    rgbshow = np.dstack([rgbshow,np.uint8((0.33*img_mask.transpose(img_tpose))*255)])

    figh,figdpi = 5.5,120 # constant figh + dpi for quicklooks
    figw = figh*img_aspect
    if image_class == 'CMFImage':
        figsize = (figw, (3.25*figh)+0.5)
        figkw = dict(figsize=figsize,sharex=True,sharey=True,dpi=figdpi)
        fig,cmfax = pl.subplots(3,1,**figkw)
        # CMFImage imgshow (cmf channel) != rgbshow (cmf rgb spectral channels)
        imgshow = img_data.copy().transpose(img_tpose)
        if img.vmin is not None:
            imgshow[imgshow<=img.vmin] = np.nan                

        cmfax[0].imshow(rgbshow,interpolation='nearest')
        cmfax[0].imshow(imgshow,vmin=img.vmin,vmax=img.vmax,
                        cmap=img.data_cmap,
                        interpolation='nearest')
        cmfax[0].set_title(imgbase,size='x-small')
        ax = cmfax[1:]
    elif image_class == 'SentinelImage':
        # SentinelImage imgshow (sar backscatter) == rgbshow (rgb colormapped sar amplitude)
        figsize = (figw, (2.25*figh)+0.5)
        figkw = dict(figsize=figsize,sharex=True,sharey=True)
        fig,ax = pl.subplots(1,2,**figkw)

    # draw rgb background in all subplots
    for axi in ax:
        axi.imshow(rgbshow,interpolation='nearest')

    labrgbshow = np.uint8(img.label_rgb.transpose(rgb_tpose))
    labalpha = np.isin(img.label_map.transpose(img_tpose),labels_all)
    labrgbshow = np.dstack([labrgbshow,np.uint8(labalpha*255)])

    # tile boundaries = opaque (alpha=1.0), interior=transparent (alpha=0.5)
    any_tile_mask = pos_tile_mask|neg_tile_mask|bg_tile_mask
    tilealpha = 48*any_tile_mask
    #tilealpha = bwdist(any_tile_mask)
    #tedgew = 2 # tile edge width
    #tilealpha[any_tile_mask & (tilealpha> tedgew)] = 64 # tile interior
    #tilealpha[any_tile_mask & (tilealpha<=tedgew)] = 255 # tile edges
    tilergbshow = np.dstack([img_tiles_rgb,np.uint8(tilealpha)])
    
    ax[0].imshow(labrgbshow,interpolation='nearest')
    ax[1].imshow(tilergbshow,interpolation='nearest')

    ax[0].set_title('labels',size='x-small')
    ax[1].set_title('tiles',size='x-small')
    pl.tight_layout()
    pl.savefig(tilefigf)
    pl.close(fig)

    return dict(paths=paths,labs=labs,isbg=isbg)

def read_cfg(scfgfile,gcfgfile,dataset,verbose=True):
    
    import json, configparser

    # ExtendedInterpolation allows "${section:variable}" syntax in .cfg files
    cfg_interpolation = configparser.ExtendedInterpolation()

    if verbose:
        print(f'data config file:    {os.path.abspath(scfgfile)}')
        print(f'default config file: {os.path.abspath(gcfgfile)}')

    config = OrderedDict()

    # use global defaults only when data/site params not present
    if gcfgfile is not None:
        gcfg = configparser.ConfigParser(interpolation=cfg_interpolation)
        gcfg.read(gcfgfile)
        
        for key,defval in gcfg['defaults'].items():
            config[key] = defval.split('#')[0].strip() # dont parse comments
    
    scfg = configparser.ConfigParser(interpolation=cfg_interpolation)
    scfg.read(scfgfile)

    # read defaults first, dataset second to override defaults
    for section in ('defaults',dataset):
        for key,val in scfg[section].items():
            # drop comments after variable defns + strip whitepace
            config[key] = val.split('#')[0].strip()

    # configparser doesn't infer dtypes, so use json.loads to do so               
    for key,val in config.items():
        # json.loads doesn't handle bools or nonetype
        if val in ('True','False'):
            config[key] = val=='True'
        elif val == 'None':
            config[key] = None
        else:
            try: 
                config[key] = json.loads(val)
            except:
                config[key] = val # default unknown dtypes -> strings
        if verbose:
            print(f'  {key}->{config[key]} ({type(config[key])})')
        
    return config

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser('imagesampler.py')

    # keyword arguments
    argparser.add_argument('-v', '--verbose', action='store_true',
    			help='Enable verbose output')
    argparser.add_argument('--clobber', action='store_true',
    			help='Remove existing outputs in out_path')
    argparser.add_argument('--randomize', action='store_true',
    			help='Remove order of inputs before processing')
    argparser.add_argument('--cfgdefs', type=str, default=gconfigf,
    			help='config file containing global defaults')

    # positional arguments     
    argparser.add_argument('cfgfile', type=str, help='data/site specific config file')
    argparser.add_argument('dataset', type=str, help='dataset in site config file')
    
    args = argparser.parse_args()
    
    verbose  = args.verbose
    clobber  = args.clobber
    gcfgfile = args.cfgdefs
    scfgfile = args.cfgfile
    dataset  = args.dataset

    config   = read_cfg(scfgfile,gcfgfile,dataset)

    img_path    = config['img_path']
    img_match  = config['img_match']
    img_suffix  = config['img_suffix']

    lab_path    = config['lab_path']
    lab_match  = config['lab_match']
    lab_suffix  = config['lab_suffix']

    rgb_req     = config['rgb_required']
    rgb_path    = config['rgb_path']
    rgb_match  = config['rgb_match']
    rgb_suffix  = config['rgb_suffix']
    
    out_path    = config['out_path']


    if lab_match != img_match:
        print(f'WARNING: file matching with img_match!=lab_match is experimental')
        input()
    
    # we're sampling unlabeled data if lab_path isn't provided
    if lab_path is None:
        imgmatch = img_match+img_suffix
        print(f'imgmatch: {imgmatch}')
        imgpaths = glob(pathjoin(img_path,imgmatch.replace('**','*')))
        nimg = len(imgpaths)
        print(f'{nimg} img files found')
        labpaths = [None]*len(imgpaths)
        nlab = 0
    else:
        labmatch = lab_match+lab_suffix
        print(f'labmatch: {pathjoin(lab_path,labmatch)}')
        labpaths = glob(pathjoin(lab_path,labmatch.replace('**','*')))
        nlab = len(labpaths)
        print(f'{nlab} lab files found')
        labf2imgf = lambda labf: labf.replace(lab_path,img_path).replace(lab_suffix,img_suffix)        
        imgpaths = [labf2imgf(labf) for labf in labpaths]
        imgpaths = [imgf for imgf in imgpaths if pathexists(imgf)]
        nimg = len(imgpaths)
        print(f'{nimg} img files found')        
        imgpaths,labpaths = find_pairs(imgpaths,labpaths,img_suffix,lab_suffix)
        nlab = len(labpaths)
        nimg = len(imgpaths)
        print(f'{nimg} img + {nlab} lab files matched')

        print(f'imgpaths: {imgpaths}')
        print(f'labpaths: {labpaths}')

    imgf2lid = lambda imgf: pathsplit(imgf)[1].split('_')[0]
    def matchlid(path,lid,prefix,suffix):
        matchptn = (prefix+lid+suffix).replace('**','*')
        lidmatch = glob(pathjoin(path,matchptn))
        if len(lidmatch)==0:
            return ''
        return lidmatch[0]
        
    if rgb_path is None:
        rgbpaths = [None]*len(imgpaths)
    else:
        rgbmatch = rgb_match+rgb_suffix
        rgbpaths = glob(pathjoin(rgb_path,rgbmatch.replace('**','*')))
        #rgbpaths = [matchlid(rgb_path,imgf2lid(imgf),rgb_match,rgb_suffix)
        #            for imgf in imgpaths]
        #rgbpaths = np.array(rgbpaths)
        #rgbpaths = rgbpaths[rgbpaths!='']

        nrgb = len(rgbpaths)
        print(f'{nrgb} rgb files found')

    if rgb_req:
        imglids = np.array([imgf2lid(imgf) for imgf in imgpaths])
        rgblids = np.array([imgf2lid(rgbf) for rgbf in rgbpaths])
        imgkeep = np.isin(imglids,rgblids)
        imgpaths = np.array(imgpaths)[imgkeep]
        labpaths = np.array(labpaths)[imgkeep]
        imglids = imglids[imgkeep]
        rgbidx = []
        for imgi,imglidi in enumerate(imglids):
            rgbidx.append(np.argmax(rgblids==imglidi))
        rgbpaths = np.array(rgbpaths)[rgbidx]

        for imgf,rgbf in zip(imgpaths,rgbpaths):
            print(f'Matched image={imgf}->rgb={rgbf}')

    nimg = len(imgpaths)
    if nimg==0:
        if rgb_req:
            print('No (imgf,labf,rgbf) matches found, check paths + match patterns')
        else:
            print('No (imgf,labf) matches found, check paths + match patterns')
        sys.exit(1)
        
    # format out_path + concat dataset id 
    if out_path.endswith('/'): 
        out_path = out_path[:-1]
    if not out_path.endswith(dataset): 
        out_path = pathjoin(out_path,dataset)

    config['out_path'] = out_path       
    if pathexists(out_path) and clobber:
        print(f'Deleting existing data in {out_path}')
        shutil.rmtree(out_path)
            
    if pathexists(out_path):
        # either clobber==False or we didn't delete existing out_path
        print(f'out_path={out_path} exists and clobber=False, exiting')
        sys.exit(1)

    os.makedirs(out_path)            
    print(f'Saving output to out_path={out_path}')

    if args.randomize:
        print('Randomizing image order')
        ridx = np.random.permutation(nimg)
        imgpaths = np.array(imgpaths)[ridx]
        labpaths = np.array(labpaths)[ridx]
        rgbpaths = np.array(rgbpaths)[ridx]
        
        
    tiles = dict(paths=[],labs=[],isbg=[])
    for imgidx,(imgf,labf,rgbf) in enumerate(zip(imgpaths,labpaths,rgbpaths)):
        print(f'\nProcessing image {imgidx+1} of {nimg}: imgf={basename(imgf)} labf={basename(labf) if labf else labf}')
        imgout = sample_image(config,imgf,labf,rgbf=rgbf)
        for outkey,outtiles in imgout.items():
            tiles[outkey].extend(outtiles)
        
    # label/bg mask files w/ relative image paths
    # dump image_path -> image_label map to file
    labcsvf = pathjoin(out_path,'tile_labs.csv')
    np.savetxt(labcsvf,np.c_[tiles['paths'],tiles['labs']],delimiter=',',
               fmt='%s',header='path,label')

    # dump image_path -> background_mask map to file
    bgmcsvf = pathjoin(out_path,'tile_isbg.csv')
    np.savetxt(bgmcsvf,np.c_[tiles['paths'],tiles['isbg']],delimiter=',',
               fmt='%s',header='path,isbg')
