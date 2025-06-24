import logging

from sklearn.metrics        import precision_recall_curve
from sklearn.metrics        import precision_recall_fscore_support as prfs
import numpy as np

def ap_opt_f1b(labs, probs):
    """
    Calculate average precision (AP) and optimal F1-based score for given labels and probabilities.

    This function computes the precision-recall curve from the provided binary labels and
    associated probability scores. It calculates the area under the precision-recall curve as an
    approximation of average precision. Additionally, it determines the maximum F1-based score
    (F1b) and identifies the corresponding threshold(s) that yield this optimal score.

    Parameters:
    - labs: array-like, shape = [n_samples]
        True binary labels in range {0, 1}.

    - probs: array-like, shape = [n_samples]
        Predicted probability scores for the positive class.

    Returns:
    - ap: float
        Approximate average precision calculated as the area under the precision-recall curve.
    - f1b_opt: float
        Maximum F1-based score derived from the intersection of precision and recall values.
    - opt_thrs: ndarray
        Array containing threshold(s) that yield the optimal F1-based score.
    """
    pre_, rec_, thr_ = precision_recall_curve(labs, probs, pos_label=1)
    ap = abs((pre_[:-1] * np.diff(rec_)).sum())

    f1b_ = (pre_ + rec_)[:-1]
    f1b_ = (2 * pre_ * rec_)[:-1] / np.where(f1b_ != 0, f1b_, 1)
    f1b_[~np.isfinite(f1b_)] = -np.inf 
    f1b_opt = f1b_.max()
    opt_thrs = np.unique(thr_[f1b_ == f1b_opt])

    return ap, f1b_opt, opt_thrs

def tilewise_accuracy(lab_masks,
                      seg_probs,
                      prob_thr=None,
                      cls_agg='max',
                      verbose=False):
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

    # Reshape for consistency
    if lab_masks.ndim==3:
        lab_masks = lab_masks[:,np.newaxis]
    if seg_probs.ndim==3:
        seg_probs = seg_probs[:,np.newaxis]

    assert lab_masks.shape == seg_probs.shape, f'lab_masks + seg_probs shape mismatch ({lab_masks.shape} vs. {seg_probs.shape})'
    assert lab_masks.shape[1] == 1, f'multi channel images of dim={lab_masks.shape[1]} not supported currently'

    # number of images
    n_img = lab_masks.shape[0]
    # class label placeholder
    cls_labs = np.zeros(n_img,dtype=np.uint8)
    # class pred placeholder
    cls_prob = np.ones(n_img,dtype=np.float32)*np.nan
    # segmentation tp,fp,fn,tn placeholder
    seg_ntp,seg_nfp,seg_nfn,seg_ntn = np.zeros([4,n_img],dtype=np.int32)
    # segmentation pos,neg placeholder
    seg_pos,seg_neg = np.ones([2,n_img],dtype=np.float32)*np.nan
    # segmentation AP and threshold placeholder
    seg_ap,seg_thr = np.ones([2,n_img])*np.nan
    # segmentation performance placeholder
    seg_prf = np.ones([n_img,3])*np.nan

    # Aggregation method for converting segmentation maps to classifications
    if cls_agg=='max':
        aggf = np.nanmax
    elif cls_agg=='mean':
        aggf = np.nanmean
    elif cls_agg=='median':
        aggf = np.nanmedian

    # The tile's classification label is positive if the label mask has anything
    cls_labs = lab_masks.any(axis=(3,2,1))

    for i in range(n_img):
        seg_labs = lab_masks[i].ravel()==1
        seg_prob = seg_probs[i].ravel()

        cls_prob[i] = aggf(seg_prob)

        # Segmentation metrics here are only calculated on tiles that have plumes
        if cls_labs[i]==1:
            seg_ap[i], f1b_opt, opt_thrs = ap_opt_f1b(seg_labs,seg_prob)

            if verbose or len(opt_thrs)>1:
                logging.info(f'pos image {i}/{n_img}: cls_prob={cls_prob[i]:8.6f}, seg_f1b={f1b_opt:8.6f} @ thr={opt_thrs}')

            seg_thr[i] = opt_thrs[-1]
            seg_pred = seg_prob >= seg_thr[i]
            seg_errs = seg_labs != seg_pred
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

    # classification metrics for all tiles
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

    return seg_out, cls_out