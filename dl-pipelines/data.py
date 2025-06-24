import csv

import torch
from torchvision    import transforms

import cmutils
import cmutils.pytorch as cmtorch



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
        rotang = [(-rotd,rotd)]+[(ra-rotd,ra+rotd) for ra in [90,180,270]]
        rottrn = [transforms.RandomAffine(rang,translate=(tdel,tdel))
                  for rang in rotang]
        rottrn = transforms.RandomChoice(rottrn)

        augment += [
            transforms.RandomApply(transforms=[rottrn],p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
        ]

    if crop:
        augment += [transforms.CenterCrop(crop)]

    preproc = transforms.Compose(preproc)
    augment = transforms.Compose(augment)

    return preproc, augment

def build_dataloader(csv_path,
                     root='/',
                     train=True,
                     batch_size=8,
                     crop=None,
                     normmax=4000,
                     num_workers=4):
    """
    Build a dataloader for segmentation tasks

    Parameters:
    - csv_path (str): Path to the CSV file containing image paths and labels.
    - root (str): Root directory for relative image paths. Defaults to '/' for absolute paths.
    - train (bool): Whether to apply training-specific augmentations. Defaults to True.
    - batch_size (int): Number of samples per batch. Defaults to 8.
    - crop (Optional[int]): Crop size for images, if applicable. Defaults to None.
    - normmax (float): Maximum value for normalization. Defaults to 4000.
    - num_workers (int): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
    - dataloader (DataLoader): A PyTorch DataLoader object containing the dataset.
    - lab_counts (list[int]): List with counts of positive and negative labels.
    """
    # Load data CSV
    datarows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            datarows.append(row)

    # Calculate loss weights to deal with imbalanced dataset
    all_labels = [1 if int(r[1]) == 1 else 0 for r in datarows]
    n = len(all_labels)
    npos = sum(all_labels)
    nneg = n - npos
    lab_counts = [npos, nneg]

    # Define transforms and dataset class
    mean, std = 0, normmax
    ch4min = 0
    ch4max = normmax

    # Define dataset
    dataset = cmtorch.SegmentDatasetCH4(
        root,
        datarows,
        *get_augment(mean, std, crop, train=train, 
                     ch4min=ch4min, ch4max=ch4max)
    )

    # Define dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )

    # Return dataloader and label counts for loss weighting
    return dataloader, lab_counts