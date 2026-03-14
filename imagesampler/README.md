# Imagesampler

This is a script that samples tiles of methane plumes and the background from
AVIRIS-NG, GAO, and EMIT flightlines.

POC: Brian Bue, bbue@jpl.nasa.gov

>[!WARNING]
> The original script, `imagesampler.py`, is provided only for algorithmic reference.
> This script relies on several internal repositories with cross-dependencies 
> that are too difficult to release in their entirety.

The algorithmic procedure is described in our preprint: https://arxiv.org/abs/2505.21806

The resulting tiled datasets are available at: https://doi.org/10.5281/zenodo.19011045

Below is the original README for this script.

## Prepare Config File

- Inspect config/defaults.cfg to assess if default parameters relevant to your dataset.
- Create a config/mydataset.cfg that defines a new [dataset\_name] section containing the img\_path and lab\_path where your data and label images are stored, respectively and an out\_path where the output tiles will be stored. Default parameters will be parsed from defaults.cfg. To override values parsed from defaults.cfg, define a [defaults] section in mydataset.cfg and specify new values for the parameters relevant to your dataset 

## Sample Image and Label Tiles 

- Run the following command to start sampling tiles while logging output to dataset_name.out and warnings/errors to dataset\_name.err. 

`python imagesampler.py config/mydataset.cfg dataset\_name 2> dataset\_name.err 1> dataset\_name.log` 

- Alternatively, if you're running on a cluster with the slurm scheduler:

`sbatch -N 1 -c 40 --mem=180G -J imagesampler\_dataset\_name -o dataset\_name.log -e dataset\_name.err --wrap "python imagesampler.py config/mydataset.cfg dataset\_name"`

## Inspect Logs and Output Images

- Ensure all images have been processed by inspecting dataset\_name.log and dataset\_name.err
- Spot check the output image and label tiles saved in out\_path/dataset_name.

## Example Use Case

Provided a set of geotiff images (`img_suffix=_data.tif`) to sample in the `img_path` directory + corresponding png (`lab_suffix=_label.png`) label images located in the `lab_path` directory, sample a set of 256x256 (`tile_dim=[256,256]`) tiles such that the tiles cover all positive & negative class label ROIs and background tiles roughly cover the remaining spatial extent of each (img_path/img_base_data.tif, lab_path/img_base_label.png) pair.

## Config settings:

- sample a single tile centered on each distinct positive label ROI (connected component) in each label image (`pos_centroids=True`)
- sample a single tile centered on each distinct negative label ROI in each label image (`neg_centroids=True`)
- sample up to 100 background tiles from any unlabeled regions of the image (`nbg_tiles=100`).
- no more than 25 labeled (pos+neg) tiles per image (`nlab_tiles=25`)
- do not allow overlap between labeled (positive|negative) vs. other (positive|negative) tiles (`max_overlap_lab=0`)
- allow up to 10% overlap between background vs. (background|negative) tiles (`max_overlap_bg=0.1`)

## ImageSampler Outputs

ImageSampler outputs the following products for each (image,label) pair:

- out\_path/dataset\_name/img\_base/: directory containing sampled image tiles + label masks extracted from image\_file, separated by class (pos|neg) or background (bg). 
- out\_path/dataset\_name/img\_base\_tilemap.png: map showing locations/coverage of sampled tiles in image\_file.
- out\_path/dataset\_name/img\_base\_tilemap.pdf: quicklook showing locations/coverage of sampled tiles in image\_file.

and uses the following naming conventions for images and label tiles:

`[img_base]_tile[txdim]x[tydim]+[txoff]+[tyoff].[img_ext]` 
`[img_base]_tile[txdim]x[tydim]+[txoff]+[tyoff].[lab_ext]` 

where 

- [img\_base][.img\_ext] are the base filename + file extension (if any) of the image data file from which the tile was sampled.
- [img\_base][.lab\_ext] are the base filename + file extension of the label file matching [img\_base][.img\_ext].
- [txdim]x[tydim] specify the x+y pixel dimensions of the tile (e.g., 256x256) 
- [txoff],[tyoff] specify the x+y pixel offsets specifying the location of the tile in image\_file with dimensions [imgrows,imgcols]. NOTE: tiles can fall outside of image boundaries (with pixels outside the image extent file with NODATA pixels). Specifically the ranges of txoff and tyoff are as follows:

  - txoff in [-txdim+1,imgcols+txdim-1]
  - tyoff in [-tydim+1,imgrows+tydim-1] 

## ImageSampler Classes

**CoordSampler**: base class, handles sampling of two-dimensional image coordinates with optional ROI/tile-based spatial stratification, ensures coordinates lie within data mask or optional label mask boundaries. Iteratively samples point batches from each image until convergence criteria satisfied.

**TileSampler**: extends CoordSampler to two-dimensional tiles with optional function-based filtering and overlap rejection. Iteratively samples batches of tiles from each image until convergence criteria satisfied.

**BinaryTileSampler**: combines two TileSampler instances to sample tiles provided a labeled pixel map indicating the locations of positive and/or negative class pixels. Prevents tile overlap for tiles representing distinct classes (e.g., positive vs negative) while optionally allowing overlap among tiles representing the same class.