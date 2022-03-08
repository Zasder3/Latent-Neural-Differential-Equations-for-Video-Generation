[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/latent-neural-differential-equations-for/video-generation-on-ucf-101-16-frames-64x64)](https://paperswithcode.com/sota/video-generation-on-ucf-101-16-frames-64x64?p=latent-neural-differential-equations-for)

# Latent Neural Differential Equations for Video Generation
Exploring the usage of Neural O/SDEs for the evolution of latent variables in videos.

This code was written while I was in high school, so sorry for the many cleanliness issues that brings. For fear of breaking the code, I have chosen not to refactor the scripts, and instead offer explanations to some of the confusing parts. 

## Data

To prepare your data first download UCF101, then use the file `ucf101/make_ucf101_tgan.py`

To use it properly alter:

```python
src_dir = 'D:/Video Datasets/UCF101_min'
src_split_dir = 'D:/Video Datasets/ucf101/ucfTrainTestList'
dst_dir = 'D:/Video Datasets/ucf101_64px_tgan'
```

Set `src_dir` and `src_split_dir` to their corresponding values, then specify your destination directory (`dst_dir`).

## Usage

In the header of each file there is a block of variables that looks like:
```python
epochs = 100000
batch_size = 32
path = 'ucf101/tgan_svc_ode_lin'
start_epoch = 0
conf = "C:/Video Datasets/ucf101_64px/train.json"
dset = "C:/Video Datasets/ucf101_64px/train.h5"
```

The variables `epochs` and `batch_size` are self-explanatory. `path` specifies a directory relative to the starting file in two folders called `checkpoints` and `video_samples`. (`checkpoints/{path}/` and `video_samples/{path}/`) One existing issue is that these directories must be created by the user BEFORE running. `conf` and `dset` are hard-coded paths to your data directory.

From there you should be good to go!

If you need to resume a run for whatever reason alter the `start_epoch` variable and uncomment the lines which handle loading checkpoints.
