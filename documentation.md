# Overview 
The AIEA lab's efforts into the Argoverse Lidar Flow Scene 2
challenge, specificially the supervised track.

The basic end goal is that given 2 lidar sweeps and an annotation
provided by argoverse, we predict where the points are going to move. 

The implementation is a pretty generic sparse flow net, where spconv
is fed into a UNET style autoencoder to produce the prediction. 

# Architecture
First there is a union step. This takes sweep 0 and sweep 1 and merges
them into a single one. By inputting voxel coordinates (and the rest
of the features), a hash key is used for idnetification of each
voxel, and then we produce an inex map of sweep 0 back to their union

These layers/blocks are used in the architecture:
- Submanifold convolutions (submConv) (3x3) wiith Batchnorm +relu
- Downblock, strided sparse conv 3d (2x2) for downsampling
- upBlock, sparseinvconv3d for upsampling
- catsparse, concatatiton of 2 sparse tensors i/ same coords

In the encoder we have:
1. submblock (10 -> 32)
1. sparseconv3d (32 -> 64)
1. submblock (64 -> 64)
1. sparseconv3d (64 ->128)
1. submblock (128 -> 128)
1. sparseconv3d (128 -> 256)
1. submblock (256 -> 256)

In the decoder we have:
1. sparseinvconv3d (256 -> 128)
1. submblock (256 -> 128)
1. sparseinvconv3d (128 -> 64)
1. submblock (128 -> 64)
1. sparseinvconv3d (64 -> 32)
1. submblock (64 -> 32)
1. submblock (kernel = 1 for this one) (32 -> 3})
Output channels are a flow vector for each voxel.
Also note that before every submblock we also catsparse in the decoder.

# Assorted other notes
Due to limitations in nautilus, we serialized all the data into .pt
files and stored them separately. This cache allows for fasr access
and compressing the dataset into ~25gb. This also would theoretically
allow for the entire dataset to fit into the memory of a GPU with high
vram (L40, A100, etc). Might get more stable gradients this
ways. Perhaps by user error, (potentially a problem with the screen
library in ubuntu, commands used and images can be found in job.yml
and start.sh), there seems to be some form of a memory leak (or pods
die with code 137) when the workers load large portions of the dataset
directly. Idk how to fix this, so the loader code has tons of like
memory freeing practice sin python so it's pretty messy. 

Training was primarily done on a Tesla V100 16GB gpu.
