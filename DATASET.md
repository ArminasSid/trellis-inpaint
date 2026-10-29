# Trellis-Inpaint

This is a modified version of the original Trellis dataset configuration. Modification revolves around preparing a dataset with full object and snapshots and partial object and snapshots.

This dataset is limited to only HSSD source.

[HSSD](https://huggingface.co/datasets/hssd/hssd-models) filtered based on aesthetic scores. For more samples, aesthetic score can be lowered in code.

This dataset serves for 3D structural inpainting generation tasks.

The dataset is provided as csv files containing the 3D assets' metadata.

## Dataset Statistics

The following table summarizes the dataset's filtering and composition:

***NOTE: Some of the 3D assets lack text captions. Please filter out such assets if captions are required.***
| Source | Aesthetic Score Threshold | Filtered Size | With Captions |
|:-:|:-:|:-:|:-:|
| HSSD | 4.5 | 6670 | 6661 |

## Dataset Location

The dataset is hosted on Hugging Face Datasets. You can preview the dataset at

[https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K](https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K)

There is no need to download the csv files manually. We provide toolkits to load and prepare the dataset.

## Dataset Toolkits

We provide [toolkits](dataset_toolkits) for data preparation.

### Step 1: Install Dependencies

```
. ./dataset_toolkits/setup.sh
```

### Step 2: Load Metadata

First, we need to load the metadata of the dataset.

```
python dataset_toolkits/build_metadata.py HSSD --output_dir <OUTPUT_DIR>
```

- `OUTPUT_DIR`: The directory to save the data.

For example, to load the metadata of the HSSD subset and save it to `datasets/HSSD`, we can run:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 3: Download Data

Next, we need to download the 3D assets.

```
python dataset_toolkits/download.py HSSD --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.

You can also specify the `RANK` and `WORLD_SIZE` of the current process if you are using multiple nodes for data preparation.

For example, to download the HSSD subset and save it to `datasets/HSSD`, we can run: 

***NOTE: The example command below sets a large `WORLD_SIZE` for demonstration purposes. Only a small portion of the dataset will be downloaded.***

```
python dataset_toolkits/download.py HSSD --output_dir datasets/HSSD --world_size 160000
```

Some datasets may require interactive login to Hugging Face or manual downloading. Please follow the instructions given by the toolkits.

After downloading, update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 4: Use blender to render damaged objects

Damaged objects can be rendered with:

```
<path_to_blender>/blender --background --python dataset_toolkits/warp_all.py --output_dir datasets/HSSD
```

### Step 5: Render Multiview Images

Multiview images can be rendered with:

```
python dataset_toolkits/render.py HSSD --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 150.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render the HSSD subset and save it to `datasets/HSSD`, we can run:

```
python dataset_toolkits/render.py HSSD --output_dir datasets/HSSD
```

Apply edge detection to the renders and update info files

```
python dataset_toolkits/edge_detection.py --output_dir datasets/HSSD
```

```
python dataset_toolkits/transforms_move.py --output_dir datasets/HSSD
```

Lastly, for rendering, merge the sketches and renders:

```
python dataset_toolkits/merge_renders_sketches.py --output_dir datasets/HSSD
```


Don't forget to update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 6: Voxelize 3D Models

We can voxelize the 3D models with:

```
python dataset_toolkits/voxelize.py HSSD --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to voxelize the HSSD subset and save it to `datasets/HSSD`, we can run:
```
python dataset_toolkits/voxelize.py HSSD --output_dir datasets/HSSD
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 7: Extract DINO Features

To prepare the training data for SLat VAE, we need to extract DINO features from multiview images and aggregate them into sparse voxel grids.

```
python dataset_toolkits/extract_features.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.


For example, to extract DINO features from the HSSD subset and save it to `datasets/HSSD`, we can run:

```
python dataset_toolkits/extract_feature.py --output_dir datasets/HSSD
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 8: Encode Sparse Structures

Encoding the sparse structures into latents to train the first stage generator:

```
python dataset_toolkits/encode_ss_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to encode the sparse structures into latents for the HSSD subset and save it to `datasets/HSSD`, we can run:

```
python dataset_toolkits/encode_ss_latent.py --output_dir datasets/HSSD
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 9: Encode SLat

Encoding SLat for second stage generator training:

```
python dataset_toolkits/encode_latent.py --output_dir <OUTPUT_DIR> [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to encode SLat for the HSSD subset and save it to `datasets/HSSD`, we can run:

```
python dataset_toolkits/encode_latent.py --output_dir datasets/HSSD
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```

### Step 10: Render Image Conditions

To train the image conditioned generator, we need to render image conditions with augmented views.

```
python dataset_toolkits/render_cond.py HSSD --output_dir <OUTPUT_DIR> [--num_views <NUM_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

- `OUTPUT_DIR`: The directory to save the data.
- `NUM_VIEWS`: The number of views to render. Default is 24.
- `RANK` and `WORLD_SIZE`: Multi-node configuration.

For example, to render image conditions for the HSSD subset and save it to `datasets/HSSD`, we can run:

```
python dataset_toolkits/render_cond.py HSSD --output_dir datasets/HSSD
```

Then update the metadata file with:

```
python dataset_toolkits/build_metadata.py HSSD --output_dir datasets/HSSD
```


