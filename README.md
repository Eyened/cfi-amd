# CFI AMD

Segmentation of intermediate AMD features on color fundus images.

## Summary

This repository contains the code for running the AMD features models.
Input to the model is a color fundus image.
The output consists of a segmentation (binary images) of:
 - Drusen
 - Reticular pseudodrusen
 - Hyperpigmentation
 - RPE degeneration

In addition, it detects the bounds of the image and the location of the fovea and edge of the optic disc to place the ETDRS-grid.
Segmentation outputs are converted to area (in mm²) and counts (number of connected components)

## Installation
To install as a python package:
```
git clone git@github.com:Eyened/cfi-amd.git
cd cfi-amd
pip install .
```
This will install the python package `cfi_amd`

Then you can import the processor like this:
```python
from cfi_amd.processor import Processor

device = 'cuda:0'
processor = Processor(device)
```

The processor can be used like this:
```python
from PIL import Image
import numpy as np
import pydicom

# jpg, png, tif etc:
image = np.array(Image.open('/path/to/image.jpg'))
# or dicom:
image = pydicom.dcmread('/path/to/image.dcm').pixel_array

result = processor.process(image)
```
Note that image is numpy array of shape (h, w, 3) of dtype np.uint8

The result will contain the segmentation for each of the features.

Check [example.ipynb](example.ipynb)

## Inference using Docker

1. Pull the docker image from the repository:
```
docker pull ghcr.io/eyened/cfi-amd:latest
```

2. Prepare your data:

The docker image expects as input a .csv file with two columns:
- `identifier` (name or ID of the image)
- `path` (the path to the image in the container including a /data prefix)

For example:
```
identifier,path
image0,/data/image0.dcm
image1,/data/image1.png
image2,/data/image2.jpg
```
3. Run

The image should process each file in `/input.csv`, and write to the folder mounted under `/output`, making a subfolder for each row in the table.

Make sure that:
- Your .csv file is available to the container under `/input.csv`
- The target directory exists and is mounted at `/output` (this is where the output will be generated)
- ensure the paths in the .csv file are accessible inside the container
- [optionally] configure available gpu devices

Using docker-compose:

Create a file `docker-compose.yml` like this:
```
services:
  inference:
    image: ghcr.io/eyened/cfi-amd:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /path/to/input.csv:/input.csv
      - /path/to/output:/output
      - /path/to/data:/data
```
Run:
```
docker-compose up
```

Alternativelu, using docker run:
```
docker run --gpus '"device=0"' -v /path/to/input.csv:/input.csv -v /path/to/output:/output -v /path/to/images:/data ghcr.io/eyened/cfi-amd:latest
``


## Building the image
To build the Docker image, first clone the git repository:
```
git clone git@github.com:Eyened/cfi-amd.git
cd cfi-amd
```
Download model weights:

- https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/discedge_july24.pt
- https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/fovea_july24.pt
- https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/drusen.zip
- https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/RPD.zip
- https://github.com/Eyened/cfi-amd/releases/download/v0.1-alpha/pigment.zip

Store them in a folder called `models`:
```
mkdir models
unzip drusen.zip
unzip RPD.zip
unzip pigment.zip
```


To obtain this folder structure:
```
cfi-amd
├── cfi_amd
│   ├── ...
├── models
│   ├── discedge_july24.pt
│   ├── fovea_july24.pt
│   ├── drusen
│   │   ├── model_0.ckpt
│   │   ├── model_1.ckpt
│   │   ├── model_2.ckpt
│   │   ├── model_3.ckpt
│   │   └── model_4.ckpt
│   ├── RPD
│   │   ├── model_0.ckpt
│   │   ├── model_1.ckpt
│   │   ├── model_2.ckpt
│   │   ├── model_3.ckpt
│   │   └── model_4.ckpt
│   ├── pigment
│   │   ├── model_0.ckpt
│   │   ├── model_1.ckpt
│   │   ├── model_2.ckpt
│   │   ├── model_3.ckpt
│   │   └── model_4.ckpt
```

Build the image:
```
docker-compose build
```

To run, check the volume mounts in `docker-compose.yml` and run
```
docker-compose up
```
