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

## Inference

1. Pull the docker image from the repository:
```sh
docker pull ghcr.io/bjliefers/cfi-amd:latest
```

2. Prepare your data:

The docker image expects as input a .csv file with two columns:
- `identifier` (name or ID of the image)
- `path` (the full path to the image)

For example:
```
identifier,path
image0,/data/image0.dcm
image1,/data/image1.png
image2,/data/image2.jpg
```
3. Run 

Option 1 (docker-compose):

Create a file `docker-compose.yml` like this:
```
services:
  inference:
    image: ghcr.io/bjliefers/cfi-amd:latest
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /full/path/to/input.csv:/input.csv
      - /full/path/to/output:/output
      - /full/path/to/data:/data
```
Make sure that:
- Your .csv file is available to the container under `/input.csv`
- The target directory is mounted at `/output` (this is where the output will be generated)
- ensure the paths in the .csv file are accessible inside the container
- [optionally] configure available gpu devices

Run:
```
docker-compose up
```
The image should process each file in `/input.csv`, and write to the folder mounted under `/output`, making a subfolder for each row in the table.

## Building the image
```
git clone git@github.com:Eyened/cfi-amd.git
```
Download model weights:
```
?
```
Store them in the folder `models`

```
docker-compose build
```