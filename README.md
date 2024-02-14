# Demo - Point clouds from tomato depth images
*By Wageningen Research - Greenhouse Horticulture*

This repository is released as part of the 4th International Autonomous Greenhouse Challenge in the experimental facilities of
the Greenhouse Horticulture Business Unit in Bleiswijk, The Netherlands in 2024. Information regarding the competition can be found here http://www.autonomousgreenhouses.com/.

This repository contains a demo python script and notebook to process stereo infra-red images of a tomato plant into a 3-D point cloud.

# Getting started
Setup your python (conda) environment:
```shell
conda create --name pcd python=3.10
conda activate pcd
pip install -r requirements.txt
```

A) Run the demo script:
```shell
python scripts/example.py
```

If all goes well, it will convert the images in `data/depth` and `data/rgb` into a point cloud file (`.pcd`) using the python package `open3d` (https://www.open3d.org/), which you can find in the folder `data/pointclouds`.

B) Check or run the demo notebook in `scripts/Example_notebook.ipynb`.


## Image acquisition
Images were taken with one Oak-D S2 POE camera facing downwards, and the example data includes the collected RGB and Depth data. The depth images were aligned to the color images. Consequently, the described camera intrinsics of both the depth and color
images and resolution (3840 x 2160) are similar. The depth images can be converted to 3D point clouds using the
intrinsics in the file `oak-d-s2-poe_conf.json` (`color_int`).


### Contributing & License
This repository contains a demo script, intended to kickstart the analysis of depth imaging. Please report any issues, but we cannot promise a quick fix.


