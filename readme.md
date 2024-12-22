# Benchmarking Poincloud Reconstruction Using Monocular Depth Estimation 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This repo contains code to load, process and benchmark data collected through the Navvis 3D scanner and IPad Lidar with code to benchmark different metric MDE models for the downstream task of pointcloud reconstruction. The models currently benchmakred are ZoeDepth, ZeroDepth, DepthAnything, PixelFormer, UniDepth & Metric3D.

## Example
Example results of pointcloud estimation from ZoeDepth.

![processed](./mde-result/zoe-depth/rockspred.png)
![output1](./mde-result/zoe-depth/zoe_pcds/rocks_mde_pcd_1.PNG)
![output2](./mde-result/zoe-depth/zoe_pcds/rocks_mde_pcd_2.PNG)
![benchmarking](./combined_results.png)
