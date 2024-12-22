#!/bin/bash
# Source the conda setup script
source /home/jstenzel/anaconda3/etc/profile.d/conda.sh
cd /home/jstenzel/cor013-truck_utilization-3d-2d/scripts/models/instance_segmentation
conda activate gate_segmentation
python run_on_image_dir.py
cd /home/jstenzel/mde-reconstruction
conda activate cargosight
python pointcloud_reconstruction/reconstruction-pipeline-jonas.py 
