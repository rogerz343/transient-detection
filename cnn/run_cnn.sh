#!/bin/bash
  
echo Running removeDup

module load python/2.7-anaconda-4.4
source activate convNet
python ~/520project/transient-detection/cnn/run_cnn_v2.py medium_sample
