#!/bin/bash
source /root/miniconda3/bin/activate /root/miniconda3/envs/slowfast

# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/slowfast/ours_slowfast_r50.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True
bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/slowfast/ours_slowfast_r50_temporal_max_focal.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True