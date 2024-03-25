#!/bin/bash
source /root/miniconda3/bin/activate /root/miniconda3/envs/slowfast

# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/ours.py 4 --cfg-options randomness.seed=0 randomness.deterministic=True

# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/slowfast/ours_slowfast_r50.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True

# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/ours_base.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True
# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/base_acar.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True --resume /opt/data/private/Chaotic_World/codes/mmaction2/work_dirs/base_acar/epoch_2.pth
# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/ours_base_no_auto_scale_lr.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True
# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/videomae_clip_base.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True
# bash tools/dist_train.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/videomae_clip_visual.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True
bash tools/dist_train.sh /opt/data/private/Chaotic_World/mmaction2/configs/detection/videomae/videomae_clip_visual_decoder.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True