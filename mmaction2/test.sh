#!/bin/bash
source /root/miniconda3/bin/activate /root/miniconda3/envs/slowfast

bash tools/dist_test.sh /opt/data/private/Chaotic_World/codes/mmaction2/configs/detection/videomae/ours.py /opt/data/private/Chaotic_World/codes/mmaction2/work_dirs/ours/epoch_1.pth 2 --dump result.pkl