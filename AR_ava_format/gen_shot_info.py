import os
import csv

dataset_dir = '/opt/data/private/Chaotic_World/dataset/frames_320x180'
res_path = '/opt/data/private/Chaotic_World/annotations/AR_ava_format/frame_num.csv'
videos = os.listdir(dataset_dir)
res = []
for video in videos:
    imgs_num = len(os.listdir(os.path.join(dataset_dir, video)))
    res.append([video, imgs_num])

with open(res_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(res)