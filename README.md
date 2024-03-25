## Requirements
Requirements are given in environment.yaml
## Usage
- Download ViT-B-16.pt in Model Zoo for CLIP, edit clip_model in config "/opt/data/private/Chaotic_World/mmaction2/configs/detection/videomae/videomae_clip_visual_decoder.py". 
- Change the corresponding file path in config "/opt/data/private/Chaotic_World/mmaction2/configs/detection/videomae/videomae_clip_visual_decoder.py". 
- Train
bash tools/dist_train.sh /opt/data/private/Chaotic_World/mmaction2/configs/detection/videomae/videomae_clip_visual_decoder.py 2 --cfg-options randomness.seed=0 randomness.deterministic=True
## Model Zoo
Trained models and clip pretrained model are provided in https://pan.baidu.com/s/1ateBYE4mjGw71ryd-KrQ9Q key:1234
