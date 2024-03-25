import clip
import json
from coop import PromptLearner, TextEncoder
def read_labelmap(labelmap_file):
    labelmap = []
    class_ids = set()
    name = ''
    class_id = ''
    for line in labelmap_file:
        if line.startswith('  name:'):
            name = line.split('"')[1]
        elif line.startswith('  id:') or line.startswith('  label_id:'):
            class_id = int(line.strip().split(' ')[-1])
            labelmap.append({'id': class_id, 'name': name})
            class_ids.add(class_id)
    return labelmap, class_ids

clip_model, _ = clip.load('/opt/data/private/Chaotic_World/codes/clip/ViT-B-16.pt', device='cpu', jit=False)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
label_map, _ = read_labelmap(open('/opt/data/private/Chaotic_World/annotations/AR_ava_format/list_action.pbtxt'))
# [{'id': class_id, 'name': name}]
classnames = [label['name'] for label in label_map]
extend_pro = []
coop_ctx_init = ""
with open('/opt/data/private/Chaotic_World/annotations/AR_ava_format/label_extend.json', 'r') as f:
    extend_text = json.load(f) 
for cls in classnames:
    extend_pro.extend(extend_text[cls])
prompt_learner = PromptLearner(n_ctx=4, ctx_init="", classnames=extend_pro, clip_model=clip_model)
prompts = prompt_learner()
tokenized_prompts = prompt_learner.tokenized_prompts
text_encoder = TextEncoder(clip_model)
text_feats = text_encoder(prompts, tokenized_prompts) 
print(text_feats.size())