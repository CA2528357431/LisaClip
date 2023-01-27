import torch
import clip
from PIL import Image

device = "cuda"
pil1 = Image.open(f"../source_pic/emma black.jpg")
pil2 = Image.open(f"../source_pic/cmp emma black.jpg")

model,pre = clip.load('ViT-B/32', device=device)

pics = [pre(pil1),pre(pil2)]
text = ["man with dark black skin and black hair"]

token = clip.tokenize(text).to(device)
image = torch.stack(pics).to(device)


logits_per_image, logits_per_text = model(image, token)
probs = logits_per_text.softmax(dim=-1)

res = probs.tolist()[0]



for i in range(len(pics)):
    print(i,":",res[i])