import torch
import clip
from PIL import Image

device = "cuda"
pil1 = Image.open(f"../source_pic/sunflower.jpg")
pil2 = Image.open(f"../result/result0.jpg")

model,pre = clip.load('ViT-B/32', device=device)

text = ["cat", "dog"]

token = clip.tokenize(text).to(device)
image = pre(pil).unsqueeze(0).to(device)


logits_per_image, logits_per_text = model(image, token)
probs = logits_per_image.softmax(dim=-1)

res = probs.tolist()[0]



for i in range(len(text)):
    print(text[i],":",res[i])