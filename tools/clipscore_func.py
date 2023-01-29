#%%

import os

import torch
from torchvision import transforms, models


import clip

import random

from PIL import Image
from matplotlib import pyplot as plt

#%%

topil = transforms.ToPILImage()
topic = transforms.ToTensor()


#%%

def random_patch(img, size):



        channels, height, width = img.shape
        half = size // 2

        point_x = random.randint(half, width - half)
        point_y = random.randint(half, height - half)

        patch = img[:, point_y - half:point_y + half, point_x - half:point_x + half].cpu()

        pil = topil(patch)

        return pil

#%%

model,pre = clip.load('ViT-B/32', device="cuda")

#%%

score_func = torch.nn.CosineSimilarity()



#%%

def get_patch_score(path, text):
    pic1 = topic(Image.open(path))
    size1 = random.randint(32,112)*2
    p1 = pre(random_patch(pic1,size1)).unsqueeze(0).to("cuda")
    v1 = model.encode_image(p1)

    text = [text]
    token = clip.tokenize(text).to("cuda")
    v2 = model.encode_text(token)

    res = score_func(v1,v2)

    return res.item()

def get_score(path, text):
    res = 0
    for _ in range(64):
        res+=get_patch_score(path, text)
    return res/64


#%%

def getnames(path):
    file_dir = path
    li = []
    for root, dirs, files in os.walk(file_dir, topdown=False):
        li.append(files)
    li = li[0]
    li = [x for x in li if x[-3:]!="txt"]
    li.sort()
    return li

#%%

def getfeature(string):
    name = string.split(".")[0]
    while name[-1] in "0123456789":
        name = name[:-1]
    feature = name.split("-")[-1]
    return feature
