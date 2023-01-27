import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import random

import math
import clip
from PIL import Image

from utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small


def init_fc(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01, mean=0)


class FC(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
            # nn.Linear(channel, channel),
        )
        self.block.apply(init_fc)
        for x in self.block.parameters():
            x.requires_grad = False
        self.requires_grad = False

    def forward(self, x):
        # x = x.float()
        # x = self.block(x.float())
        # min = x.min()
        # max = x.max()
        # x = (x-min)/(max-min)
        return x


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)


class Preprocess(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=True)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        mean = mean.reshape(1, 3, 1, 1)
        mean = mean.expand(x.shape).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
        std = std.reshape(1, 3, 1, 1)
        std = std.expand(x.shape).to(self.device)

        std.requires_grad = False
        mean.requires_grad = False

        x = (x - mean) / std

        return x


class CLIPLoss(torch.nn.Module):
    def __init__(self, device, direction_loss_type='cosine', clip_model='ViT-B/32'):
        super().__init__()

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)
        for x in self.model.parameters():
            x.requires_grad = False

        # self.clip_preprocess = clip_preprocess

        # self.preprocess = transforms.Compose(
        #     [
        #         # transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
        #         transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        #     ]
        # )

        self.preprocess = Preprocess(self.device)

        # self.preprocess = transforms.Compose(
        #     [
        #         # transforms.Normalize(mean=(-1.0, -1.0, -1.0), std=(2.0, 2.0, 2.0)),
        #         # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        #         transforms.ToPILImage()
        #     ] +
        #     clip_preprocess.transforms
        # )

        # self.texture_loss = torch.nn.MSELoss()
        # self.angle_loss = torch.nn.L1Loss()
        # self.patch_loss = DirectionLoss("mae")

        # self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)

        self.direction_loss = DirectionLoss(direction_loss_type)

        self.fc = FC(512).to(self.device)
        # if clip_model == "ViT-B/32":
        #     self.fc = torch.load('fc1.pth')
        # elif clip_model == "RN101":
        #     self.fc = torch.load('fc2.pth')
        # else:
        #     self.fc.block.apply(init_fc)

        self.target_direction = None
        self.src_text_features = None
        self.patch_text_directions = None

        self.patch_points = torch.zeros(0, 2).int()
        self.patch_size = 384
        self.treshold = 0.7
        self.top_lambda = 1
        self.right_patch = []


    def tokenize(self, strings):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens):
        code = self.model.encode_text(tokens)
        code = self.fc(code)
        return code

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # images = torch.nn.functional.interpolate(images, (224,224), mode="bilinear")
        # images = self.preprocess(images.squeeze(0).cpu())
        # images = images.to(self.device).unsqueeze(0)
        images = self.preprocess(images)
        code = self.model.encode_image(images)
        code = self.fc(code)

        return code

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                              target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class).detach()

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum().item() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True))

        return self.direction_loss(edit_direction, self.target_direction).mean()

    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        image = self.preprocess(img)
        logits_per_image, _ = self.model(image, tokens)

        return (1. - logits_per_image / 100).mean()


        # tokens = clip.tokenize(text).to(self.device)
        # text_v = self.encode_text(tokens)
        # image_v = self.encode_images(img)
        # logits_per_image = torch.nn.functional.cosine_similarity(image_v, text_v).unsqueeze(0)

        # return (1. - logits_per_image).mean()

    def random_patch_points(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape
        half = size // 2

        w = torch.randint(low=half, high=width - half, size=(num_patches, 1))
        h = torch.randint(low=half, high=height - half, size=(num_patches, 1))
        points = torch.cat([h, w], dim=1)

        return points

    def generate_patches(self, img: torch.Tensor, patch_points, size):

        num_patches = patch_points.shape[0]

        patches = []
        half = size // 2

        for patch_idx in range(num_patches):
            point_x = patch_points[0 * num_patches + patch_idx][0]
            point_y = patch_points[0 * num_patches + patch_idx][1]
            patch = img[0:1, :, point_y - half:point_y + half, point_x - half:point_x + half]
            patch = torch.nn.functional.interpolate(patch, (512, 512), mode="bilinear", align_corners=True)
            patches.append(patch)

        patches = torch.cat(patches, dim=0)

        return patches

    def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                               target_class: str):

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class).detach()

        patch_size = 128
        # patch_size = self.patch_size
        patch_num = 64

        patch_points = self.random_patch_points(src_img.shape, patch_num, patch_size)

        src_patches = self.generate_patches(src_img, patch_points, patch_size)
        src_features = self.get_image_features(src_patches)

        target_patches = self.generate_patches(target_img, patch_points, patch_size)
        target_features = self.get_image_features(target_patches)

        # src_feature = self.get_image_features(src_img)
        # src_features = src_feature.expand(target_features.shape)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        dirs = self.direction_loss(edit_direction, self.target_direction)
        dirs[dirs < 0.7] = 0

        if self.patch_size>64:
            self.patch_size-=2

        return dirs.mean()

    def get_image_prior_losses(self, img):
        diff1 = img[:, :, :, :-1] - img[:, :, :, 1:]
        diff2 = img[:, :, :-1, :] - img[:, :, 1:, :]
        diff3 = img[:, :, 1:, :-1] - img[:, :, :-1, 1:]
        diff4 = img[:, :, :-1, :-1] - img[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

        return loss_var_l2





    def forward_gol(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):

        # gol_loss = 1 * self.clip_directional_loss(src_img, source_class, target_img, target_class)
        gol_loss = 1 * self.global_clip_loss(target_img, target_class)
        # loss += 1 * self.clip_angle_loss(src_img, source_class, target_img, target_class)
        # print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
        # loss += loss1 + loss2 + loss3 + loss4
        # loss.requires_grad = True
        return gol_loss

    def forward_dir(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):

        dir_loss = 1 * self.clip_directional_loss(src_img, source_class, target_img, target_class)
        # dir_loss += 1 * self.patch_directional_loss(src_img, source_class, target_img, target_class)
        # dir_loss = 1 * self.global_clip_loss(target_img, f"a {target_class}")
        # dir_loss += 1 * self.clip_angle_loss(src_img, source_class, target_img, target_class)
        # print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
        # loss += loss1 + loss2 + loss3 + loss4
        # loss.requires_grad = True
        return dir_loss

    def forward_patch(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):

        # dir_loss = 1 * self.clip_directional_loss(src_img, source_class, target_img, target_class)
        patch_loss = 1 * self.patch_directional_loss(src_img, source_class, target_img, target_class)
        # dir_loss = 1 * self.global_clip_loss(target_img, f"a {target_class}")
        # dir_loss += 1 * self.clip_angle_loss(src_img, source_class, target_img, target_class)
        # print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
        # loss += loss1 + loss2 + loss3 + loss4
        # loss.requires_grad = True

        return patch_loss

    def forward_prior(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):
        prior_loss = 1 * self.get_image_prior_losses(target_img)

        return prior_loss

    def patch_directional_loss_sec(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor,
                                   target_class: str):

        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class).detach()

        patch_size = self.patch_size
        patch_num = 64

        new_points_num = patch_num - self.patch_points.shape[0]
        patch_points = torch.cat(
            [self.patch_points, self.random_patch_points(src_img.shape, new_points_num, patch_size)], dim=0)

        src_patches = self.generate_patches(src_img, patch_points, patch_size)
        src_features = self.get_image_features(src_patches)

        target_patches = self.generate_patches(target_img, patch_points, patch_size)
        target_features = self.get_image_features(target_patches)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        dirs = self.direction_loss(edit_direction, self.target_direction)


        with torch.no_grad():
            li = [x.item() for x in dirs]
            avg = dirs.mean()
            dirs[dirs < min(self.treshold,avg.item())] = 0

        # cnt = 0

        toploss = torch.max(dirs)

        next_points = []

        fast = torch.zeros(1, ).to(self.device) + toploss * self.top_lambda
        slow = torch.zeros(1, ).to(self.device) + toploss * self.top_lambda

        right_patch = 0
        for i in range(patch_num):
            if dirs[i] <= avg:
                fast += dirs[i]
            else:
                slow += dirs[i]

        for i in range(patch_num):
            if dirs[i] > avg:
            # if dirs[i] > avg*1000:
                half = patch_size // 2 - 1
                y, x = patch_points[i]
                if y - half >= 0 and y + half <= 512 and x - half >= 0 and x + half <= 512:
                    next_points.append(patch_points[i])
                    right_patch += 1
        self.right_patch.append(right_patch)



        # fast = torch.zeros(1, ).to(self.device) + toploss*self.top_lambda
        # slow = torch.zeros(1, ).to(self.device) + toploss*self.top_lambda
        #
        # soft = torch.softmax(dirs,dim=0)
        # li = [x for x in range(patch_num)]
        # li.sort(key=lambda x:soft[x].item())
        # right_patch = 0
        # for ind in range(patch_num):
        #     i = li[ind]
        #     if ind < patch_num//2:
        #         fast += dirs[i]
        #     else:
        #         slow += dirs[i]
        #         half = patch_size // 2 - 1
        #         y, x = patch_points[i]
        #         if y - half >= 0 and y + half <= 512 and x - half >= 0 and x + half <= 512:
        #             next_points.append(patch_points[i])
        #             right_patch += 1
        # self.right_patch.append(right_patch)



        if next_points:
            self.patch_points = torch.stack(next_points, dim=0)
        else:
            self.patch_points = torch.zeros(0, 2).int()

        if self.patch_size >= 32:
            self.patch_size -= 2
        #     self.treshold += 0.0002

        return fast / patch_num, slow / patch_num, li

    def forward_patch_sec(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str):

        # dir_loss = 1 * self.clip_directional_loss(src_img, source_class, target_img, target_class)
        # patch_loss = 1 * self.patch_directional_loss(src_img, source_class, target_img, target_class)
        # dir_loss = 1 * self.global_clip_loss(target_img, f"a {target_class}")
        # dir_loss += 1 * self.clip_angle_loss(src_img, source_class, target_img, target_class)
        # print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
        # loss += loss1 + loss2 + loss3 + loss4
        # loss.requires_grad = True
        fast, slow, li = 1 * self.patch_directional_loss_sec(src_img, source_class, target_img,
                                                         target_class)
        return fast, slow, li


