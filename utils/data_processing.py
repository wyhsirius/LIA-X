import os
import torch
import torchvision
from PIL import Image
import numpy as np
import imageio
from einops import rearrange, repeat


def load_image(img, size):
    # img = Image.open(filename).convert('RGB')
    if not isinstance(img, np.ndarray):
        img = Image.open(img).convert('RGB')
        img = img.resize((size, size))
        img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def resize(img, size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size, antialias=True),
        torchvision.transforms.CenterCrop(size)
    ])

    return transform(img)


def vid_preprocessing(vid_path, size):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)  # btchw
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    vid_norm = torch.cat([
        resize(vid_norm[:, i, :, :, :], size).unsqueeze(1) for i in range(vid.size(1))
    ], dim=1)

    return vid_norm, fps


def img_denorm(img):
    img = img.clamp(-1, 1).cpu()
    img = (img - img.min()) / (img.max() - img.min())

    return img


def vid_denorm(vid):
    vid = vid.clamp(-1, 1).cpu()
    vid = (vid - vid.min()) / (vid.max() - vid.min())

    return vid


def save_img_edit(save_dir, img, img_e):
    # img: BCHW
    # img_e: BCHW

    output_img_path = os.path.join(save_dir, "img_edit.png")
    output_img_all_path = os.path.join(save_dir, "img_all.png")

    img = rearrange(img, 'b c h w -> b h w c')
    img_e = rearrange(img_e, 'b c h w -> b h w c')
    img_all = torch.cat([img, img_e], dim=2)

    img_e_np = (img_denorm(img_e[0]).numpy() * 255).astype('uint8')
    img_all_np = (img_denorm(img_all[0]).numpy() * 255).astype('uint8')

    imageio.imwrite(output_img_path, img_e_np, quality=8)
    imageio.imwrite(output_img_all_path, img_all_np, quality=8)

    return


def save_vid_edit(save_dir, vid_d, vid_a, fps):
    # img_s: BCHW
    # vid_d: BTCHW
    # vid_a: BCTHW

    output_vid_a_path = os.path.join(save_dir, "vid_animation.mp4")
    output_vid_all_path = os.path.join(save_dir, "vid_all.mp4")

    vid_d = rearrange(vid_d, 'b t c h w -> b t h w c')
    vid_a = rearrange(vid_a, 'b c t h w -> b t h w c')
    vid_all = torch.cat([vid_d, vid_a], dim=3)

    vid_a_np = (vid_denorm(vid_a[0]).numpy() * 255).astype('uint8')
    vid_all_np = (vid_denorm(vid_all[0]).numpy() * 255).astype('uint8')

    imageio.mimwrite(output_vid_a_path, vid_a_np, fps=fps, codec='libx264', quality=8)
    imageio.mimwrite(output_vid_all_path, vid_all_np, fps=fps, codec='libx264', quality=8)

    return


def save_animation(save_dir, img_s, vid_d, vid_a, fps):
    # img_s: BCHW
    # vid_d: BTCHW
    # vid_a: BCTHW

    output_vid_a_path = os.path.join(save_dir, "vid_animation.mp4")
    output_img_e_path = os.path.join(save_dir, "img_edit.png")
    output_vid_all_path = os.path.join(save_dir, "vid_all.mp4")

    vid_d = rearrange(vid_d, 'b t c h w -> b t h w c')
    vid_a = rearrange(vid_a, 'b c t h w -> b t h w c')
    img_s = repeat(rearrange(img_s, 'b c h w -> b h w c'), 'b h w c -> b t h w c', t=vid_d.size(1))
    vid_all = torch.cat([img_s, vid_d, vid_a], dim=3)

    vid_a_np = (vid_denorm(vid_a[0]).numpy() * 255).astype('uint8')
    img_e_np = vid_a_np[0]
    vid_all_np = (vid_denorm(vid_all[0]).numpy() * 255).astype('uint8')

    imageio.mimwrite(output_vid_a_path, vid_a_np, fps=fps, codec='libx264', quality=8)
    imageio.mimwrite(output_vid_all_path, vid_all_np, fps=fps, codec='libx264', quality=8)
    imageio.imwrite(output_img_e_path, img_e_np, quality=8)

    return


def save_linear_manipulation(save_dir, vid, fps):
    # vid: BCTHW

    output_vid_path = os.path.join(save_dir, "vid_interpolation.mp4")

    vid = rearrange(vid, 'b c t h w -> b t h w c')
    vid_np = (vid_denorm(vid[0]).numpy() * 255).astype('uint8')

    imageio.mimwrite(output_vid_path, vid_np, fps=fps, codec='libx264', quality=8)

    return