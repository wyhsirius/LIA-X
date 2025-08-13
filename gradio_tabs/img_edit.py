import gradio as gr
import os
import torch
import torchvision
from PIL import Image
import numpy as np
import imageio
from einops import rearrange

output_dir = "./res_gradio"
os.makedirs(output_dir, exist_ok=True)

# lables
labels_k = [
	'yaw1',
	'yaw2',
	'pitch',
	'roll1',
	'roll2',
	'neck',

	'pout',
	'open->close',
	'"O" Mouth',
	'smile',

	'close->open',
	'eyebrows',
	'eyeballs1',
	'eyeballs2',

]

labels_v = [
	37, 39, 28, 15, 33, 31,
	6, 25, 16, 19,
	13, 24, 17, 26
]


def load_image(img, size):
	img = Image.open(img).convert('RGB')
	w, h = img.size
	img = img.resize((size, size))
	img = np.asarray(img)
	img = np.transpose(img, (2, 0, 1))	# 3 x 256 x 256

	return img / 255.0, w, h


def img_preprocessing(img_path, size):
	img, w, h = load_image(img_path, size)  # [0, 1]
	img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
	imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

	return imgs_norm, w, h


def resize(img, size):
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size, antialias=True),
		torchvision.transforms.CenterCrop(size)
	])

	return transform(img)

def resize_back(img, w, h):
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize((h, w), antialias=True),
	])

	return transform(img)

def vid_preprocessing(vid_path, size):
	vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
	vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)	# btchw
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


def img_postprocessing(image, w, h, output_path=output_dir + "/output_img.png"):

	image = resize_back(image, w, h)
	image = image.permute(0, 2, 3, 1)
	edited_image = img_denorm(image)
	img_output = (edited_image[0].numpy() * 255).astype(np.uint8)
	imageio.imwrite(output_path, img_output, quality=6)

	return output_path


def vid_postprocessing(video, w, h, fps, output_path=output_dir + "/output_vid.mp4"):
	# video: BCTHW

	b,c,t,_,_ = video.size()
	vid_batch = resize_back(rearrange(video, "b c t h w -> (b t) c h w"), w, h)
	vid = rearrange(vid_batch, "(b t) c h w -> b t h w c", b=b)	# B T H W C
	vid_np = (vid_denorm(vid[0]).numpy() * 255).astype('uint8')
	imageio.mimwrite(output_path, vid_np, fps=fps, codec='libx264', quality=10)

	return output_path


def clear_media():
	return None, *([0] * len(labels_k))


def img_edit(gen, device):

	@torch.no_grad()
	def edit_img(image, *selected_s):

		image_tensor, w, h = img_preprocessing(image, 512)
		image_tensor = image_tensor.to(device)

		edited_image_tensor = gen.edit_img(image_tensor, labels_v, selected_s)

		# de-norm
		edited_image = img_postprocessing(edited_image_tensor, w, h)

		return edited_image


	with gr.Tab("Image Editing"):

		inputs_s = []

		with gr.Row():
			with gr.Column(scale=1):
				with gr.Row():
					with gr.Accordion(open=True, label="Image"):
						image_input = gr.Image(type="filepath", width=512)	# , height=550)
						gr.Examples(
							examples=[
								["./data/source/macron.png"],
								["./data/source/einstein.png"],
								["./data/source/taylor.png"],
								["./data/source/portrait1.png"],
								["./data/source/portrait2.png"],
								["./data/source/portrait3.png"],
							],
							inputs=[image_input],
							cache_examples=False,
							visible=True,
							)


				with gr.Row():
					with gr.Column(scale=1):
						with gr.Row():	# Buttons now within a single Row
							edit_btn = gr.Button("Edit")
							clear_btn = gr.Button("Clear")
						with gr.Row():
							animate_btn = gr.Button("Generate")



			with gr.Column(scale=1):

				with gr.Row():
					with gr.Accordion(open=True, label="Edited Image"):
						image_output = gr.Image(label="Output Image", type='numpy', interactive=False, width=512)


				with gr.Accordion("Control Panel", open=True):
					with gr.Tab("Head"):
						with gr.Row():
							for k in labels_k[:3]:
								slider = gr.Slider(minimum=-1.0, maximum=0.5, value=0, label=k)
								inputs_s.append(slider)
						with gr.Row():
							for k in labels_k[3:6]:
								slider = gr.Slider(minimum=-0.5, maximum=0.5, value=0, label=k)
								inputs_s.append(slider)

					with gr.Tab("Mouth"):
						with gr.Row():
							for k in labels_k[6:8]:
								slider = gr.Slider(minimum=-0.4, maximum=0.4, value=0, label=k)
								inputs_s.append(slider)
						with gr.Row():
							for k in labels_k[8:10]:
								slider = gr.Slider(minimum=-0.4, maximum=0.4, value=0, label=k)
								inputs_s.append(slider)

					with gr.Tab("Eyes"):
						with gr.Row():
							for k in labels_k[10:12]:
								slider = gr.Slider(minimum=-0.4, maximum=0.4, value=0, label=k)
								inputs_s.append(slider)
						with gr.Row():
							for k in labels_k[12:14]:
								slider = gr.Slider(minimum=-0.2, maximum=0.2, value=0, label=k)
								inputs_s.append(slider)


		edit_btn.click(
			fn=edit_img,
			inputs=[image_input] + inputs_s,
			outputs=[image_output],
			show_progress=True
		)

		clear_btn.click(
			fn=clear_media,
			outputs=[image_output] + inputs_s
		)
