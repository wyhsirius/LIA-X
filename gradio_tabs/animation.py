# Copyright (C) 2025, Shanghai AI Laboratory, Inria STARS research group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  wyhsirius@gmail.com, francois.bremond@inria.fr, antitza.dancheva@inria.fr
#

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
	'apple cheek',

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
	img, w, h = load_image(img_path, size)	# [0, 1]
	img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
	imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

	return imgs_norm, w, h


def resize(img, size):
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(
			size,
			interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
			antialias=True),
	])

	return transform(img)


def vid_preprocessing(vid_path, size):
	vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
	vid = vid_dict[0].permute(0, 3, 1, 2) # tchw
	fps = vid_dict[2]['video_fps']
	vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]
	vid_norm = resize(vid_norm, (size, size)) # tchw

	return vid_norm, fps


def denorm(x):
	x = x.clamp(-1, 1)
	x = (x - x.min()) / (x.max() - x.min())

	return x


def img_postprocessing(image, w, h, output_path=output_dir + "/output_img.png"):
	# image: BCHW

	image = resize(image, (h, w))
	image = rearrange(image, "b c h w -> b h w c")
	img_np = (denorm(image[0]).cpu().numpy() * 255).astype(np.uint8)
	imageio.imwrite(output_path, img_np, quality=8)

	return output_path


def vid_postprocessing(video, w, h, fps, output_path=output_dir + "/output_vid.mp4"):
	# video: tchw

	video = resize(video, (h, w)) # tchw
	video = rearrange(video, "t c h w -> t h w c") # thwc
	vid_np = (denorm(video).cpu().numpy() * 255).astype('uint8')
	imageio.mimwrite(output_path, vid_np, fps=fps, codec='libx264', quality=8)

	return output_path


def animation(gen, chunk_size, device):
	
	@torch.inference_mode()
	def edit_media(image, *selected_s):

		image_tensor, w, h = img_preprocessing(image, 512)
		image_tensor = image_tensor.to(device)

		edited_image_tensor = gen.edit_img(image_tensor, labels_v, selected_s)

		# de-norm
		edited_image = img_postprocessing(edited_image_tensor, w, h)

		return edited_image

	@torch.inference_mode()
	def animate_media(image, video, *selected_s):

		image_tensor, w, h = img_preprocessing(image, 512)
		vid_target_tensor, fps = vid_preprocessing(video, 512)
		image_tensor = image_tensor.to(device) # bchw
		video_target_tensor = vid_target_tensor.to(device) # tchw

		animated_video = gen.animate(image_tensor, video_target_tensor, labels_v, selected_s, chunk_size) # tchw
		edited_image = animated_video[0:1,:,:,:] # bchw

		# postprocessing
		animated_video = vid_postprocessing(animated_video, w, h, fps)
		edited_image = img_postprocessing(edited_image, w, h)
		return edited_image, animated_video


	def clear_media():
		return None, None, *([0] * len(labels_k))

	
	with gr.Tab("Image Animation"):

		inputs_s = []

		with gr.Row():
			with gr.Column(scale=1):
				with gr.Row():
					with gr.Accordion(open=True, label="Source Image"):
						image_input = gr.Image(type="filepath", elem_id="input_img", width=512)	# , height=550)
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

					with gr.Accordion(open=True, label="Driving Video"):
						video_input = gr.Video(width=512, elem_id="input_vid",)  # , height=550)
						gr.Examples(
							examples=[
								["./data/driving/driving6.mp4"],
								["./data/driving/driving1.mp4"],
								["./data/driving/driving2.mp4"],
								["./data/driving/driving4.mp4"],
								["./data/driving/driving8.mp4"],
							],
							inputs=[video_input],
							cache_examples=False,
							visible=True,
							)

				with gr.Row():
					with gr.Column(scale=1):
						with gr.Row():	# Buttons now within a single Row
							edit_btn = gr.Button("Edit", elem_id="button_edit",)
							clear_btn = gr.Button("Clear", elem_id="button_clear")
						with gr.Row():
							animate_btn = gr.Button("Animate", elem_id="button_animate")



			with gr.Column(scale=1):

				with gr.Row():
					with gr.Accordion(open=True, label="Edited Source Image"):
						image_output = gr.Image(label="Output Image", elem_id="output_img", type='numpy', interactive=False, width=512)


					with gr.Accordion(open=True, label="Animated Video"):
						video_output = gr.Video(label="Output Video", elem_id="output_vid", width=512)

				with gr.Accordion("Control Panel", open=True):
					with gr.Tab("Head"):
						with gr.Row():
							for k in labels_k[:3]:
								slider = gr.Slider(minimum=-1.0, maximum=0.5, value=0, label=k, elem_id="slider_"+str(k))
								inputs_s.append(slider)
						with gr.Row():
							for k in labels_k[3:6]:
								slider = gr.Slider(minimum=-0.5, maximum=0.5, value=0, label=k, elem_id="slider_"+str(k))
								inputs_s.append(slider)

					with gr.Tab("Mouth"):
						with gr.Row():
							for k in labels_k[6:8]:
								slider = gr.Slider(minimum=-0.4, maximum=0.4, value=0, label=k, elem_id="slider_"+str(k))
								inputs_s.append(slider)
						with gr.Row():
							for k in labels_k[8:10]:
								slider = gr.Slider(minimum=-0.4, maximum=0.4, value=0, label=k, elem_id="slider_"+str(k))
								inputs_s.append(slider)

					with gr.Tab("Eyes"):
						with gr.Row():
							for k in labels_k[10:12]:
								slider = gr.Slider(minimum=-0.4, maximum=0.4, value=0, label=k, elem_id="slider_"+str(k))
								inputs_s.append(slider)
						with gr.Row():
							for k in labels_k[12:14]:
								slider = gr.Slider(minimum=-0.2, maximum=0.2, value=0, label=k, elem_id="slider_"+str(k))
								inputs_s.append(slider)


		edit_btn.click(
			fn=edit_media,
			inputs=[image_input] + inputs_s,
			outputs=[image_output],
			show_progress=True
		)

		animate_btn.click(
			fn=animate_media,
			inputs=[image_input, video_input] + inputs_s,
			outputs=[image_output, video_output],
			show_progress=True
		)

		clear_btn.click(
			fn=clear_media,
			outputs=[image_output, video_output] + inputs_s
		)

		gr.Examples(
			examples=[
				['./data/source/macron.png', './data/driving/driving6.mp4',-0.37,-0.34,0,0,0,0,0,0,0,0,0,0,0,0],
				['./data/source/taylor.png', './data/driving/driving6.mp4', -0.31, -0.2, 0, -0.26, -0.14, 0, 0.068, 0.131, 0, 0, 0,
				 0, -0.058, 0.087],
				['./data/source/macron.png', './data/driving/driving1.mp4', 0.14,0,-0.26,-0.29,-0.11,0,-0.13,-0.18,0,0,0,0,-0.02,0.07],
				['./data/source/portrait3.png', './data/driving/driving1.mp4', -0.03,0.21,-0.31,-0.12,-0.11,0,-0.05,-0.16,0,0,0,0,-0.02,0.07],
				['./data/source/einstein.png','./data/driving/driving2.mp4',-0.31,0,0,0.16,0.08,0,-0.07,0,0.13,0,0,0,0,0],
				['./data/source/portrait1.png', './data/driving/driving4.mp4', 0, 0, -0.17, -0.19, 0.25, 0, 0, -0.086,
				 0.087, 0, 0, 0, 0, 0],
				['./data/source/portrait2.png','./data/driving/driving8.mp4',0,0,-0.25,0,0,0,0,0,0,0.126,0,0,0,0],
				
			],
			inputs=[image_input, video_input] + inputs_s,
		)


