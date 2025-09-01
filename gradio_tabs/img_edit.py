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
	img = np.transpose(img, (2, 0, 1))

	return img / 255.0, w, h


def img_preprocessing(img_path, size):
	img, w, h = load_image(img_path, size)  # [0, 1]
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


def clear_media():
	return None, *([0] * len(labels_k))


def img_edit(gen, device):

	@torch.inference_mode()
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
						image_input = gr.Image(type="filepath", width=512)
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
