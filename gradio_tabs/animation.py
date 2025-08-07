import gradio as gr
import os
import torch
import torchvision
from PIL import Image
import numpy as np
import imageio

extensions_dir = "./torch_extension/"
os.environ["TORCH_EXTENSIONS_DIR"] = extensions_dir

from networks.generator import Generator

device = torch.device("cuda")
ckpt_path = './models/lia-x.pt'
gen = Generator(size=512, motion_dim=40, scale=2).to(device)
gen.load_state_dict(torch.load(ckpt_path, weights_only=False))
gen.eval()

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
	# img = Image.open(filename).convert('RGB')
	if not isinstance(img, np.ndarray):
		img = Image.open(img).convert('RGB')
		img = img.resize((size, size))
		img = np.asarray(img)
	img = np.transpose(img, (2, 0, 1))	# 3 x 256 x 256

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


def img_postprocessing(image, output_path=output_dir + "/output_img.png"):

	image = image.permute(0, 2, 3, 1)
	edited_image = img_denorm(image)
	img_output = (edited_image[0].numpy() * 255).astype(np.uint8)
	imageio.imwrite(output_path, img_output, quality=6)

	return output_path


def vid_postprocessing(video, fps, output_path=output_dir + "/output_vid.mp4"):
	# video: BCTHW

	vid = video.permute(0, 2, 3, 4, 1)	# B T H W C
	vid_np = (vid_denorm(vid[0]).numpy() * 255).astype('uint8')
	imageio.mimwrite(output_path, vid_np, fps=fps, codec='libx264', quality=10)

	return output_path


@torch.no_grad()
def edit_media(image, *selected_s):

	image_tensor = img_preprocessing(image, 512)
	image_tensor = image_tensor.to(device)

	edited_image_tensor = gen.edit_img(image_tensor, labels_v, selected_s)

	# de-norm
	edited_image = img_postprocessing(edited_image_tensor)

	return edited_image


@torch.no_grad()
def animate_media(image, video, *selected_s):

	image_tensor = img_preprocessing(image, 512)
	vid_target_tensor, fps = vid_preprocessing(video, 512)
	image_tensor = image_tensor.to(device)
	video_target_tensor = vid_target_tensor.to(device)

	animated_video = gen.animate(image_tensor, video_target_tensor, labels_v, selected_s)

	# postprocessing
	animated_video = vid_postprocessing(animated_video, fps)

	return animated_video


def clear_media():
	return None, None, *([0] * len(labels_k))


image_output = gr.Image(label="Output Image", type='numpy', interactive=False, width=512)
video_output = gr.Video(label="Output Video", width=512)


def animation():
	with gr.Tab("Animation & Image Editing"):

		inputs_s = []

		with gr.Row():
			with gr.Column(scale=1):
				with gr.Row():
					with gr.Accordion(open=True, label="Source Image"):
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

					with gr.Accordion(open=True, label="Driving Video"):
						video_input = gr.Video(width=512)  # , height=550)
						gr.Examples(
							examples=[
								["./data/driving/driving1.mp4"],
								["./data/driving/driving2.mp4"],
								["./data/driving/driving4.mp4"],
								#["./data/driving/driving5.mp4"],
								["./data/driving/driving6.mp4"],
								#["./data/driving/driving7.mp4"],
								["./data/driving/driving8.mov"],
							],
							inputs=[video_input],
							cache_examples=False,
							visible=True,
							)

				with gr.Row():
					with gr.Column(scale=1):
						with gr.Row():	# Buttons now within a single Row
							edit_btn = gr.Button("Edit")
							clear_btn = gr.Button("Clear")
						with gr.Row():
							animate_btn = gr.Button("Animate")



			with gr.Column(scale=1):

				with gr.Row():
					with gr.Accordion(open=True, label="Edited Source Image"):
						image_output.render()

					with gr.Accordion(open=True, label="Animated Video"):
						video_output.render()

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
			fn=edit_media,
			inputs=[image_input] + inputs_s,
			outputs=[image_output],
			show_progress=True
		)

		animate_btn.click(
			fn=animate_media,
			inputs=[image_input, video_input] + inputs_s,  # [image_input, video_input] + inputs_s,
			outputs=[video_output],
		)

		clear_btn.click(
			fn=clear_media,
			outputs=[image_output, video_output] + inputs_s
		)

		gr.Examples(
			examples=[
				['./data/source/macron.png', './data/driving/driving1.mp4', 0.14,0,-0.26,-0.29,-0.11,0,-0.13,-0.18,0,0,0,0,-0.02,0.07],
				['./data/source/portrait1.png', './data/driving/driving2.mp4', -0.1, 0, 0, 0.17, 0.16, 0, 0.01, 0, 0.17,0.17, 0, 0, 0, 0],
				['./data/source/macron.png', './data/driving/driving4.mp4', -0.24, -0.17, -0.15, 0, 0, 0, 0, -0.16,
				 0.08, 0, 0, 0, 0, 0],
				['./data/source/portrait2.png', './data/driving/driving3.mp4', 0.33, 0.38, -0.22, 0.25, -0.23, 0, -0.16,
				 0, 0.06, 0, 0, 0, 0, 0],
				['./data/source/portrait2.png', './data/driving/driving6.mp4', -0.27, -0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0],
				['./data/source/portrait2.png','./data/driving/driving1.mp4',-0.03,0.21,-0.41,-0.29,-0.11,0,0,-0.23,0,0,0,0,-0.02,0.07],
				['./data/source/portrait3.png','./data/driving/driving1.mp4',-0.03,0.21,-0.31,-0.12,-0.11,0,-0.05,-0.16,0,0,0,0,-0.02,0.07],
				['./data/source/portrait1.png','./data/driving/driving1.mp4',-0.03,0.21,-0.31,-0.12,-0.11,0,-0.1,-0.12,0,0.11,0,0,-0.02,0.07],
				['./data/source/einstein.png','./data/driving/driving2.mp4',-0.31,0,0,0.16,0.08,0,-0.07,0,0.13,0,0,0,0,0],
				['./data/source/einstein.png', './data/driving/driving4.mp4',0,0,0,0,0,0,0,-0.14,0.1,0,0,0,0,0],
				['./data/source/portrait1.png', './data/driving/driving4.mp4',0,0,0,0,0,0,0,-0.1,0.19,0,0,0,0,0],
				['./data/source/macron.png', './data/driving/driving6.mp4',-0.37,-0.34,0,0,0,0,0,0,0,0,0,0,0,0],
			],
			inputs=[image_input, video_input] + inputs_s
		)


