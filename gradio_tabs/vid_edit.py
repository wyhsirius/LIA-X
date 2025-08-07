import gradio as gr
import os
import torch
import torchvision
from PIL import Image
import numpy as np
import imageio
from einops import rearrange

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
	'"O" mouth',
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


def vid_all_save(vid_d, vid_a, fps, output_path=output_dir + "/output_vid.mp4", output_all_path=output_dir + "/output_all_vid.mp4"):

	vid_d = rearrange(vid_d, 'b t c h w -> b t h w c')
	vid_a = rearrange(vid_a, 'b c t h w -> b t h w c')
	vid_all = torch.cat([vid_d, vid_a], dim=3)

	vid_a_np = (vid_denorm(vid_a[0]).numpy() * 255).astype('uint8')
	vid_all_np = (vid_denorm(vid_all[0]).numpy() * 255).astype('uint8')

	imageio.mimwrite(output_path, vid_a_np, fps=fps, codec='libx264', quality=8)
	imageio.mimwrite(output_all_path, vid_all_np, fps=fps, codec='libx264', quality=8)

	return output_path, output_all_path


@torch.no_grad()
def edit_img(video, *selected_s):

	vid_target_tensor, fps = vid_preprocessing(video, 512)
	video_target_tensor = vid_target_tensor.to(device)
	image_tensor = video_target_tensor[:,0,:,:,:]

	edited_image_tensor = gen.edit_img(image_tensor, labels_v, selected_s)

	# de-norm
	edited_image = img_postprocessing(edited_image_tensor)

	return edited_image


@torch.no_grad()
def edit_vid(video, *selected_s):

	video_target_tensor, fps = vid_preprocessing(video, 512)
	video_target_tensor = video_target_tensor.to(device)

	edited_video_tensor = gen.edit_vid(video_target_tensor, labels_v, selected_s)

	# de-norm
	animated_video, animated_all_video = vid_all_save(video_target_tensor, edited_video_tensor, fps)

	return animated_video, animated_all_video


def clear_media():
	return None, None, None, *([0] * len(labels_k))


image_output = gr.Image(label="Image", type='numpy', interactive=False, width=512)
video_output = gr.Video(label="Video", width=512)
video_all_output = gr.Video(label="Videos")


def vid_edit():
	with gr.Tab("Video Editing"):

		inputs_c = []
		inputs_s = []

		with gr.Row():
			with gr.Column(scale=1):
				with gr.Row():
					with gr.Accordion(open=True, label="Video"):
						video_input = gr.Video(width=512)  # , height=550)
						gr.Examples(
							examples=[
								["./data/driving/driving1.mp4"],
								["./data/driving/driving2.mp4"],
								["./data/driving/driving4.mp4"],
								#["./data/driving/driving5.mp4"],
								#["./data/driving/driving6.mp4"],
								#["./data/driving/driving7.mp4"],
								["./data/driving/driving3.mp4"],
								["./data/driving/driving8.mov"],
								["./data/driving/driving9.mov"],
							],
							inputs=[video_input],
							cache_examples=False,
							visible=True,
						)

				# with gr.Row():
				# 	with gr.Column(scale=1):
				# 		with gr.Row():	# Buttons now within a single Row
				# 			edit_btn = gr.Button("Edit")
				# 			clear_btn = gr.Button("Clear")
				# 		with gr.Row():
				# 			animate_btn = gr.Button("Generate")

			with gr.Column(scale=2):

				with gr.Row():
					with gr.Accordion(open=True, label="Edited First Frame"):
						image_output.render()

					with gr.Accordion(open=True, label="Edited Video"):
						video_output.render()
				
				with gr.Row():
					with gr.Accordion(open=True, label="Original & Edited Videos"):
						video_all_output.render()
			
			with gr.Column(scale=1):
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

				with gr.Row():
					with gr.Column(scale=1):
						with gr.Row():	# Buttons now within a single Row
							edit_btn = gr.Button("Edit")
							clear_btn = gr.Button("Clear")
						with gr.Row():
							animate_btn = gr.Button("Generate")

		edit_btn.click(
			fn=edit_img,
			inputs=[video_input] + inputs_s,
			outputs=[image_output],
			show_progress=True
		)

		animate_btn.click(
			fn=edit_vid,
			inputs=[video_input] + inputs_s,  # [image_input, video_input] + inputs_s,
			outputs=[video_output, video_all_output],
		)

		clear_btn.click(
			fn=clear_media,
			outputs=[image_output, video_output, video_all_output] + inputs_s
		)

		gr.Examples(
			examples=[
				['./data/driving/driving1.mp4', 0.5, 0.5, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0, 0],
				['./data/driving/driving2.mp4', 0.5, 0.5, 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0],
				['./data/driving/driving1.mp4', 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, -0.3, 0, 0],
				['./data/driving/driving3.mp4', -0.6, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, 0, 0],
				['./data/driving/driving9.mov', 0, 0, 0, 0, 0, 0, 0,
				 0, 0, 0, 0, 0, -0.1, 0.07],
			],
			inputs=[video_input] + inputs_s
		)



