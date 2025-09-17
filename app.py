# Copyright (C) 2025, Shanghai AI Laboratory, Inria STARS research group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  wyhsirius@gmail.com, francois.bremond@inria.fr, antitza.dancheva@inria.fr
#

import gradio as gr
import torch

from networks.generator import Generator

device = torch.device("cuda")
gen = Generator(motion_dim=40, scale=2).to(device)
ckpt_path = './model/lia-x.pt'
gen.load_state_dict(torch.load(ckpt_path, weights_only=True))
gen.eval()

chunk_size=8 # number of frames to be generated at the same time

def load_file(path):

	with open(path, 'r', encoding='utf-8') as f:
		content  = f.read()

	return content

custom_css = """
<style>
  body {
	font-family: Georgia, serif; /* Change to your desired font */
  }
  h1 {
	color: black; /* Change title color */
  }
</style>
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

	gr.HTML(load_file("assets/title.md"))
	with gr.Row():
		with gr.Accordion(open=False, label="Instruction"):
			gr.Markdown(load_file("assets/instruction.md"))
	
	with gr.Row():
		with gr.Tabs():
			from gradio_tabs.animation import animation
			from gradio_tabs.vid_edit import vid_edit
			from gradio_tabs.img_edit import img_edit
			animation(gen, chunk_size, device)
			img_edit(gen, device)
			vid_edit(gen, chunk_size, device)

	
demo.launch(
	server_name='0.0.0.0',
	server_port=10006,
	allowed_paths=["./data/source","./data/driving"]
)
