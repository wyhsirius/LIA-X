import os
import torch
import argparse
import yaml
from pathlib import Path
from utils.data_processing import img_preprocessing, vid_preprocessing, save_animation, save_img_edit, save_vid_edit, save_linear_manipulation

extensions_dir = "./torch_extension/"
os.environ["TORCH_EXTENSIONS_DIR"] = extensions_dir

from networks.generator import Generator

@torch.no_grad()
def run_linear_manipulation(cfg, gen, save_dir):

	print("==> loading data")
	img = img_preprocessing(cfg["source_path"], cfg["size"])
	img = img.to(device)

	print("==> running")
	res = gen.interpolate_img(img, cfg["motion_id"], cfg["motion_value"])
	# save results
	save_linear_manipulation(save_dir, res, fps=12)

	return


@torch.no_grad()
def run_img_edit(cfg, gen, save_dir):

	print("==> loading data")
	print("image path: ", cfg["source_path"])
	img = img_preprocessing(cfg["source_path"], cfg["size"])
	img = img.to(device)

	print("==> running")
	img_edit = gen.edit_img(img, cfg["motion_id"], cfg["motion_value"])
	# save results
	save_img_edit(save_dir, img, img_edit)
	print("save path: ", save_dir)

	return


@torch.no_grad()
def run_vid_edit(cfg, gen, chunk_size, save_dir):

	print("==> loading data")
	print("video path: ", cfg["driving_path"])
	vid, fps = vid_preprocessing(cfg["driving_path"], cfg["size"])
	vid = vid.to(device)

	print("==> running")
	if chunk_size == 1:
		vid_edit = gen.edit_vid(vid, cfg["motion_id"], cfg["motion_value"])
	else:
		vid_edit = gen.edit_vid_batch(vid, cfg["motion_id"], cfg["motion_value"], chunk_size)

	# save results
	save_vid_edit(save_dir, vid, vid_edit, fps=fps)
	print("save path: ", save_dir)

	return


@torch.no_grad()
def run_animation(cfg, gen, chunk_size, save_dir):

	print("==> loading data")
	print("image path: ", cfg["source_path"])
	print("video path: ", cfg["driving_path"])
	img_source = img_preprocessing(cfg["source_path"], cfg["size"])
	vid_driving, fps = vid_preprocessing(cfg["driving_path"], cfg["size"])
	img_source = img_source.to(device) # BCHW
	vid_driving = vid_driving.to(device) # BTCHW

	print("==> running")
	if chunk_size == 1:
		vid_animated = gen.animate(img_source, vid_driving, cfg["motion_id"], cfg["motion_value"])
	else:
		vid_animated = gen.animate_batch(img_source, vid_driving, cfg["motion_id"], cfg["motion_value"], chunk_size)
	
	# save results
	save_animation(save_dir, img_source, vid_driving, vid_animated, fps)
	print("save path: ", save_dir)

	return


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--cfg", type=str, default='')
	parser.add_argument("--mode", type=str, choices=['animation','img_edit','vid_edit','manipulation'])
	parser.add_argument("--chunk_size", type=int, default=8) # process multiple frames each iteration, accelerate inference speed
	args = parser.parse_args()

	# loading cfg
	with open(args.cfg) as f:
		cfg = yaml.load(f, Loader=yaml.FullLoader)

	# loading model
	device = torch.device("cuda")
	gen = Generator(size=cfg['size'], motion_dim=cfg['motion_dim'], scale=cfg['scale']).to(device)
	gen.load_state_dict(torch.load(cfg['ckpt_path'], weights_only=False))
	gen.eval()

	# save dir
	save_dir = os.path.join(cfg['save_path'], Path(args.cfg).stem)
	os.makedirs(save_dir, exist_ok=True)

	# running
	if args.mode == "animation":
		print("==> start animation")
		run_animation(cfg, gen, args.chunk_size, save_dir)
	elif args.mode == "vid_edit":
		print("==> start video editing")
		run_vid_edit(cfg, gen, args.chunk_size, save_dir)
	elif args.mode == "img_edit":
		print("==> start image editing")
		run_img_edit(cfg, gen, save_dir)
	elif args.mode == "manipulation":
		print("==> start linear manipulation")
		run_linear_manipulation(cfg, gen, save_dir)
	else:
		raise NotImplementedError
