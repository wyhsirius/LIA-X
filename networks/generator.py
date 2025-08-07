import torch
from torch import nn
from networks.encoder import Encoder
from networks.decoder import Decoder
import numpy as np
from tqdm import tqdm


class Generator(nn.Module):
	def __init__(self, size, style_dim=512, motion_dim=40, scale=1):
		super(Generator, self).__init__()

		style_dim = style_dim * scale

		# encoder
		self.enc = Encoder(style_dim, motion_dim, scale)
		self.dec = Decoder(style_dim, motion_dim, scale)
	
	def get_alpha(self, x):
		return self.enc.enc_motion(x)

	def edit_img(self, img_source, d_l, v_l):

		z_s2r, feat_rgb = self.enc.enc_2r(img_source)
		alpha_r2s = self.enc.enc_r2t(z_s2r)
		alpha_r2s[:, d_l] = alpha_r2s[:, d_l] + torch.FloatTensor(v_l).unsqueeze(0).to('cuda')
		img_recon = self.dec(z_s2r, [alpha_r2s], feat_rgb)

		return img_recon

	def animate(self, img_source, vid_target, d_l, v_l):

		alpha_start = self.get_alpha(vid_target[:, 0, :, :, :])

		vid_target_recon = []
		z_s2r, feat_rgb = self.enc.enc_2r(img_source)
		alpha_r2s = self.enc.enc_r2t(z_s2r)
		alpha_r2s[:, d_l] = alpha_r2s[:, d_l] + torch.FloatTensor(v_l).unsqueeze(0).to('cuda')

		for i in tqdm(range(vid_target.size(1))):
			img_target = vid_target[:, i, :, :, :]
			alpha = self.enc.enc_transfer_vid(alpha_r2s, img_target, alpha_start)
			img_recon = self.dec(z_s2r, alpha, feat_rgb)
			vid_target_recon.append(img_recon.unsqueeze(2))
		vid_target_recon = torch.cat(vid_target_recon, dim=2) # BCTHW

		return vid_target_recon

	def edit_vid(self, vid_target, d_l, v_l):

		img_source = vid_target[:, 0, :, :, :]
		alpha_start = self.get_alpha(vid_target[:, 0, :, :, :])

		vid_target_recon = []
		z_s2r, feat_rgb = self.enc.enc_2r(img_source)
		alpha_r2s = self.enc.enc_r2t(z_s2r)
		alpha_r2s[:, d_l] = alpha_r2s[:, d_l] + torch.FloatTensor(v_l).unsqueeze(0).to('cuda')

		for i in tqdm(range(vid_target.size(1))):
			img_target = vid_target[:, i, :, :, :]
			alpha = self.enc.enc_transfer_vid(alpha_r2s, img_target, alpha_start)
			img_recon = self.dec(z_s2r, alpha, feat_rgb)
			vid_target_recon.append(img_recon.unsqueeze(2))
		vid_target_recon = torch.cat(vid_target_recon, dim=2) # BCTHW

		return vid_target_recon


	def interpolate_img(self, img_source, d_l, v_l):

		vid_target_recon = []

		step = 16
		v_start = np.array([0.] * len(v_l))
		v_end = np.array(v_l)
		stride = (v_end - v_start) / step

		z_s2r, feat_rgb = self.enc.enc_2r(img_source)

		v_tmp = v_start
		for i in range(step):
			v_tmp = v_tmp + stride
			alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
			img_recon = self.dec(z_s2r, alpha, feat_rgb)
			vid_target_recon.append(img_recon.unsqueeze(2))

		for i in range(step):
			v_tmp = v_tmp - stride
			alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
			img_recon = self.dec(z_s2r, alpha, feat_rgb)
			vid_target_recon.append(img_recon.unsqueeze(2))

		if (v_l[6]!=0) or (v_l[7]!=0) or (v_l[8]!=0) or (v_l[9]!=0):
			for i in range(step):
				v_tmp = v_tmp + stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon.unsqueeze(2))

			for i in range(step):
				v_tmp = v_tmp - stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon.unsqueeze(2))
		else:
			for i in range(step):
				v_tmp = v_tmp - stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon.unsqueeze(2))

			for i in range(step):
				v_tmp = v_tmp + stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon.unsqueeze(2))

		vid_target_recon = torch.cat(vid_target_recon, dim=2)  # BCTHW

		return vid_target_recon

