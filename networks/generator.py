import torch
from torch import nn
from networks.encoder import Encoder
from networks.decoder import Decoder
import numpy as np
from einops import rearrange, repeat


class Generator(nn.Module):
	def __init__(self, style_dim=512, motion_dim=40, scale=1):
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

	def animate(self, img_source, vid_target, d_l, v_l, chunk_size):
		# img_source: 1xCHW
		# vid_target: TCHW

		t,c,h,w = vid_target.size()
		alpha_start = self.get_alpha(vid_target[0:1, :, :, :]) # 1x40

		z_s2r, feat_rgb = self.enc.enc_2r(img_source)
		alpha_r2s = self.enc.enc_r2t(z_s2r)
		v_l_tensor = torch.tensor(v_l, device=alpha_r2s.device, dtype=alpha_r2s.dtype).unsqueeze(0)
		alpha_r2s[:, d_l] = alpha_r2s[:, d_l] + v_l_tensor # 1x40

		chunks = (t + chunk_size - 1) // chunk_size
		bs = chunk_size

		alpha_start_r = repeat(alpha_start, 'b c -> (repeat b) c', repeat=bs)
		alpha_r2s_r = repeat(alpha_r2s, 'b c -> (repeat b) c', repeat=bs)
		feat_rgb_r = [repeat(feat, 'b c h w -> (repeat b) c h w', repeat=bs) for feat in feat_rgb]
		z_s2r_r = repeat(z_s2r, 'b c -> (repeat b) c', repeat=bs)

		vid_target_recon = []
		for i in range(chunks):
			start_index = i * bs
			end_index = start_index + bs if i < chunks - 1 else t

			img_target = vid_target[start_index:end_index, :, :, :]

			# Adjust the batch size for the last chunk
			current_bs = end_index - start_index
			alpha_start_curr = alpha_start_r[:current_bs]
			alpha_r2s_curr = alpha_r2s_r[:current_bs]
			feat_rgb_curr = [feat[:current_bs] for feat in feat_rgb_r]
			z_s2r_curr = z_s2r_r[:current_bs]

			alpha = self.enc.enc_transfer_vid(alpha_r2s_curr, img_target, alpha_start_curr)
			img_recon = self.dec(z_s2r_curr, alpha, feat_rgb_curr) # bs x 3 x h x w
			vid_target_recon.append(img_recon)
		vid_target_recon = torch.cat(vid_target_recon, dim=0) # TCHW

		return vid_target_recon

	def edit_vid(self, vid_target, d_l, v_l, chunk_size):
		# vid_target: TCHW
		
		t, c, h, w = vid_target.size()
		img_source = vid_target[0:1, :, :, :]
		alpha_start = self.get_alpha(img_source)

		z_s2r, feat_rgb = self.enc.enc_2r(img_source)
		alpha_r2s = self.enc.enc_r2t(z_s2r)
		v_l_tensor = torch.tensor(v_l, device=alpha_r2s.device, dtype=alpha_r2s.dtype).unsqueeze(0)
		alpha_r2s[:, d_l] = alpha_r2s[:, d_l] + v_l_tensor
		
		chunks = (t + chunk_size - 1) // chunk_size
		bs = chunk_size

		alpha_start_r = repeat(alpha_start, 'b c -> (repeat b) c', repeat=bs)
		alpha_r2s_r = repeat(alpha_r2s, 'b c -> (repeat b) c', repeat=bs)
		feat_rgb_r = [repeat(feat, 'b c h w -> (repeat b) c h w', repeat=bs) for feat in feat_rgb]
		z_s2r_r = repeat(z_s2r, 'b c -> (repeat b) c', repeat=bs)

		vid_target_recon = []
		for i in range(chunks):
			start_index = i * bs
			end_index = start_index + bs if i < chunks - 1 else t

			img_target = vid_target[start_index:end_index, :, :, :]

			# Adjust the batch size for the last chunk
			current_bs = end_index - start_index
			alpha_start_curr = alpha_start_r[:current_bs]
			alpha_r2s_curr = alpha_r2s_r[:current_bs]
			feat_rgb_curr = [feat[:current_bs] for feat in feat_rgb_r]
			z_s2r_curr = z_s2r_r[:current_bs]	

			alpha = self.enc.enc_transfer_vid(alpha_r2s_curr, img_target, alpha_start_curr)
			img_recon = self.dec(z_s2r_curr, alpha, feat_rgb_curr)	# bs x 3 x h x w
			vid_target_recon.append(img_recon)
		vid_target_recon = torch.cat(vid_target_recon, dim=0) # TCHW

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
			vid_target_recon.append(img_recon)

		for i in range(step):
			v_tmp = v_tmp - stride
			alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
			img_recon = self.dec(z_s2r, alpha, feat_rgb)
			vid_target_recon.append(img_recon)

		if (v_l[6]!=0) or (v_l[7]!=0) or (v_l[8]!=0) or (v_l[9]!=0):
			for i in range(step):
				v_tmp = v_tmp + stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon)

			for i in range(step):
				v_tmp = v_tmp - stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon)
		else:
			for i in range(step):
				v_tmp = v_tmp - stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon)

			for i in range(step):
				v_tmp = v_tmp + stride
				alpha = self.enc.enc_transfer_img(z_s2r, d_l, v_tmp)
				img_recon = self.dec(z_s2r, alpha, feat_rgb)
				vid_target_recon.append(img_recon)

		vid_target_recon = torch.cat(vid_target_recon, dim=0)  # TCHW

		return vid_target_recon

