# Copyright (C) 2025, Shanghai AI Laboratory, Inria STARS research group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  wyhsirius@gmail.com, francois.bremond@inria.fr, antitza.dancheva@inria.fr
#

import math
import torch
from torch import nn
from torch.nn import functional as F
from .ops import (EqualConv2d, EqualLinear, ConvLayer)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, out_channel, 3)
		self.conv2 = ConvLayer(out_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)


	def forward(self, x):
		
		h = x
		
		h = self.conv1(h)
		h = self.conv2(h)

		skip = self.skip(x)
		h = (h + skip) / math.sqrt(2)

		return h


class Encoder2R(nn.Module):
	def __init__(self, latent_dim=512, scale=1):
		super(Encoder2R, self).__init__()
		
		channels = [64*scale, 128*scale, 256*scale, 512*scale]

		# version1
		self.block1 = ConvLayer(3, channels[0], 1) # 256, 3 -> 64
		self.block2 = nn.Sequential(
			ResBlock(channels[0], channels[1])
		) # 64 -> 128
		self.block3 = nn.Sequential(
			ResBlock(channels[1], channels[2])
		) # 128 -> 256
		self.block4 = nn.Sequential(
			ResBlock(channels[2], channels[3])
		) # 256 -> 512
		self.block5 = nn.Sequential(
			ResBlock(channels[3], channels[3])
		) # 512 -> 512
		self.block6 = nn.Sequential(
			ResBlock(channels[3], channels[3])
		) # 512 -> 512
		self.block7 = nn.Sequential(
			ResBlock(channels[3], channels[3])
		) # 512 -> 512
		
		self.block_512 = ResBlock(channels[3], channels[3])
		self.block8 = EqualConv2d(channels[3], latent_dim, 4, padding=0, bias=False)

	def forward(self, x):

		res = []
		h = x
		h = self.block1(h) # 256
		res.append(h)
		h = self.block2(h) # 128
		res.append(h)
		h = self.block3(h) # 64
		res.append(h)
		h = self.block4(h) # 32
		res.append(h)
		h = self.block5(h) # 16
		res.append(h)
		h = self.block6(h) # 8
		res.append(h)
		h = self.block7(h) # 4
		res.append(h)
		h = self.block_512(h)
		h = self.block8(h) # 1

		return h.squeeze(-1).squeeze(-1), res[::-1]


class Encoder(nn.Module):
	def __init__(self, dim=512, dim_motion=20, scale=1):
		super(Encoder, self).__init__()

		# 2R netmork
		self.enc_2r = Encoder2R(dim, scale)

		# R2T
		self.enc_r2t = nn.Sequential(
			EqualLinear(dim, dim_motion)
		)
	
	def enc_motion(self, x):

		z_t2r, _ = self.enc_2r(x)
		alpha_r2t = self.enc_r2t(z_t2r)

		return alpha_r2t


	def enc_transfer_img(self, z_s2r, d_l, s_l):

		alpha_r2s = self.enc_r2t(z_s2r)
		alpha_r2s[:, d_l] = alpha_r2s[:, d_l] + torch.FloatTensor(s_l).unsqueeze(0).to('cuda')
		alpha = [alpha_r2s]

		return alpha

	def enc_transfer_vid(self, alpha_r2s, input_target, alpha_start):

		z_t2r, _ = self.enc_2r(input_target)
		alpha_r2t = self.enc_r2t(z_t2r)
		alpha = [alpha_r2t, alpha_r2s, alpha_start]

		return alpha


	def forward(self, input_source, input_target, alpha_start=None):

		if input_target is not None:

			z_s2r, feats = self.enc_2r(input_source)
			z_t2r, _ = self.enc_2r(input_target)

			alpha_r2t = self.enc_r2t(z_t2r)

			if alpha_start is not None:
				alpha_r2s = self.enc_r2t(z_s2r)
				alpha = [alpha_r2t, alpha_r2s, alpha_start]
			else:
				alpha = [alpha_r2t]

			return z_s2r, alpha, feats
		else:
			z_s2r, feats = self.enc_2r(input_source)

			return z_s2r, None, feats
