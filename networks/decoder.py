import math
import torch
from torch import nn
import torch.nn.functional as F
from .ops import (ConstantInput, ConvLayer, StyledConv, ToFlow, ToRGB, Direction)


class FlowResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, style_dim):
		super().__init__()

		self.norm = nn.GroupNorm(32, out_channel)

		self.conv1 = StyledConv(in_channel, out_channel, 3, style_dim, False)
		self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim, False)

		self.gamma = nn.Parameter(1e-5 * torch.ones([1, out_channel, 1, 1]))

	def forward(self, x, style):
		h = x
		h = self.conv1(h, style)
		skip = h

		h = self.norm(h)
		h = self.conv2(h, style)
		h = self.gamma * h

		return h + skip


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, out_channel, 3, upsample=False)
		self.conv2 = ConvLayer(out_channel, out_channel, 3, upsample=False)

		if in_channel != out_channel:
			self.skip = ConvLayer(in_channel, out_channel, 1, upsample=False, activate=False, bias=False)
		else:
			self.skip = torch.nn.Identity()

	def forward(self, x):

		h = x
		h = self.conv1(h)
		h = self.conv2(h)
		skip = self.skip(x)

		return (h + skip) / math.sqrt(2)


class Decoder(nn.Module):
	def __init__(self, style_dim, motion_dim, scale=1):
		super().__init__()
		
		channels = [512*scale, 256 * scale, 128 * scale, 64 * scale]

		self.direction = Direction(style_dim, motion_dim)

		self.input = ConstantInput(channels[0], size=4)  # 4

		# block1, 4
		self.conv1 = StyledConv(channels[0], channels[0], 3, style_dim, False)
		
		# for 512
		self.conv_512_1 = StyledConv(channels[0], channels[0], 3, style_dim, True)
		self.conv_512_2 = nn.ModuleList([
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
		])
		self.conv_512_2_rgb = nn.ModuleList([
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
		])
		self.rgb_512 = ToRGB(channels[0])
		self.flow_512 = ToFlow(channels[0], style_dim)	# 16	

		# block2, 8
		self.conv2_1 = StyledConv(channels[0], channels[0], 3, style_dim, True)
		self.conv2_2 = nn.ModuleList([
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
		])
		self.conv2_2_up = ConvLayer(channels[0], channels[0], 3, upsample=True)
		self.conv2_2_rgb = nn.ModuleList([
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
		])
		self.rgb2 = ToRGB(channels[0])
		self.flow2 = ToFlow(channels[0], style_dim)  # 16

		# block3, 16
		self.conv3_1 = StyledConv(channels[0], channels[0], 3, style_dim, True)
		self.conv3_2 = nn.ModuleList([
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
		])
		self.conv3_2_up = ConvLayer(channels[0], channels[0], 3, upsample=True)
		self.conv3_2_rgb = nn.ModuleList([
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
		])
		self.rgb3 = ToRGB(channels[0])
		self.flow3 = ToFlow(channels[0], style_dim)  # 32

		# block4, 32
		self.conv4_1 = StyledConv(channels[0], channels[0], 3, style_dim, True)
		self.conv4_2 = nn.ModuleList([
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
			FlowResBlock(channels[0], channels[0], style_dim),
		])
		self.conv4_2_up = ConvLayer(channels[0], channels[0], 3, upsample=True)
		self.conv4_2_rgb = nn.ModuleList([
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
			ResBlock(channels[0], channels[0]),
		]) 
		self.rgb4 = ToRGB(channels[0])
		self.flow4 = ToFlow(channels[0], style_dim)  # 64

		# block5, 64
		self.conv5_1 = StyledConv(channels[0], channels[1], 3, style_dim, True)
		self.conv5_2 = nn.ModuleList([
			FlowResBlock(channels[1], channels[1], style_dim),
			FlowResBlock(channels[1], channels[1], style_dim),
			FlowResBlock(channels[1], channels[1], style_dim),
			FlowResBlock(channels[1], channels[1], style_dim),
		])
		self.conv5_2_up = ConvLayer(channels[0], channels[1], 3, upsample=True)
		self.conv5_2_rgb = nn.ModuleList([
			ResBlock(channels[1], channels[1]),
			ResBlock(channels[1], channels[1]),
			ResBlock(channels[1], channels[1]),
			ResBlock(channels[1], channels[1]),
		])
		self.rgb5 = ToRGB(channels[1])
		self.flow5 = ToFlow(channels[1], style_dim)  # 128

		# block6, 128
		self.conv6_1 = StyledConv(channels[1], channels[2], 3, style_dim, True)
		self.conv6_2 = nn.ModuleList([
			FlowResBlock(channels[2], channels[2], style_dim),
			FlowResBlock(channels[2], channels[2], style_dim),
			FlowResBlock(channels[2], channels[2], style_dim),
			FlowResBlock(channels[2], channels[2], style_dim),
		])
		self.conv6_2_up = ConvLayer(channels[1], channels[2], 3, upsample=True)
		self.conv6_2_rgb = nn.ModuleList([
			ResBlock(channels[2], channels[2]),
			ResBlock(channels[2], channels[2]),
			ResBlock(channels[2], channels[2]),
			ResBlock(channels[2], channels[2]),
		])
		self.rgb6 = ToRGB(channels[2])
		self.flow6 = ToFlow(channels[2], style_dim)  # 128

		# block7, 256
		self.conv7_1 = StyledConv(channels[2], channels[3], 3, style_dim, True)
		self.conv7_2 = nn.ModuleList([
			FlowResBlock(channels[3], channels[3], style_dim),
			FlowResBlock(channels[3], channels[3], style_dim),
			FlowResBlock(channels[3], channels[3], style_dim),
			FlowResBlock(channels[3], channels[3], style_dim),
		])
		self.conv7_2_up = ConvLayer(channels[2], channels[3], 3, upsample=True)
		self.conv7_2_rgb = nn.ModuleList([
			ResBlock(channels[3], channels[3]),
			ResBlock(channels[3], channels[3]),
			ResBlock(channels[3], channels[3]),
			ResBlock(channels[3], channels[3]),
		])
		self.rgb7 = ToRGB(channels[3])
		self.flow7 = ToFlow(channels[3], style_dim)  # 128

	def navigation(self, z_s2r, alpha):

		if alpha is not None:
			# generating moving directions
			if len(alpha) > 1:
				z_r2t = self.direction(alpha[0])  # target
				z_r2s = self.direction(alpha[1])  # source
				z_start = self.direction(alpha[2])	# start
				z_s2t = z_s2r + (z_r2t - z_start) + z_r2s
			else:
				z_r2t = self.direction(alpha[0])
				z_s2t = z_s2r + z_r2t  # wa + directions
		else:
			z_s2t = z_s2r

		return z_s2t

	def apply_flow(self, h, mask, flow, feat):

		feat_warp = F.grid_sample(feat, flow) * mask
		h = feat_warp + (1 - mask) * h

		return feat_warp, h

	def forward(self, z_s2r, alpha, feats):
		# z_s2r: bs x style_dim
		# alpha: bs x style_dim

		z_s2t = self.navigation(z_s2r, alpha)

		h = self.input(z_s2t)
		h = self.conv1(h, z_s2t)
		
		#for 512
		h = self.conv_512_1(h, z_s2t)
		for conv in self.conv_512_2:
			h = conv(h, z_s2t)
		h_warp_512, h, h_flow_512 = self.flow_512(h, z_s2t, feats[0])
		for conv in self.conv_512_2_rgb:
			h_warp_512 = conv(h_warp_512)
		rgb_512 = self.rgb_512(h_warp_512)

		h = self.conv2_1(h, z_s2t)
		for conv in self.conv2_2:
			h = conv(h, z_s2t)
		h_warp2, h, h_flow2 = self.flow2(h, z_s2t, feats[1], h_flow_512)
		h_warp2 = h_warp2 + self.conv2_2_up(h_warp_512)
		for conv in self.conv2_2_rgb:
			h_warp2 = conv(h_warp2)
		rgb2 = self.rgb2(h_warp2, rgb_512)

		h = self.conv3_1(h, z_s2t)
		for conv in self.conv3_2:
			h = conv(h, z_s2t)
		h_warp3, h, h_flow3 = self.flow3(h, z_s2t, feats[2], h_flow2)
		h_warp3 = h_warp3 + self.conv3_2_up(h_warp2)
		for conv in self.conv3_2_rgb:
			h_warp3 = conv(h_warp3)
		rgb3 = self.rgb3(h_warp3, rgb2)

		h = self.conv4_1(h, z_s2t)
		for conv in self.conv4_2:
			h = conv(h, z_s2t)
		h_warp4, h, h_flow4 = self.flow4(h, z_s2t, feats[3], h_flow3)
		h_warp4 = h_warp4 + self.conv4_2_up(h_warp3)
		for conv in self.conv4_2_rgb:
			h_warp4 = conv(h_warp4)
		rgb4 = self.rgb4(h_warp4, rgb3)

		h = self.conv5_1(h, z_s2t)
		for conv in self.conv5_2:
			h = conv(h, z_s2t)
		h_warp5, h, h_flow5 = self.flow5(h, z_s2t, feats[4], h_flow4)
		h_warp5 = h_warp5 + self.conv5_2_up(h_warp4)
		for conv in self.conv5_2_rgb:
			h_warp5 = conv(h_warp5)
		rgb5 = self.rgb5(h_warp5, rgb4)

		h = self.conv6_1(h, z_s2t)
		for conv in self.conv6_2:
			h = conv(h, z_s2t)
		h_warp6, h, h_flow6 = self.flow6(h, z_s2t, feats[5], h_flow5)
		h_warp6 = h_warp6 + self.conv6_2_up(h_warp5)
		for conv in self.conv6_2_rgb:
			h_warp6 = conv(h_warp6)
		rgb6 = self.rgb6(h_warp6, rgb5)

		h = self.conv7_1(h, z_s2t)
		for conv in self.conv7_2:
			h = conv(h, z_s2t)
		h_warp7, h, h_flow7 = self.flow7(h, z_s2t, feats[6], h_flow6)
		h_warp7 = h_warp7 + self.conv7_2_up(h_warp6)
		for conv in self.conv7_2_rgb:
			h_warp7 = conv(h_warp7)
		out = self.rgb7(h_warp7, rgb6)

		return out
