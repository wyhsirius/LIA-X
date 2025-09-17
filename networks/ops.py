# Copyright (C) 2025, Shanghai AI Laboratory, Inria STARS research group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  wyhsirius@gmail.com, francois.bremond@inria.fr, antitza.dancheva@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .op import (FusedLeakyReLU, fused_leaky_relu, upfirdn2d)
import numpy as np


def make_kernel(k):
	k = torch.tensor(k, dtype=torch.float32)

	if k.ndim == 1:
		k = k[None, :] * k[:, None]

	k /= k.sum()

	return k


class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor=1):
		super().__init__()

		kernel = make_kernel(kernel)

		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)

		self.register_buffer('kernel', kernel)

		self.pad = pad

	def forward(self, input):
		return upfirdn2d(input, self.kernel, pad=self.pad)


class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope=0.2):
		super().__init__()

		self.negative_slope = negative_slope

	def forward(self, input):
		return F.leaky_relu(input, negative_slope=self.negative_slope)


class EqualConv2d(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

		self.stride = stride
		self.padding = padding

		if bias:
			self.bias = nn.Parameter(torch.zeros(out_channel))
		else:
			self.bias = None

	def forward(self, input):

		return F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)

	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
			f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
		)


class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
		super().__init__()

		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
		
		bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_dim])
		if bias:
			self.bias = nn.Parameter(torch.from_numpy(bias_init / lr_mul))
			#self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
		else:
			self.bias = None

		self.activation = activation

		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul

	def forward(self, input):

		if self.activation:
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)
		else:
			out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

		return out

	def __repr__(self):
		return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')


class ConvLayer(nn.Sequential):
	def __init__(
			self,
			in_channel,
			out_channel,
			kernel_size,
			downsample=False,
			upsample=False,
			blur_kernel=[1, 3, 3, 1],
			bias=True,
			activate=True,
	):
		layers = []

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

			stride = 2
			self.padding = 0
		
		elif upsample:
			layers.append(Upsample(blur_kernel))
			
			stride = 1
			self.padding = kernel_size // 2	
		else:
			stride = 1
			self.padding = kernel_size // 2

		layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride,
								  bias=bias and not activate))

		if activate:
			if bias:
				layers.append(FusedLeakyReLU(out_channel))
			else:
				layers.append(ScaledLeakyReLU(0.2))

		super().__init__(*layers)


class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3)
		self.skip = nn.Identity()

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class ResDownBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

		self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)	

	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)

		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)

		return out


class ResUpBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
		super().__init__()

		self.conv1 = ConvLayer(in_channel, out_channel, 3, upsample=True)
		self.conv2 = ConvLayer(out_channel, out_channel, 3, upsample=False)
		
		if in_channel != out_channel:
			self.skip = ConvLayer(in_channel, out_channel, 1, upsample=True, activate=False, bias=False)
		else:
			self.skip = torch.nn.Identity()	

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)

		skip = self.skip(x)
		out = (out + skip) / math.sqrt(2)

		return out


class Upsample(nn.Module):
	def __init__(self, kernel, factor=2):
		super().__init__()

		self.factor = factor
		kernel = make_kernel(kernel) * (factor ** 2)
		self.register_buffer('kernel', kernel)

		p = kernel.shape[0] - factor

		pad0 = (p + 1) // 2 + factor - 1
		pad1 = p // 2

		self.pad = (pad0, pad1)

	def forward(self, input):
		return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


class Downsample(nn.Module):
	def __init__(self, kernel, factor=2):
		super().__init__()

		self.factor = factor
		kernel = make_kernel(kernel)
		self.register_buffer('kernel', kernel)

		p = kernel.shape[0] - factor

		pad0 = (p + 1) // 2
		pad1 = p // 2

		self.pad = (pad0, pad1)

	def forward(self, input):
		return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)


class ModulatedConv2d(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False,
				 downsample=False, blur_kernel=[1, 3, 3, 1], ):
		super().__init__()

		self.eps = 1e-8
		self.kernel_size = kernel_size
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.upsample = upsample
		self.downsample = downsample

		if upsample:
			factor = 2
			p = (len(blur_kernel) - factor) - (kernel_size - 1)
			pad0 = (p + 1) // 2 + factor - 1
			pad1 = p // 2 + 1

			self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2

			self.blur = Blur(blur_kernel, pad=(pad0, pad1))

		fan_in = in_channel * kernel_size ** 2
		self.scale = 1 / math.sqrt(fan_in)
		self.padding = kernel_size // 2

		self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

		self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
		self.demodulate = demodulate
	
	def __repr__(self):
		return (
			f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
			f'upsample={self.upsample}, downsample={self.downsample})'
		)

	def forward(self, input, style):
		batch, in_channel, height, width = input.shape

		style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
		weight = self.scale * self.weight * style

		if self.demodulate:
			demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
			weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

		weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

		if self.upsample:
			input = input.view(1, batch * in_channel, height, width)
			weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
			weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel, self.kernel_size,
													self.kernel_size)
			out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
			out = self.blur(out)
		elif self.downsample:
			input = self.blur(input)
			_, _, height, width = input.shape
			input = input.view(1, batch * in_channel, height, width)
			out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
		else:
			input = input.view(1, batch * in_channel, height, width)
			out = F.conv2d(input, weight, padding=self.padding, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)

		return out


class ConstantInput(nn.Module):
	def __init__(self, channel, size=4):
		super().__init__()

		self.input = nn.Parameter(torch.randn(1, channel, size, size))

	def forward(self, input):
		batch = input.shape[0]
		out = self.input.repeat(batch, 1, 1, 1)

		return out

class StyledConv(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, demodulate=True):
		super().__init__()

		self.conv = ModulatedConv2d(
			in_channel,
			out_channel,
			kernel_size,
			style_dim,
			upsample=upsample,
			blur_kernel=[1,3,3,1],
			demodulate=demodulate,
		)

		self.activate = FusedLeakyReLU(out_channel)

	def forward(self, input, style):
		out = self.conv(input, style)
		out = self.activate(out)

		return out

class ToRGB(nn.Module):
	def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
		super().__init__()
		
		self.upsample = upsample

		if upsample:
			self.up = Upsample(blur_kernel)

		self.conv = ConvLayer(in_channel, 3, 1)
		self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

	def forward(self, input, skip=None):
		out = self.conv(input)
		out = out + self.bias

		if skip is not None:
			skip = self.up(skip)
			out = out + skip

		return out

class ToFlow(nn.Module):
	def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
		super().__init__()
		
		self.upsample = upsample
		if upsample:
			self.up = Upsample(blur_kernel)

		self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
		self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

	def forward(self, h, style, feat, skip=None):

		out = self.conv(h, style)
		out = out + self.bias

		if skip is not None:
			if self.upsample:
				skip = self.up(skip)
			out = out + skip
		
		xs = torch.linspace(-1, 1, out.size(2)).to(h.device)
		xs = torch.meshgrid(xs, xs, indexing='xy')
		xs = torch.stack(xs, 2)
		xs = xs.unsqueeze(0).repeat(out.size(0), 1, 1, 1)

		sampler = torch.tanh(out[:, 0:2, :, :])
		mask = torch.sigmoid(out[:, 2:3, :, :])
		flow = sampler.permute(0, 2, 3, 1) + xs	
		
		feat_warp = F.grid_sample(feat, flow, align_corners=True) * mask
		h = feat_warp + (1 - mask) * h

		#return h, out
		return feat_warp, h, out


class Direction(nn.Module):
	def __init__(self, style_dim, motion_dim):
		super(Direction, self).__init__()

		self.weight = nn.Parameter(torch.randn(style_dim, motion_dim))

	def forward(self, input):
		# input: (bs*t) x 512

		weight = self.weight + 1e-8
		Q, R = torch.linalg.qr(weight)	# get eignvector, orthogonal [n1, n2, n3, n4]

		input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
		out = torch.matmul(input_diag, Q.T)
		out = torch.sum(out, dim=1)

		return out




			
















































