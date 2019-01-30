import numpy as np
from PIL import Image
import PIL
import utils
import scipy.ndimage as nd
import os

# Transforms take 4d input of shape (series, slice, height, width)
def new_seed():
#	Need the next line for random seed to be different for different calls
#	when using multiprocessing
	np.random.seed(int.from_bytes(os.urandom(4), byteorder='big'))
	return


class Square(object):
	def __init__(self):
		self.a = 0

	def engage_helper(self, im):
		im = Image.fromarray(im)
		width, height = im.size
		im = np.array(im)
		x0 = np.abs(width-height)//2
		x1 = np.abs(width-height) - x0
		if width > height:
			padding = ((x0, x1), (0,0))
		else:
			padding = ((0,0), (x0, x1))
		im = np.pad(im, padding, mode='constant')
		return im

	def engage(self, vol, seg):
		newvol = []
		for series in vol:
			v = [self.engage_helper(a) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a) for a in series]
			newseg.append(s)

		return np.array(newvol), np.array(newseg)

class RandomCrop(object):

	def __init__(self, dim):
		self.dim = dim

	def engage_helper(self, vol, x0, x1, y0, y1):
		return vol[x0:x1,y0:y1]

	def engage(self, vol, seg):
		new_seed()
		shape = vol[0].shape[-2:]
		assert self.dim > np.array(shape).any()
		yoffset = shape[0] - self.dim
		xoffset = shape[1] - self.dim
		x0 = np.random.randint(xoffset+1)
		x1 = x0+self.dim
		y0 = np.random.randint(yoffset+1)
		y1 = y0+self.dim

		newvol = []
		for series in vol:
			v = [self.engage_helper(a, x0, x1, y0, y1) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a, x0, x1, y0, y1) for a in series]
			newseg.append(s)
		return np.array(newvol), np.array(newseg)

class CenterCrop(object):

	def __init__(self, dim):
		self.dim = dim

	def engage(self, vol, is_list):
		shape = vol[0].shape[-2:]
		x0 = (shape[1] - self.dim)//2
		x1 = (shape[1] + self.dim)//2
		y0 = (shape[0] - self.dim)//2
		y1 = (shape[0] + self.dim)//2

		newvol = []
		for series in vol:
			newvol.append([v[x0:x1,y0:y1] for v in series])

		return np.squeeze(newvol)

class CenterCrop2(object):

	def __init__(self, dim):
		self.dim = dim

	def engage_helper(self, vol, x0, x1, y0, y1):
		return vol[x0:x1,y0:y1]

	def engage(self, vol, seg):
		shape = vol[0].shape[-2:]
		x0 = (shape[1] - self.dim)//2
		x1 = (shape[1] + self.dim)//2
		y0 = (shape[0] - self.dim)//2
		y1 = (shape[0] + self.dim)//2

		newvol = []
		for series in vol:
			v = [self.engage_helper(a, x0, x1, y0, y1) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a, x0, x1, y0, y1) for a in series]
			newseg.append(s)

		return np.array(newvol), np.array(newseg)

class DepthCenterCrop(object):

	def __init__(self, dim):
		self.dim = dim

	def engage(self, vol):
		shape = vol.shape
		z0 = (shape[2] - self.dim)//2
		z1 = (shape[2] + self.dim)//2
		return vol[:,:,z0:z1]

class Scale(object):

	def __init__(self, dim):
		self.dim = dim

	def engage_helper(self, vol, seg=False):
		if seg:
			vol = utils.get_hot(vol.astype(np.int),
					np.max(vol.astype(np.int)+1))
			stack = []
			for v in vol:
				v = Image.fromarray(v.astype(np.float))
				v = v.resize((self.dim, self.dim), resample=1)
				v = np.array(v)
				stack.append(v)
			vol = np.array(stack)
			vol = np.argmax(stack, axis=0).astype(np.float)

		else:
			vol = Image.fromarray(vol)
			vol = vol.resize((self.dim, self.dim), resample=1)
			vol = np.array(vol)
		return vol

	def engage(self, vol, seg):
		newvol = []
		for series in vol:
			v = [self.engage_helper(a) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a, seg=True) for a in series]
			newseg.append(s)

		return np.array(newvol), np.array(newseg)

class Affine(object):

	def __init__(self, scale=0.2, chance=0.5):
		self.scale = scale
		self.chance = chance

	def engage_helper(self, vol, scale):
		center = np.array(vol.shape)[0:2]//2
		translate = (center * scale).astype(np.int) - center

		aff = np.array([[scale, 0],[0, scale]])

		tx = nd.interpolation.affine_transform(vol, aff, offset=-translate)
		return tx

	def engage(self, vol, seg):
		new_seed()
		if np.random.rand() < self.chance:
			scale = 1 + (np.random.rand()*2-1) * self.scale
		else:
			scale = 1
		newvol = []
		for series in vol:
			v = [self.engage_helper(a, scale) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a, scale) for a in series]
			newseg.append(s)

		return np.array(newvol), np.array(newseg)

class Rotate(object):

	def __init__(self, chance, max_arc):
		self.chance = chance
		self.max_arc = max_arc

	def engage_helper(self, vol, arc):
		vol = Image.fromarray(vol)
		vol = vol.rotate(arc)
		vol = np.array(vol)
		return vol

	def engage(self, vol, seg):
		new_seed()
		if np.random.rand() < self.chance:
			arc = np.random.rand() * self.max_arc
			if np.random.rand() < 0.5:
				arc *= -1

			newvol = []
			for series in vol:
				v = [self.engage_helper(a, arc) for a in series]
				newvol.append(v)

			newseg = []
			for series in seg:
				s = [self.engage_helper(a, arc) for a in series]
				newseg.append(s)

			return np.array(newvol), np.array(newseg)

		else:

			return vol, seg

class Flip(object):

	def __init__(self):
		self.i = 0

	def engage_helper(self, vol, rand1, rand2):
		vol = Image.fromarray(vol)
		if rand1:
			vol = vol.transpose(method=PIL.Image.FLIP_TOP_BOTTOM)
		if rand2:
			vol = vol.transpose(method=PIL.Image.FLIP_LEFT_RIGHT)
		vol = np.array(vol)
		return vol

	def engage(self, vol, seg):
		new_seed()
		rand1 = np.random.rand() < 0.5
		rand2 = np.random.rand() < 0.5

		newvol = []
		for series in vol:
			v = [self.engage_helper(a, rand1, rand2) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a, rand1, rand2) for a in series]
			newseg.append(s)

		return np.array(newvol), np.array(newseg)

class Pad(object):

	def __init__(self, width):
		self.width = width

	def engage_helper(self, vol):
		return np.pad(vol, self.width, 'constant')

	def engage(self, vol, seg):

		newvol = []
		for series in vol:
			v = [self.engage_helper(a) for a in series]
			newvol.append(v)

		newseg = []
		for series in seg:
			s = [self.engage_helper(a) for a in series]
			newseg.append(s)

		return np.array(newvol), np.array(newseg)

class PadEntireVol(object):

	def __init__(self, width):
		self.width = width

	def engage(self, vol, seg, in_z, out_z):
		w = self.width
		vol = np.pad(vol, ((w, w), (w, w), (0, 0)), 'constant')
		seg = np.pad(seg, ((w, w), (w, w), (0, 0)), 'constant')
		return vol, seg

class DepthPad(object):

	def __init__(self, width):
		self.width = width

	def engage(self, vol, seg, in_z, out_z):
		vol = np.pad(vol, ((0, 0), (0, 0), (self.width, self.width)), 'constant')
		seg = np.pad(seg, ((0, 0), (0, 0), (self.width, self.width)), 'constant')
		return vol, seg

class Noise(object):

	def __init__(self, chance, mag):
		self.mag = mag
		self.chance = chance

	def engage_helper(self, vol, m):
		noise = np.random.randn(*vol.shape)*m
		vol += noise
		return vol

	def engage(self, vol, seg):
		new_seed()
		if np.random.rand() < self.chance:
			dynamic_range = np.max(vol) - np.min(vol)
			m = np.random.rand()*self.mag*dynamic_range

			newvol = []
			for series in vol:
				v = [self.engage_helper(a, m) for a in series]
				newvol.append(v)

			return np.array(newvol), np.array(seg)
		else:
			return np.array(vol), np.array(seg)

class IntensityRescale(object):
	#This has to be the last transform!!!!! If it's not, noise, affines, rotations, etc will
	#cause the final dynamic range to be outside 0:1

	def __init__(self, window, level, window_noise = 0, level_noise = 0, noise_chance=0.5):
		self.window = window
		self.level = level
		self.window_noise = window_noise
		self.level_noise = level_noise
		self.noise_chance = noise_chance

	def engage(self, vol, seg):

		out_window = self.window
		out_level = self.level

		new_seed()

		if np.random.rand() < self.noise_chance:

			if self.window_noise > 0:
				out_window += self.window_noise * np.random.randn() * out_window

			if self.level_noise > 0:
				out_level += self.level_noise * np.random.randn() * out_level

		upper_bound = out_level + out_window / 2
		lower_bound = out_level - out_window / 2

		newvol = []
		for series in vol:
			nascent = [np.clip(v, lower_bound, upper_bound) for v in series]
			nascent = [v - np.min(v) for v in nascent]
			nascent = [v / np.max(v) for v in nascent]
			newvol.append(nascent)

		return np.array(newvol), np.array(seg)
