import torch
import numpy as np
import transforms
import preprocess
import utils
import models
import re
import os

######################## Test Parameters ########################

volpath = 'volume folder'
segpath = 'segmentation folder'
out_file = 'destination for output files'
checkpoint_path = 'path to model checkpoint'
batch_size = 20
pre_crop_dim = 256
post_crop_dim = 224
num_labels=2
cuda = True

#####################################################################

if not os.path.exists(out_file):
    os.mkdir(out_file)
model = models.Net(num_labels)
if cuda:
    model.cuda()

model.load_state_dict(torch.load(checkpoint_path))

sqr = transforms.Square()
center = transforms.CenterCrop2(post_crop_dim)
scale = transforms.Scale(pre_crop_dim)
iRescale = transforms.IntensityRescale(1200, 1300, 0.1, 0.1, noise_chance=0)
transform_plan = [sqr, scale, center, iRescale]
series_names = ['V']
seg_series_names = ['S']

f_s = preprocess.gen_filepaths(segpath)
f_v = preprocess.gen_filepaths(volpath)

mult_inds = []
for i in f_s:
    if 'segmentation' in i:
        mult_inds.append(int(re.findall('\d+', i)[0]))

mult_inds = sorted(mult_inds)

mult_inds = np.unique(mult_inds)

volpaths, segpaths = utils.get_paths(mult_inds, f_s, f_v, series_names, \
        seg_series_names, volpath, segpath)

out = utils.test_net_cheap(mult_inds, 0, model,\
        transform_plan, pre_crop_dim, batch_size, out_file, num_labels,\
        volpaths, segpaths, nrrd=True, vol_only=False,\
        get_dice=True, make_niis=True, cuda=cuda)
out_csv = os.path.join(checkpoint_path, 'out.csv')
out.to_csv(out_csv, index=False)
