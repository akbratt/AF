import torch
import transforms
import time
import preprocess
import torch.nn.functional as F
import torch.optim as optim
import utils
import models
import sys
import os

######################## Training Parameters ########################

volpath = 'volume folder'
segpath = 'segmentation folder'
checkpoint_path = 'destination path for model checkpoints'
num_steps = 50000
batch_size = 40
pre_crop_dim = 256
post_crop_dim = 224
lr = 1e-4
num_labels = 2
print_interval = 200
cuda = True

#####################################################################

batch_size = batch_size
sqr = transforms.Square()
aff = transforms.Affine()
crop = transforms.RandomCrop(post_crop_dim)
scale = transforms.Scale(pre_crop_dim)
rotate = transforms.Rotate(0.5, 30)
noise = transforms.Noise(0.02, 0.5)
iRescale = transforms.IntensityRescale(1200, 1300, 0.1, 0.1, noise_chance=0.5)
flip = transforms.Flip()
transform_plan = [sqr, scale, aff, rotate, crop, noise, iRescale]
series_names = ['V']
seg_series_names = ['S']


model = models.Net(num_labels)

if cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=lr)

out_z, center_crop_sz = utils.get_out_size(pre_crop_dim, 0,\
        transform_plan, model)

t0 = time.time()

counter = 0
weight = torch.FloatTensor([0.1,0.90]).cuda()
model.train()
for i in range(num_steps):
    vol_, seg_, inds = preprocess.get_batch(volpath, segpath, batch_size, 0,\
            out_z, center_crop_sz, series_names, seg_series_names,\
            transform_plan, 8, nrrd=True)
    vol_ = torch.unsqueeze(vol_, 1)
    vol = vol_.cuda()
    seg = seg_.cuda()

    out = model(vol).squeeze()

    loss = F.cross_entropy(out, seg, weight=weight)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    counter += 1

    sys.stdout.write('\r{:.2f}%'.format(counter*batch_size/print_interval))
    sys.stdout.flush()
    torch.cuda.empty_cache()

    if counter*batch_size >= print_interval and i > 0:


        checkpoint_file = os.path.join(checkpoint_path,\
                'checkpoint.pth')
        torch.save(model.state_dict(),checkpoint_file)
        counter = 0

        t0 = time.time()
