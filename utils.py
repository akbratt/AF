import preprocess
import transforms
import numpy as np
import torch
from torch.autograd import Variable
import nibabel as nib
import sys
import time
import skimage.segmentation as skseg
import sklearn.metrics.pairwise as pair
import nrrd
import re
import os
from skimage import color
import pandas as pd
import torch.nn.functional as F

def get_liver(seg):
    seg = seg.data.cpu().numpy()
    bleh = np.array([len(np.unique(s)) for s in seg])
    livers = np.where(bleh>1)[0]
    ind = 0
    if livers.size is not 0:
        ind = np.random.choice(livers)
    return ind

def get_out_size(orig_dim, in_z, transform_plan, net, softmask=False, cuda=True):
    dummy = np.random.rand(1, in_z*2+1, orig_dim, orig_dim)
    for i in transform_plan:
        dummy, _ = i.engage(dummy, dummy)
    dummy = np.squeeze(dummy)

    dummy = np.stack([dummy, dummy])
    dummy = np.expand_dims(dummy, axis=1)
    dummy = Variable(torch.from_numpy(dummy).float())
    if cuda:
        dummy = dummy.cuda()

    if softmask:
        dummy = net(dummy)[-1]
    else:
        dummy = net(dummy)
    if len(dummy.size()) == 4:
        dummy = dummy.unsqueeze(2)
    out_dim = dummy.size()[-1]
    out_depth = dummy.size()[2]
    return out_depth, out_dim

def get_hot(seg, num_labels):
    hot = [np.where(seg==a, 1, 0) for a in range(num_labels)]
    hot = np.stack(hot, axis=0)
    return hot

def dice(real, fake):
    dice = np.sum(fake[real==1])*2.0 / (np.sum(fake) + np.sum(real))
    return dice

def jaccard(real, fake):
    jaccard = np.sum(fake[real==1]) / (np.sum(fake) + np.sum(real) -\
            np.sum(fake[real==1]))
    return jaccard

def open_double_vol(nii_paths):
    vols = []
    segs = []
    for nii_path in nii_paths:
        vol = nib.as_closest_canonical(nib.load(nii_path))
        vol = vol.get_data().astype(np.int16)
        vols.append(vol)
        segs.append(vol.copy())

    return vols, segs

def get_subvols_cheap(series, seg_series, slice_inds, in_z, out_z, \
        center_crop_sz, model, num_labels, batch_size, txforms=None,\
        verbose=True, cuda=True, softmask=False):

    # get beginning index of output since the z dim is smaller than vol
    z0 = (in_z*2+1 - out_z)//2

    sz = np.array([num_labels, slice_inds.shape[0]+2*in_z, center_crop_sz,\
            center_crop_sz])

    bigout = np.zeros(sz)
    bigvol = np.zeros(sz[1:])
    bigseg = np.zeros(sz[1:])

    center = transforms.CenterCrop(center_crop_sz)
    depth_center = transforms.DepthCenterCrop(out_z)
    vols = []
    segs = []
    batch_ind = 0
    absolute_ind = 0
    for i in slice_inds:
        if in_z == 0:
            nascent_series = [vol[:,:,i] for vol in series]
            nascent_seg_series = [seg[:,:,i] for seg in seg_series]
            nascent_series = np.expand_dims(nascent_series, axis=0)
            nascent_seg_series = np.expand_dims(nascent_seg_series, axis=0)
        else:
            nascent_series = [v[:,:,i-in_z:i+1+in_z] for v in series]
            assert nascent_series[0].shape[2]==in_z*2+1
            nascent_series = [np.squeeze(np.split(v,\
                    v.shape[2], axis=2)) for v in nascent_series]

            nascent_seg_series = [s[:,:,i-in_z:i+1+in_z] for s in seg_series]
            nascent_seg_series = [depth_center.engage(s) for s in\
                    nascent_seg_series]
            nascent_seg_series = [np.squeeze(np.split(s,\
                    s.shape[2], axis=2)) for s in nascent_seg_series]

            if out_z == 1:
                nascent_seg_series = \
                        np.expand_dims(nascent_seg_series, axis=0)

        if txforms is not None:
            for j in txforms:
                nascent_series, nascent_seg_series = \
                        j.engage(nascent_series, nascent_seg_series)

            vols.append(np.squeeze(nascent_series))

            segs.append(np.squeeze(center.engage(nascent_seg_series, \
                    out_z > 1)))

            absolute_ind += 1

        if (absolute_ind >= batch_size or (i >= slice_inds[-1] and vols)):
            nascent_series = np.array(vols)
            nascent_seg_series = np.array(segs)
            nascent_series = preprocess.rot_and_flip(nascent_series)
            nascent_seg_series = preprocess.rot_and_flip(nascent_seg_series)

            if len(nascent_series.shape) < 4:
                nascent_series = np.expand_dims(nascent_series, 0)

            tv = torch.from_numpy(nascent_series).float()
            tv = Variable(tv)
            if cuda:
                tv = tv.cuda()
            if verbose:
                sys.stdout.write('\r{:.2f}%'.format(i/sz[1]))
                sys.stdout.flush()
            if in_z == 0:
                tv = tv.permute(1,0,2,3)

            if softmask:
                cls_, h5_, h4_, h3_, h1_, seg_ = model(tv)
                h5_, h4_, h3_, h1_, seg_ = pare_masks(cls_, h5_, h4_,\
                        h3_, h1_, seg_)
                tout = F.interpolate(seg_.float(), (224, 224))
                tout = tout.data.cpu().numpy().astype(np.int8)
            else:
                tout = model(tv).data.cpu().numpy().astype(np.int8)

            if in_z == 0:
                nascent_series = nascent_series.squeeze()
                if np.array(nascent_series.shape).shape[0] < 3:
                    nascent_series = np.expand_dims(nascent_series, 0)
            for j in range(len(nascent_series)):

                bsz = len(nascent_series)
                beg = i - in_z + z0 - bsz + j + 1
                end = i - in_z + z0 - bsz + j + out_z + 1
                bigout[:,beg:end] += np.expand_dims(tout[j], 1)
                bigseg[beg:end] = nascent_seg_series[j]

                beg = i - in_z + 1 - bsz + j
                end = i + in_z - bsz + j + 2
                bigvol[beg:end] = nascent_series[j]

            absolute_ind = 0
            batch_ind += 1
            vols = []
            segs = []

    return bigout, bigvol, bigseg

def test_net_cheap(mult_inds, in_z, model,\
        t_transform_plan, orig_dim, batch_size, out_file, num_labels,\
        volpaths, segpaths, nrrd=True, vol_only=False,\
        get_dice=False, make_niis=False,\
        verbose=True, cuda=True, softmask=False):

    t_out_z, t_center_crop_sz = get_out_size(orig_dim, in_z,\
            t_transform_plan, model, softmask=softmask, cuda=cuda)
    t_center = transforms.CenterCrop(t_center_crop_sz)

    dices = []
    jaccards = []
    hds = []
    assds = []
    dice_inds = []
    times = []
    print(mult_inds)
    for ind in range(len(mult_inds)):
        t0 = time.time()
        if vol_only:
            series, seg_series = open_double_vol(volpaths[ind])
            seg_series = [a*0 for a in seg_series]
        else:
            series, seg_series = preprocess.get_nii_nrrd(volpaths[ind],\
                    segpaths[ind])
        num_slices = np.arange(np.shape(series[0])[2])
        if in_z == 0:
            num_slices = num_slices
        else:
            num_slices = num_slices[in_z:-in_z]

        slice_inds = num_slices
        for slice_ind in slice_inds:
            assert slice_ind >= np.min(num_slices)\
                    and slice_ind <= np.max(num_slices)

        tout, tvol, tseg = get_subvols_cheap(series, seg_series, slice_inds,\
                in_z, t_out_z, t_center_crop_sz, model, num_labels,\
                batch_size, t_transform_plan, verbose=verbose, cuda=cuda, softmask=softmask)
        duration = time.time() - t0
        tseg = np.clip(tseg, 0,1)
        times.append(duration)


        if get_dice:
            hd, assd = get_dists_non_volumetric(tseg.astype(np.int64),\
                    np.argmax(tout, axis=0))
            tseg_hot = get_hot(tseg, num_labels)
            tout_hot = np.argmax(tout,axis=0)
            tout_hot = np.clip(tout_hot, 0,1)
            tout_hot = get_hot(tout_hot, num_labels)
            dce = dice(tseg_hot[1:],tout_hot[1:])
            jc = jaccard(tseg_hot[1:], tout_hot[1:])

            if verbose:
                print(('\r{}: Duration: {:.2f} ; Dice: {:.2f} ; Jaccard: {:.2f}' +\
                        ' ; Hausdorff: {:.2f} ; ASSD: {:.2f}').format(\
                        mult_inds[ind], duration, dce, jc, np.mean(hd),\
                        np.mean(assd)))
            jaccards.append(jc)
            dices.append(dce)
            hds.append(hd)
            assds.append(assd)
            dice_inds.append(mult_inds[ind])
        else:
            if verbose:
                print('\r{}'.format(mult_inds[ind]))

        if make_niis:
            tv = np.stack(t_center.engage(np.expand_dims(tvol, 0),True))
            vol_out = tv
            vol_out = np.flip(vol_out, -1)
            vol_out = np.rot90(vol_out, k=-1, axes=(-2,-1))
            vol_out = np.transpose(vol_out,[1,2,0])
            vol_out = nib.Nifti1Image(vol_out, np.eye(4))
            nib.save(vol_out, \
                    out_file + '/tvol-{}.nii'.format(\
                    mult_inds[ind]))

            seg_out = tseg
            seg_out = np.flip(seg_out, -1)
            seg_out = np.rot90(seg_out, k=-1, axes=(-2,-1))
            seg_out = np.transpose(seg_out,[1,2,0])
            write_nrrd(seg_out.astype(np.uint8),\
                    out_file + '/tseg-{}.seg.nrrd'.format(mult_inds[ind]))

            out_out = tout
            out_out = np.argmax(out_out, axis=0)
            out_out = np.flip(out_out, -1)
            out_out = np.rot90(out_out, k=-1, axes=(-2,-1))
            out_out = np.transpose(out_out,[1,2,0])
            write_nrrd(out_out.astype(np.uint8),\
                    out_file + '/tout-{}.seg.nrrd'.format(mult_inds[ind]))

    if get_dice:
        columns = ['id','time','dice','jaccard','hd','assd']
        out = pd.DataFrame(np.stack([np.array(dice_inds),\
                np.array(times), np.array(dices), np.array(jaccards),\
                np.array(hds), np.array(assds)], axis=1), columns=columns)
        return out

        return np.array(dice_inds), np.array(dices), np.array(jaccards),\
                np.array(hds), np.array(assds),\
                np.array(times)
    else:
        return

# get hausdorff and average symmetric surface distance over a volume
def get_dists_volumetric(real, fake):
    pres = []

    for r, f in zip(real, fake):
        f_border = skseg.find_boundaries(f)
        r_border = skseg.find_boundaries(r)

        f_coords = np.argwhere(f_border == 1).copy(order='C').astype(np.float64)
        r_coords = np.argwhere(r_border == 1).copy(order='C').astype(np.float64)

        try:
            euclid_distances = pair.euclidean_distances(f_coords, r_coords)

            ab = np.min(euclid_distances, axis=1)
            ba = np.min(euclid_distances, axis=0)

            pre = np.concatenate([ab, ba])
            pres.append(pre)
        except ValueError:
            # print('surface error')
            pass

    try:
        hd = np.max(np.concatenate(pres))
        assd = np.mean(np.concatenate(pres))
        return hd, assd
    except ValueError:
        print('no valid surfaces')
        return 0, 0

def get_dists_non_volumetric(real, fake):
    pres = []
    hds = []

    for r, f in zip(real, fake):
        f_border = skseg.find_boundaries(f)
        r_border = skseg.find_boundaries(r)

        f_coords = np.argwhere(f_border == 1).copy(order='C').astype(np.float64)
        r_coords = np.argwhere(r_border == 1).copy(order='C').astype(np.float64)
        # print(r_coords.shape)
        # print(f_coords.shape)

        try:
            euclid_distances = pair.euclidean_distances(f_coords, r_coords)

            ab = np.min(euclid_distances, axis=1)
            ba = np.min(euclid_distances, axis=0)

            pre = np.concatenate([ab, ba])
            hds.append(np.max(pre))
            pres.append(pre)
        except ValueError:
            # print('surface error')
            pass

    try:
        hd = np.mean(hds)
        assd = np.mean(np.concatenate(pres))
        return hd, assd
    except ValueError:
        print('no valid surfaces')
        return 0, 0

def bounding_box(seg):
    x = np.any(np.any(seg, axis=0), axis=1)
    y = np.any(np.any(seg, axis=1), axis=1)
    z = np.any(np.any(seg, axis=1), axis=0)
    ymin, ymax = np.where(y)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    bbox = np.array([ymin,ymax,xmin,xmax,zmin,zmax])
    return bbox

def get_shape_origin(seg):
    bbox = bounding_box(seg)
    ymin, ymax, xmin, xmax, zmin, zmax = bbox
    shape = list(np.array([ymax-ymin, xmax-xmin, zmax-zmin]) + 1)
    origin = [ymin, xmin, zmin]
    return shape, origin

def sparsify(a):
    ncols = a.max() + 1
    ncols = np.maximum(ncols, 3)
    out = np.zeros( (a.size,ncols), dtype=np.uint8)
    out[np.arange(a.size),a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    out = np.transpose(out, axes=(3,0,1,2))
    return out

def write_nrrd(seg, out_file):
    options = {}
    options['dimension'] = 4
    options['encoding'] = 'gzip'
    options['kinds'] = ['complex', 'domain', 'domain', 'domain']
    options['measurement frame'] = [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
    options['space'] = 'right-anterior-superior'
    options['space directions'] = [['0', '0', '0'], ['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']]
    options['type'] = 'unsigned char'
    options['keyvaluepairs'] = {}

#    print(seg.shape)
    box = bounding_box(seg)
    seg_cut = seg[box[0]:box[1]+1,box[2]:box[3]+1,box[4]:box[5]+1]
    sparse = sparsify(seg_cut)[1:,:,:,:]
    shape, origin = get_shape_origin(seg)

    options['sizes'] = [np.max(seg), *shape]
    options['space origin'] = [str(a) for a in origin]

    keyvaluepairs = {}

    for i in range(np.maximum(np.max(seg), 2)):
        seg_slice = sparse[i]
        name = 'Segment{}'.format(i)
        keyvaluepairs[name + '_Color'] = ' '.join([str(a) for a in np.random.rand(3)])
        keyvaluepairs[name + '_ColorAutoGenerated'] = '1'
        if np.max(seg) == 1 and i == 1:
            keyvaluepairs[name + '_Extent'] = ' '.join([str(a) for a in bounding_box(sparse[i-1])])
        else:
            keyvaluepairs[name + '_Extent'] = ' '.join([str(a) for a in bounding_box(seg_slice)])
        keyvaluepairs[name + '_ID'] = 'Segment_{}'.format(i+1)
        keyvaluepairs[name + '_Name'] = 'Segment_{}'.format(i+1)
        keyvaluepairs[name + '_NameAutoGenerated'] = 1
        keyvaluepairs[name + '_Tags'] = 'TerminologyEntry:Segmentation category' +\
            ' and type - 3D Slicer General Anatomy list~SRT^T-D0050^Tissue~SRT^' +\
            'T-D0050^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'

    keyvaluepairs['Segmentation_ContainedRepresentationNames'] = 'Binary labelmap|'
    keyvaluepairs['Segmentation_ConversionParameters'] = 'placeholder'
    keyvaluepairs['Segmentation_MasterRepresentation'] = 'Binary labelmap'
    keyvaluepairs['Segmentation_ReferenceImageExtentOffset'] = ' '.join(options['space origin'])

    options['keyvaluepairs'] = keyvaluepairs
    nrrd.write(out_file, sparse, options = options)

def get_paths(inds, f_s, f_v, series_names, seg_series_names, volpath, segpath):
    volpaths = []
    segpaths = []

    for i in inds:
        vol_files = []
        seg_files = []
        for name in series_names:
            for j in f_v:
                if 'volume' in j and name in j:
                    ind0 = int(re.findall('\d+', j)[0])
                    if i == ind0:
                        vol_files.append(os.path.join(volpath,j))

        volpaths.append(vol_files)

        for name in seg_series_names:
            for j in f_s:
                if 'segmentation' in j and name in j:
                    ind0 = int(re.findall('\d+', j)[0])
                    if i == ind0:
                        seg_files.append(os.path.join(segpath,j))
        segpaths.append(seg_files)
    return volpaths, segpaths

def seg2mask(seg, num_labels):
    r_mask = np.zeros((seg.shape[0], seg.shape[1], 3))
    colors = np.array([[1,0,0], [0,1,0], [0,0,1]])
    for i in np.arange(1,num_labels):
        r_mask = np.where(np.expand_dims(seg, -1)==i,colors[i-1],r_mask)
    return r_mask

def makeMask(vol, seg, num_labels, alpha):

    if 'torch' in str(type(vol)):
        if 'cuda' in str(type(vol)):
            vol = vol.data.cpu().numpy()
            seg = seg.data.cpu().numpy()
        elif 'Variable' in str(type(vol)):
            vol = vol.data.numpy()
            seg = seg.data.numpy()
        else:
            vol = vol.numpy()
            seg = seg.numpy()

    if np.max(vol) > 1:
        vol = vol - np.min(vol)
        vol = vol/np.max(vol)
    color_mask = seg2mask(seg, num_labels)

    # Construct RGB version of grey-level image
    img_color = np.dstack((vol,vol,vol))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def make_pyramid(seg):
    cls, _ = seg.max(dim=1)
    cls, _ = cls.max(dim=1)
    seg = seg.unsqueeze(1).float()

    h5 = F.max_pool2d(seg, 32, stride=32).long()
    h4 = F.max_pool2d(seg, 16, stride=16).long()
    h3 = F.max_pool2d(seg, 8, stride=8).long()
    h1 = F.max_pool2d(seg, 4, stride=4).long()

    return cls.squeeze(), h5.squeeze(), h4.squeeze(), h3.squeeze(), h1.squeeze()

def logits2mask(x, thresh):
    one = torch.FloatTensor([1]).cuda()
    zero = torch.FloatTensor([0]).cuda()
    return torch.where(F.softmax(x,1)[:,1] > thresh, one, zero).unsqueeze(1)

def pare_masks(cls, h5, h4, h3, h1, seg):
    one = torch.FloatTensor([1]).cuda()
    zero = torch.FloatTensor([0]).cuda()
    thresh = 0.5

    # print(cls)

    if len(cls.shape) < 2:
        cls = cls.unsqueeze(0)
    cls_ = torch.where(cls[:,1] > thresh, one, zero).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # print(cls_.squeeze())
    h5_out = F.interpolate(cls_, (7, 7)) * h5

    h4_out = F.interpolate(logits2mask(h5_out, 0.5), (14,14)) * h4
    # h4_out = logits2mask(h5_out, 0.5)
    h3_out = F.interpolate(logits2mask(h4_out, 0.5), (28,28)) * h3
    h1_out = F.interpolate(logits2mask(h3_out, 0.5), (56,56)) * h1
    seg_out = F.interpolate(logits2mask(h1_out, 0.5), (224,224)) * seg

    return h5_out, h4_out, h3_out, h1_out, seg_out

def pare_masks2(cls, h5, h4, h3, h1, seg,\
        cls_r, h5_r, h4_r, h3_r, h1_r):
    #this differs from the function above because it uses the real
    #segmentation maps as masks rather than the predicted masks

    h5_out = F.interpolate(cls_r.unsqueeze(1).unsqueeze(1).unsqueeze(1).float(), (7, 7)) * h5

    sizes = [(14,14), (28,28), (56,56), (224,224)]
    segs = [h4, h3, h1, seg]
    real = [h5_r, h4_r, h3_r, h1_r]

    h4_out, h3_out, h1_out, seg_out = [F.interpolate(a.unsqueeze(1).float(),\
            b) * c for a, b, c in \
            zip(real, sizes, segs)]

    return h5_out, h4_out, h3_out, h1_out, seg_out
