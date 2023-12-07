import os
import csv
import numpy as np
from skimage import io
from collections import  defaultdict, OrderedDict

import utils.util as util


def get_frame_dict(num_frm = 'all'):
    params = util.params
    frameseq2imgfile = params['dir_location']['frameseq2imgfile']
    img_dir = params['dir_location']['img_dir']
    frmidx2imgname = util.frameid_to_imageid(frameseq2imgfile)
    frames = OrderedDict()  # {frmidx: {"img_name": xyz.jpg, "img":numpy_image}}

    if num_frm =='all': num_frm = len(frmidx2imgname)
    for frame_idx in range(num_frm):
        img_name = frmidx2imgname[frame_idx].split('/')[-1]
        img_path = os.path.join(img_dir, params['phase'], 'vid1', img_name)
        img = io.imread(img_path)
        frames[frame_idx] = {'img_name':img_name, 'img':img}

    return frames


def get_detection_by_frame(num_frm = 'all'):
    params = util.params
    det_dir = os.path.join(params['dir_location']['detection_base_path'], params['phase'], 'vid1', 'det',
                           'detection_val500.txt')
    seq_dets = np.loadtxt(det_dir, usecols=range(10))
    detection_by_frmidx = defaultdict(list)  # {frmidx: {"img_name": xyz.jpg, "img":numpy_image}}
    if num_frm == 'all': num_frm = 9999999999
    for det in seq_dets:
        det[4:6] += det[2:4]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        conf = min(det[6], .99)
        det = list(map(int, det))
        bbox = det[2:6]
        frame_idx = det[0]
        if frame_idx>=num_frm: continue
        detection_by_frmidx[frame_idx].append( [*bbox, conf])

    return detection_by_frmidx


def get_detection_with_id_by_frame(num_frm = 'all'):
    det_dir = 'src/utils/C009.txt'
    seq_dets = np.loadtxt(det_dir, usecols=range(10), delimiter=',')
    detection_by_frmidx = {}  # {frmidx: {"img_name": xyz.jpg, "img":numpy_image}}
    if num_frm == 'all': num_frm = 9999999999
    for det in seq_dets:
        det[4:6] += det[2:4]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        conf = min(det[6], .99)
        det = list(map(int, det))
        bbox = det[2:6]
        frame_idx = det[0]
        obj_id = det[1]
        if frame_idx>=num_frm: continue
        if not frame_idx in detection_by_frmidx:
            detection_by_frmidx[frame_idx] = defaultdict(list)

        detection_by_frmidx[frame_idx][obj_id].append([*bbox, conf])

    return detection_by_frmidx


def save_list(l, f_dir, f_name):
    if not os.path.isdir(f_dir): os.mkdir(f_dir)
    file_path = os.path.join(f_dir, f_name)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerows(l)