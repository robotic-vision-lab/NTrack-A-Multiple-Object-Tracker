# This file removes bbox based on provided margin

import numpy as np
import os
from src.utils import io_util

tracker_output_base_dir = r'C:\Users\Ahmed\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\Evaluation\HOTA\data\trackers\cotton\cotton22-train'
gt_base_dir = r'C:\Users\Ahmed\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\Evaluation\HOTA\data\gt\cotton_challenge\cotton22-train'
w_margin=200

def remove_bbox_from_tracker_output(w_margin, h_margin=0):
    trackers = os.listdir(tracker_output_base_dir)
    for t in trackers:
        tracker_base_dir = os.path.join(tracker_output_base_dir, t, 'data')
        sequences = os.listdir(tracker_base_dir)
        for seq in sequences:
            bboxs = np.loadtxt(os.path.join(tracker_base_dir, seq), delimiter=',')
            ind = np.logical_and(bboxs[:,2] > w_margin , bboxs[:,2]+ bboxs[:,4] < 3840 - w_margin)
            bboxs = bboxs[ind]
            io_util.save_list(bboxs, os.path.join(tracker_output_base_dir, t, 'data1'), seq)

def remove_bbox_from_gt(w_margin, h_margin=0):
    sequences = os.listdir(gt_base_dir) #\vid09_02\gt

    for seq in sequences:
        bboxs = np.loadtxt(os.path.join(gt_base_dir, seq, 'gt', 'gt.txt'), delimiter=',')
        ind = np.logical_and(bboxs[:,2] > w_margin , bboxs[:,2]+ bboxs[:,4] < 3840 - w_margin)
        bboxs = bboxs[ind]
        print(np.mean(bboxs[:,4]), np.sqrt(np.var(bboxs[:,4])))
        #io_util.save_list(bboxs, os.path.join(gt_base_dir, seq,'gt'), 'gt.txt')

remove_bbox_from_tracker_output(w_margin)
remove_bbox_from_gt(w_margin)
