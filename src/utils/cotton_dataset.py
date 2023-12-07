import configparser
import csv
import os
import os.path as osp
import numpy as np
#from PIL import Image
from skimage import io


class CottonDataset():
    """A Wrapper for the MOT_Sequence class to return multiple sequences."""

    def __init__(self, data_base_dir, split):

        test_sequences = [ 'vid09_01', 'vid09_02', 'vid09_03',
                        'vid25_01', 'vid25_02', 'vid25_03',
                        'vid26_01', 'vid26_02', 'vid26_03',
                        'vid23_01', 'vid23_02', 'vid23_03',
                        'vid14_01']
        #train_sequences = ['vid14_01']

        train_sequences = ['vid25_04', 'vid25_05', 'vid26_04', 'vid26_05',
                       'vid23_04', 'vid23_05', 'vid14_02']

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences

        self.sequence = []
        for s in sequences:
            self.sequence.append(CottonSequence(s, os.path.join(data_base_dir, split)))

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx]


class CottonSequence():
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name, data_base_dir):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self.seq_name = seq_name
        self.seq_dir = osp.join(data_base_dir, seq_name)
        self.data, self.no_gt = self._sequence()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = io.imread(data['im_path']) #Image.open(data['im_path']).convert("RGB")
        # img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] =  np.array([det[:5] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def _sequence(self):
        img_dir = osp.join(self.seq_dir, 'img1')
        gt_path = osp.join(self.seq_dir, 'gt', 'gt.txt')
        det_path = osp.join(self.seq_dir, 'det', 'det.txt')

        total = []
        boxes = {}
        dets = {}
        visibility = {}
        img_files = sorted(os.listdir(img_dir))
        seqLength = len(img_files)

        for i in range(1, seqLength + 1):
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []

        no_gt = False
        if osp.exists(gt_path):
            with open(gt_path, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2])
                    y1 = float(row[3])
                    w, h = float(row[4]), float(row[5])
                    bb = np.array([x1, y1, w, h], dtype=np.float32)
                    frm_id, obj_id = int(row[0]), int(row[1])
                    boxes[frm_id][obj_id] = bb
                    visibility[frm_id][obj_id] = float(row[8])
        else:
            no_gt = True

        if osp.exists(det_path):
            with open(det_path, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    '''Todo edit det generation process'''
                    #row = row[0].split()
                    if len(row)<3:
                        #print(row)
                        continue
                    x1 = float(row[2])
                    y1 = float(row[3])
                    w, h = float(row[4]), float(row[5])
                    score = float(row[6])
                    bb = np.array([x1, y1, w, h, score], dtype=np.float32)
                    frm_id = int(row[0])
                    dets[frm_id].append(bb)

        for i, img_file in enumerate(img_files,1):
            im_path = osp.join(img_dir, img_file)

            sample = {'gt': boxes[i],
                      'im_path': im_path,
                      'vis': visibility[i],
                      'dets': dets[i], }

            total.append(sample)

        return total, no_gt

    def __str__(self):
        return self.seq_name

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(osp.join(output_dir, f"{self._seq_name}.txt"), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1,
                         i + 1,
                         x1 + 1,
                         y1 + 1,
                         x2 - x1 + 1,
                         y2 - y1 + 1,
                         -1, -1, -1, -1])

    def load_results(self, output_dir):
        file_path = osp.join(output_dir, self._seq_name)
        results = {}

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as of:
            csv_reader = csv.reader(of, delimiter=',')
            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if not track_id in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = [x1, y1, x2, y2]

        return results
