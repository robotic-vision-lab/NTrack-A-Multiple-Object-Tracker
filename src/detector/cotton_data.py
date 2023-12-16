import configparser
import csv
import os
import os.path as osp
import pickle

import numpy as np
import pycocotools.mask as rletools
import scipy
import torch
from pycocotools.coco import COCO
from PIL import Image
import random
import  tqdm
sample_image_per_videos = 200

class MOTCottonDetect(torch.utils.data.Dataset):
    """ Data class for the Multiple Object Tracking Dataset
    """

    # def __init__(self, root, transforms=None, vis_threshold=0.25,
    #              split_seqs=None, frame_range_start=0.0, frame_range_end=1.0):
    #     self.root = root
    #     self.transforms = transforms
    #     self._vis_threshold = vis_threshold
    #     self._classes = ('background', 'pedestrian')
    #     self._img_paths = []
    #     self._split_seqs = split_seqs
    #     self.img2frmidx = {} #Map the image file name to frm index
    #
    #     self.mots_gts = {}
    #     for f in sorted(os.listdir(root)):
    #         path = os.path.join(root, f)
    #         dat = self.load(root, f)
    #         if not os.path.isdir(path):
    #             continue
    #
    #         # if split_seqs is not None and f not in split_seqs:
    #         #     continue
    #
    #         im_ext = '.jpg'
    #         im_dir = 'img1'
    #
    #         img_dir = os.path.join(path, im_dir)
    #         images = sorted(os.listdir(img_dir))
    #         seq_len = len(images)#int(config['Sequence']['seqLength'])
    #
    #         start_frame = int(frame_range_start * seq_len)
    #         end_frame = int(frame_range_end * seq_len)
    #
    #         # for i in range(seq_len):
    #         #for i in range(start_frame, end_frame):
    #         for frm_idx, img_file in enumerate(images):
    #             img_path = os.path.join(img_dir, img_file)#os.path.join(img_dir, f"{i + 1:06d}{im_ext}")
    #             assert os.path.exists(img_path), f'Path does not exist: {img_path}'
    #             self._img_paths.append(img_path)
    #             self.img2frmidx[img_file] = frm_idx+1 #index start from 1
    #         # print(len(self._img_paths))

    def __init__(self, root, transforms=None, vis_threshold=0.25,
                 split_seqs=None, frame_range_start=0.0, frame_range_end=1.0):
        root = os.path.dirname(root) #removing train part
        self.root = root
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ('background', 'pedestrian')
        self._img_paths = []
        self._split_seqs = split_seqs
        self._annos = []  # Map the image file name to frm index
        self.img2frmidx = {}
        self.mots_gts = {}
        idx = 1
        for f in sorted(os.listdir(root)):
            if f not in split_seqs:
                continue
            path = os.path.join(root, f)
            if not os.path.isdir(path):
                continue
            dat = self.load(root, f)

            for frame in dat:
                self._img_paths.append(frame['file_name'])
                boxes =  frame['boxes']
                num_objs = len(frame['class'])
                annos = {'boxes': boxes,
                         'labels': torch.ones((num_objs,), dtype=torch.int64),
                         'image_id': torch.tensor([idx]),
                         'area': (boxes[:, 3] - boxes[:, 1]) * (
                                     boxes[:, 2] - boxes[:, 0]),
                         'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                         'visibilities':torch.ones((num_objs), dtype=torch.float32),
                         'track_ids': torch.ones((num_objs), dtype=torch.long), }

                self._annos.append(annos)
                idx +=1

    def __str__(self):
        if self._split_seqs is None:
            return self.root
        return f"{self.root}/{self._split_seqs}"

    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        """
        """
        if 'test' in self.root:
            num_objs = 0
            boxes = torch.zeros((num_objs, 4), dtype=torch.float32)

            return {'boxes': boxes,
                'labels': torch.ones((num_objs,), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                'visibilities': torch.zeros((num_objs), dtype=torch.float32)}

        img_path = self._img_paths[idx]
        #file_index = int(os.path.basename(img_path).split('.')[0])
        file_index = self.img2frmidx[os.path.split(img_path)[1]]
        gt_file = os.path.join(os.path.dirname(
            os.path.dirname(img_path)), 'gt', 'gt.txt')

        assert os.path.exists(gt_file), \
            'GT file does not exist: {}'.format(gt_file)

        bounding_boxes = []

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                visibility = 1#float(row[8])

                if int(row[0]) == file_index :
                    bb = {}
                    bb['bb_left'] = int(row[2])
                    bb['bb_top'] = int(row[3])
                    bb['bb_width'] = int(row[4])
                    bb['bb_height'] = int(row[5])
                    bb['visibility'] = 1.#float(row[8])
                    bb['track_id'] = int(row[1])

                    bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        visibilities = torch.zeros((num_objs), dtype=torch.float32)
        track_ids = torch.zeros((num_objs), dtype=torch.long)

        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left']# - 1
            y1 = bb['bb_top']# - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width']# - 1
            y2 = y1 + bb['bb_height']# - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb['visibility']
            track_ids[i] = bb['track_id']

        annos = {'boxes': boxes,
                 'labels': torch.ones((num_objs,), dtype=torch.int64),
                 'image_id': torch.tensor([idx]),
                 'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                 'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                 'visibilities': visibilities,
                 'track_ids': track_ids,}


        return annos

    @property
    def has_masks(self):
        return '/MOTS20/' in self.root

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        if not os.path.isfile(img_path): print(img_path)
        img = Image.open(img_path).convert("RGB")

        target = self._annos[idx]#self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)

        ############################################################################

    def load(self,basedir, split, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        basedir = os.path.expanduser(basedir)
        self._imgdir = os.path.realpath(os.path.join(basedir, split))
        assert os.path.isdir(self._imgdir), "{} is not a directory!".format(
            self._imgdir)
        annotation_file = os.path.join(
            basedir, 'annotations','instances_{}.json'.format(split))
        assert os.path.isfile(annotation_file), annotation_file

        coco = COCO(annotation_file)
        annotation_file = annotation_file


        img_ids = coco.getImgIds()
        if not annotation_file.endswith(
                ('vid10.json', 'vid24.json', 'val.json')):
            img_ids = random.sample(img_ids, sample_image_per_videos)
        img_ids.sort()
        # list of dict, each has keys: height,width,id,file_name
        imgs = coco.loadImgs(img_ids)

        for idx, img in enumerate(tqdm.tqdm(imgs)):
            img['image_id'] = img.pop('id')
            img['file_name'] = os.path.join(self._imgdir,
                                            img["file_name"].split("/")[-1])
            if idx == 0:
                # make sure the directories are correctly set
                assert os.path.isfile(img["file_name"]), img["file_name"]
            if add_gt:
                self._add_detection_gt(img, coco, annotation_file)
        return imgs

    def _add_detection_gt(self, img, coco, annotation_file):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['image_id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = coco.imgToAnns[
            img['image_id']]  # equivalent but faster than the above two lines
        if 'minival' not in annotation_file:
            # TODO better to check across the entire json, rather than per-image
            ann_ids = [ann["id"] for ann in objs]
            assert len(set(ann_ids)) == len(ann_ids), \
                "Annotation ids in '{}' are not unique!".format(annotation_file)

        # clean-up boxes
        width = img.pop('width')
        height = img.pop('height')


        all_cls = []
        all_iscrowd = []
        num_objs = len(objs)
        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        for objid, obj in enumerate(objs):
            # if obj.get('ignore', 0) == 1:
            #     continue
            x1, y1, w, h = list(map(float, obj['bbox']))
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel
            x2, y2 = x1 + w, y1 + h

            # np.clip would be quite slow here
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)

            boxes[objid, 0] = x1
            boxes[objid, 1] = y1
            boxes[objid, 2] = x2
            boxes[objid, 3] = y2
            w, h = x2 - x1, y2 - y1
            # Require non-zero seg area and more than 1x1 box size
            # if obj['area'] > 1 and w > 0 and h > 0:
            #
            all_cls.append(1)
            #     iscrowd = obj.get("iscrowd", 0)
            #     all_iscrowd.append(iscrowd)


        # all geometrically-valid boxes are returned

        img['boxes'] = boxes#np.asarray(all_boxes, dtype='float32')  # (n, 4)

        img['class'] = np.ones_like(all_cls)  # n, always >0
        img['is_crowd'] = np.asarray(all_iscrowd, dtype='int8')  # n,

    def write_results_files(self, results, output_dir):
        """Write the detections in the format for MOT17Det sumbission

        all_boxes[image] = N x 5 array of detections in (x1, y1, x2, y2, score)

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT17-01.txt
        ./MOT17-02.txt
        ./MOT17-03.txt
        ./MOT17-04.txt
        ./MOT17-05.txt
        ./MOT17-06.txt
        ./MOT17-07.txt
        ./MOT17-08.txt
        ./MOT17-09.txt
        ./MOT17-10.txt
        ./MOT17-11.txt
        ./MOT17-12.txt
        ./MOT17-13.txt
        ./MOT17-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        files = {}
        for image_id, res in results.items():
            path = self._img_paths[image_id]
            img1, name = osp.split(path)
            # get image number out of name
            frame = int(name.split('.')[0])
            # smth like /train/MOT17-09-FRCNN or /train/MOT17-09
            tmp = osp.dirname(img1)
            # get the folder name of the sequence and split it
            tmp = osp.basename(tmp).split('-')
            # Now get the output name of the file
            out = tmp[0]+'-'+tmp[1]+'.txt'
            outfile = osp.join(output_dir, out)

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            if 'masks' in res:
                delimiter = ' '
                # print(torch.unique(res['masks'][0]))
                masks = res['masks'].squeeze(dim=1)# > 0.5 #res['masks'].bool()

                index_map = torch.arange(masks.size(0))[:, None, None]
                index_map = index_map.expand_as(masks)

                masks = torch.logical_and(
                    # remove background
                    masks > 0.5,
                    # remove overlapp by largest probablity
                    index_map == masks.argmax(dim=0)
                )
                for res_i in range(len(masks)):
                    track_id = -1
                    if 'track_ids' in res:
                        track_id = res['track_ids'][res_i].item()
                    mask = masks[res_i]
                    mask = np.asfortranarray(mask)

                    rle_mask = rletools.encode(mask)

                    files[outfile].append(
                        [frame,
                         track_id,
                         2,  # class pedestrian
                         mask.shape[0],
                         mask.shape[1],
                         rle_mask['counts'].decode(encoding='UTF-8')])
            else:
                delimiter = ','
                for res_i in range(len(res['boxes'])):
                    track_id = -1
                    if 'track_ids' in res:
                        track_id = res['track_ids'][res_i].item()
                    box = res['boxes'][res_i]
                    score = res['scores'][res_i]

                    x1 = box[0].item()
                    y1 = box[1].item()
                    x2 = box[2].item()
                    y2 = box[3].item()

                    out = [frame, track_id, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1]

                    if 'keypoints' in res:
                        out.extend(res['keypoints'][res_i][:, :2].flatten().tolist())
                        out.extend(res['keypoints_scores'][res_i].flatten().tolist())

                    files[outfile].append(out)

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=delimiter)
                for d in v:
                    writer.writerow(d)

class SegmentedObject:
    """
    Helper class for segmentation objects.
    """
    def __init__(self, mask: dict, class_id: int, track_id: int, full_bbox=None) -> None:
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id
        self.full_bbox = full_bbox

def load_mots_gt(path: str) -> dict:
    """Load MOTS ground truth from path."""
    objects_per_frame = {}
    track_ids_per_frame = {}  # Check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # Check that no frame contains overlapping masks

    with open(path, "r") as gt_file:
        for line in gt_file:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            # if frame not in track_ids_per_frame:
            #     track_ids_per_frame[frame] = set()
            # if int(fields[1]) in track_ids_per_frame[frame]:
            #     assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
            # else:
            #     track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {
                'size': [int(fields[3]), int(fields[4])],
                'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([
                    combined_mask_per_frame[frame], mask],
                    intersect=True)):
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask],
                    intersect=False)

            full_bbox = None
            if len(fields) == 10:
                full_bbox = [int(fields[6]), int(fields[7]), int(fields[8]), int(fields[9])]

            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1]),
                full_bbox
            ))

    return objects_per_frame
