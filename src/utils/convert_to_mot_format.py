import glob
import os
from pycocotools.coco import COCO
from collections import defaultdict
#basedir = "D:\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\MOT_format_for_transtrack\\train"
basedir = r'C:\Users\Ahmed\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\MOT_format_for_transtrack\train'
basedir = r'C:\Users\Ahmed\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\MOT_format_for_transtrack\New_data\train'
split_list = {'vid09':[ (919,1218), (1425,1725), (1825,2125)],
               'vid25':[(986,1286), (1450,1751), (1964,2264), (2555,2855), (3327,3622)],
              'vid26': [(133,433),  (853,1153), (1801,2102), (2558,2857), (3280,3580)],
              'vid23': [(1137,1578), (2190,2490), (2726,3300), (3564,4164), (6840,7440 )],
              'vid14':[(825,1427),  (1560,1908)]
              }



#TODO:  add img1 directory for writing images
def distribute_into_dir(rename_img_frm_idx1 ):
    for vid_name in list(split_list.keys()):
        vid_dir = os.path.join(basedir, vid_name)
        os.chdir(basedir)
        for i in range(1,len(split_list[vid_name])+1):
            split_name = "{}_{:02d}".format(vid_name, i)
            if not os.path.isdir(split_name): os.mkdir(split_name)

        for file in glob.glob("{}/*.jpg".format(vid_name)):
            file = os.path.split(file)[-1]
            id = int(file[-9:-4])
            for split in range(len(split_list[vid_name])):
                if split_list[vid_name][split][0] <= id <= split_list[vid_name][split][1]:
                    split_dir = "{}_{:02d}".format(vid_name, split+1)
                    os.rename(os.path.join(vid_dir,file), os.path.join(basedir, split_dir, file))
                    break;
        if rename_img_frm_idx1:
            for i in range(1, len(split_list[vid_name]) + 1):
                split_dir = f"{vid_name}_{i:02d}"
                img_files = sorted(os.listdir(split_dir))
                for fidx, f in img_files:
                    os.rename(os.path.join(basedir, split_dir, f), os.path.join(basedir, split_dir, f'{fidx:04d}.jpg'))


import csv
def save_list(l, f_dir, f_name):
    if not os.path.isdir(f_dir): os.mkdir(f_dir)
    file_path = os.path.join(f_dir, f_name)
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerows(l)

def generate_gt():
    annotaion_dir = os.path.join(os.path.split(basedir)[0], 'annotations')
    det ={}
    for vid_name in list(split_list.keys()):
        vid_ann_file = os.path.join(annotaion_dir, 'instances_{}.json'.format(vid_name))
        for i in range(1,len(split_list[vid_name])+1):
            split_name = "{}_{:02d}".format(vid_name, i)
            det[split_name] = []

        coco = COCO(vid_ann_file)
        annIds = coco.getAnnIds()
        anns = coco.loadAnns(annIds)

        for ann in anns:
            # print('{},{},{},{}'int(*ann['bbox']))
            #det.append([ann['image_id'], ann['category_id'], *ann['bbox'], 1, -1, -1, -1])
            img_id = ann['image_id']
            img = coco.loadImgs(img_id)[0]
            img_file = img["file_name"].split("/")[-1]  # take only filename from root path
            img_idx = int(img_file[-9:-4]) # frame number in the video

            for split in range(len(split_list[vid_name])):
                if split_list[vid_name][split][0] <= img_idx <= split_list[vid_name][split][1]:
                    split_name = "{}_{:02d}".format(vid_name, split+1)
                    frm_idx = img_idx - split_list[vid_name][split][0] + 1 # frm index sataring from 1, in the sequence not in the whole video
                    det[split_name].append([frm_idx, ann['category_id'], *ann['bbox'], 1, -1, -1, -1])
                    break;

        '''Write to file'''
        for i in range(1,len(split_list[vid_name])+1):
            split_name = "{}_{:02d}".format(vid_name, i)
            save_dir = os.path.join(basedir, split_name, 'gt')
            save_list(det[split_name], save_dir, 'gt.txt')

            # print(det[-1])
            # print(ann['image_id'], -1, ann['bbox'], 1)
    return

def rename_file_starting_idx1(basedir):
    os.chdir(basedir)
    for vid_name in list(split_list.keys()):
        for i in range(1, len(split_list[vid_name]) + 1):
            split = f"{vid_name}_{i:02d}"
            split_img_dir = os.path.join(split, 'img1')
            img_files = sorted(os.listdir(split_img_dir))
            for fidx, f in enumerate(img_files ,1):
                os.rename(os.path.join(split_img_dir, f),
                          os.path.join( split_img_dir, f'{fidx:04d}.jpg'))




def convert_coco_to_mot():
    annotaion_dir = os.path.join(os.path.split(basedir)[0], 'annotations')
    os.chdir(basedir)
    for vid_name in list(split_list.keys()):
        vid_dir = os.path.join(basedir, vid_name)
        vid_ann_file = os.path.join(annotaion_dir,'instances_{}.json'.format(vid_name))
        if not os.path.exists(vid_ann_file):
            print(f"No annotation file for {vid_name}")
            continue
        det = {}
        for i in range(1, len(split_list[vid_name]) + 1):
            split_name = "{}_{:02d}".format(vid_name, i)
            det[split_name] = defaultdict(list)

        coco = COCO(vid_ann_file)
        annIds = coco.getAnnIds()
        anns = coco.loadAnns(annIds)

        for ann in anns:
            # print('{},{},{},{}'int(*ann['bbox']))
            # det.append([ann['image_id'], ann['category_id'], *ann['bbox'], 1, -1, -1, -1])
            img_id = ann['image_id']
            img = coco.loadImgs(img_id)[0]
            img_file = img["file_name"].split("/")[-1]  # take only filename from root path
            img_idx = int(img_file[-9:-4])  # frame number in the video

            for split in range(len(split_list[vid_name])):
                if split_list[vid_name][split][0]-1 <= img_idx <=  split_list[vid_name][split][1]: # -1 because deeplabel behave weirdly, it decrease image index while export
                    split_name = "{}_{:02d}".format(vid_name, split + 1)
                    det[split_name][img_idx].append([ann['category_id'], *ann['bbox'], 1, -1, -1,-1])
                    break;

        for split_name in det.keys():
            seq = det[split_name]
            seq_gt = []
            img_sorted = sorted(seq.keys())
            if not os.path.isdir(split_name):
                os.mkdir(split_name)
                os.mkdir(os.path.join(split_name,'img1'))
            for frm_idx, img in enumerate(img_sorted, 1):
                # Handle image
                source_img_file = f'C00{vid_name[-2:]}___{img:06d}.jpg'
                target_img_file =  f'{frm_idx:04d}.jpg'
                os.rename(os.path.join(vid_dir, source_img_file), os.path.join(basedir, split_name, 'img1', target_img_file))

                #Handle annotation
                img_gts = seq[img]
                for gt in img_gts:
                    seq_gt.append([frm_idx, *gt])

            # save gt to file
            save_dir = os.path.join(basedir, split_name, 'gt')
            save_list(seq_gt, save_dir, 'gt.txt')

        # '''Write to file'''
        # for i in range(1, len(split_list[vid_name]) + 1):
        #     split_name = "{}_{:02d}".format(vid_name, i)
        #     save_dir = os.path.join(basedir, split_name, 'gt')
        #     save_list(det[split_name], save_dir, 'gt.txt')

            # print(det[-1])
            # print(ann['image_id'], -1, ann['bbox'], 1)
    return

#distribute_into_dir()
#generate_gt()
#rename_file_starting_idx1(basedir)
convert_coco_to_mot()