from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


dataDir='./COCOdataset2017'
dataType='val'
annFile='/Users/mdahmedalmuzaddid/Data/cotton/train.json'
annFile = 'D:\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\Dataset_deeplabel\deeplabel_work\out_c0009\train.json'
annFile = '/Users/mdahmedalmuzaddid/OneDrive - University of Texas at Arlington/PhD/RVL/cotton/data/Dataset_deeplabel/deeplabel_work/out_c0009/train.json'
def test_library(annFile):
    # Initialize the COCO api for instance annotations
    coco=COCO(annFile)

    # Load the categories in a variable
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    print(cats)

    imgIds = coco.getImgIds()
    print("Number of images containing all the  classes:", len(imgIds))

    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
    print(img)

    annIds = coco.getAnnIds()
    anns = coco.loadAnns(annIds)
    #print(anns)

def save_as_csv(det, out_filename):
    np.savetxt(out_filename,
               det,
               delimiter =", ",
               fmt ='%s')

def coco2det(annFile):
    coco=COCO(annFile)
    annIds = coco.getAnnIds()
    anns = coco.loadAnns(annIds)
    det = []
    for ann in anns:
        #print('{},{},{},{}'int(*ann['bbox']))
        det.append([ann['image_id'], ann['category_id'], *ann['bbox'], 1, -1, -1, -1])
        #print(det[-1])
        #print(ann['image_id'], -1, ann['bbox'], 1)
    return det



def frameid_to_imageid(annFile):
    '''
    return image file dictionary
    key = img id from coco file
    value = img file name
    '''

    # Initialize the COCO api for instance annotations
    coco=COCO(annFile)
    imgIds = coco.getImgIds()
    print("Number of images containing all the  classes:", len(imgIds))
    frame2img = {}
    for i in imgIds:
        img = coco.loadImgs(imgIds[i])[0]
        frame2img[i] = img["file_name"].split("/")[-1] # take only filename from root path
    return frame2img

def show_img():
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ) #aspect='equal'

    fn = os.path.join('mot_cotton', "train", "vid1" , "cotton_000002.jpg")
    im =io.imread(fn)
    ax1.imshow(im)
    fig.canvas.flush_events()
    plt.draw()
    ax1.cla()

    #plt.title(seq + ' Tracked Targets')


    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg
    # img = mpimg.imread(os.path.join('mot_cotton', "train", "vid1" , "cotton_000002.jpg"))
    # imgplot = plt.imshow(img)
    #plt.show()



def preapare_and_save_for_tracking(vid_id):

    dettection = coco2det(annFile)
    save_as_csv(dettection, vid_id+'.csv')
    frm2img = frameid_to_imageid(annFile)
    np.save(vid_id+".npy", frm2img)


preapare_and_save_for_tracking(vid_id = "C009")
#show_img()
#