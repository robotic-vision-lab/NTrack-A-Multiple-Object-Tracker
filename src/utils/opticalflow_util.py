import cv2 as cv
import numpy as np
from utils import util
color = np.random.randint(0,255,(100,3))


def get_feature(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    roi = [1200, 1350, 1650, 1800]
    mask[roi[0]:roi[1], roi[2]:roi[3]] = 255
    corners = cv.goodFeaturesToTrack(gray, 10, 0.01, 10, mask=mask)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv.circle(img, (x, y), 10, 255, -1)
    cv.rectangle(img,(1650, 1200 ),(1800, 1350), (0, 255, 0), 3)  # (1780, 650),(2060,920)
    return img


class Opticalflow_Manager:
    def __init__(self, init_frame):
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=3,
                              qualityLevel=0.3,
                              minDistance=20,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=7,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it
        self.old_frame = init_frame
        self.old_gray = cv.cvtColor(self.old_frame, cv.COLOR_BGR2GRAY)

    def get_initial_feature2track(self,gray_img, roi):
        roi = roi.astype(int)
        roi_mask = np.zeros_like(gray_img)
        roi_mask[roi[1]:roi[3], roi[0]:roi[2] ] = 255
        p0 = cv.goodFeaturesToTrack(gray_img, mask=roi_mask, **self.feature_params)
        return p0

    def update_old_gray(self, old_gray):
        self.old_gray = old_gray

    def get_flow(self,frame_gray, feature2track, optflow_layer, draw_flow=True):
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, feature2track, None, **self.lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = feature2track[st == 1] #self.p0[st == 1]
        else:
            print("No good tracker found!!!")
            return np.empty((0,1,2)), optflow_layer
        # draw the tracks
        if draw_flow:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                optflow_layer = cv.line(optflow_layer, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 3)
        return good_new.reshape(-1, 1, 2), optflow_layer

    def get_dense_flow(self, frame_gray):
        flow = cv.calcOpticalFlowFarneback(self.old_gray, frame_gray,  None, 0.5, 6, 25, 3, 5, 1.2, 0)


def get_flow_summary(denseflow, roi, window = (1,1)):
    '''
    :param denseflow:
    :param roi: roi could be center (len(roi)==2) or could be bbox (len(roi)>2)
    :param window:
    :return:
    '''
    roi = np.array(roi).astype(int)
    if len(roi)>2:
        x, y, w, h = tuple(map(int, util.get_center_wh(roi)))
    else: x,y = roi
    img_h, img_w = np.shape(denseflow)[:2]

    flow_window = denseflow[max(0,y-window[0]):min(img_h, y+window[0]), max(0,x-window[1]):min(img_w, x+window[1]),:]#denseflow[max(0,roi[1]):roi[3], max(0,roi[0]):roi[2],:]
    return np.mean(flow_window, axis=(0,1))