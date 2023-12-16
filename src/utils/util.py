import numpy as np
import yaml
import os
from sklearn.neighbors import NearestNeighbors

def read_config(file_path):
    # parser = configargparse.YAMLConfigFileParser()
    # with open(file_path, "r") as stream:
    #     args = parser.parse(stream)
    # MyTuple = namedtuple('MyTuple', args)
    # params =MyTuple(**args)
    # print(params.tracker)
    print(os.getcwd())
    with open(file_path, "r") as ymlfile:
        params = yaml.load(ymlfile, Loader=yaml.FullLoader)
    machine = params['machine']
    params['dir_location'] = params['dir_location'][machine]
    return params

params = None#read_config("config/basic_config.yaml")

def init_params(filepath):
    global params
    params = read_config(filepath)

def get_center_wh(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    return (x,y,w,h)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

def tracker_to_bbox_displacement(trackers, bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    tracker_mean = np.mean(trackers,axis=0)[0]
    return tracker_mean - (x,y)

def translate_bbox(bbox, delta_vec):
    """
        Takes a bounding box in the centre form [x1,y1,x2,y2] and returns its translated one
          [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
    x, y, w, h = get_center_wh(bbox)
    bbox[0:2] += delta_vec
    bbox[2:4] += delta_vec
    return bbox
    # center = (x,y) + delta_vec
    # return np.array([center[0]-w/2.,center[1]-h/2.,center[0]+w/2.,center[1]+h/2.])

def convert_track_features_to_bbox(features, diplacement, prev_bbox):
    tracker_mean = np.squeeze(np.mean(features, axis=0))
    bbox_center = tracker_mean- diplacement
    w = prev_bbox[2] - prev_bbox[0]
    h = prev_bbox[3] - prev_bbox[1]

    return np.array([bbox_center[0]-w/2.,bbox_center[1]-h/2.,
                     bbox_center[0]+w/2.,bbox_center[1]+h/2.]).reshape((1,4))
    # # w, h = 100, 100
    # return np.array([tracker_mean[0] - w / 2., tracker_mean[1] - h / 2.,
    #                  tracker_mean[0] + w / 2., tracker_mean[1] + h / 2.]).reshape((1, 4))

def frameid_to_imageid(frameseq2imgfile):
    frame2img = np.load(frameseq2imgfile, allow_pickle=True)
    return frame2img.item()

def nearest_neighbor(positions, query, n_neighbor=3):
    neigh = NearestNeighbors(n_neighbors=3)
    neigh.fit(positions)
    nbrs = neigh.kneighbors(query, n_neighbor, return_distance=False)
    return nbrs
