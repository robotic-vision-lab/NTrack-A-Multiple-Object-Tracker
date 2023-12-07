from filterpy.kalman import KalmanFilter
from filterpy.common import Saver
import numpy as np
from utils import util #
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox, init_v, feature2track):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.feature2track = feature2track
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        #P : state cov
        #R : measurement cov
        #Q : Process cov
        self.kf.R[2:,2:] *= 10000000. #10. uncertainity about the detection bbox
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities #p[4:,4:]
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        #self.kf.Q[2:4,2:4] *= 1000.

        self.kf.x[:4] = util.convert_bbox_to_z(bbox)
        self.kf.x[4:] = init_v
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        #tracer to bbox displacement
        self.tracker_to_bbox_displacement = 0
        self.recent_detection = bbox
        self.track_history = {} # tuple of 4 frm_idx:( x, y , s, w, matched) #tuple of 5 (id, x, y , s, w, matched, frame)
        self.neighbors = {} # [[track_id, x,y]] x,y=>relative vector

    def update(self,bbox, newfeature2track=None):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(util.convert_bbox_to_z(bbox))
        self.feature2track = newfeature2track
        self.recent_detection = bbox

    def update_with_optflow(self,bbox, newfeature2track=None): # No associated dectaction found

        z = util.convert_bbox_to_z(bbox)
        # z[2:] = self.kf.x[2:4] # only update center, not size
        self.kf.update(z)
        #self.kf.x[0:2] =z[0:2]
        self.feature2track = newfeature2track
        # d = tracker_to_bbox_displacement(self.feature2track,
        #                                  bbox)
        # return


    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(util.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return util.convert_x_to_bbox(self.kf.x)

    def get_state_z(self):
        return self.kf.x

    def get_state_center(self):
        return np.squeeze(self.kf.x[:2])

    def add_track_history(self, matched, frame_id):
        self.track_history[frame_id] = (np.squeeze(self.get_state_z()[:4]), matched )

