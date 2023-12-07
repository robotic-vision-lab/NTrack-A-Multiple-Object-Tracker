import numpy as np
from numpy.random import uniform
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from filterpy.monte_carlo import stratified_resample
from utils import util
import cv2 as cv
from kalman_box_tracker import KalmanBoxTracker
from scipy.stats import multivariate_normal


def convert_wh2scaleaspect(w, h):
    return w*h, w/h


class ParticleFilterTracker(object):
    count = 0
    N_particle = 1000 #2000
    def __init__(self, center, width, height, state_space_w=3840, stae_space_h=2160,  bbox = None, measurement_std=50, meu_v = 0): # measure_std_error=25,5 ,50
        N = ParticleFilterTracker.N_particle
        self.ss_w = state_space_w
        self.ss_h = stae_space_h
        self.kf_tracker = None
        self.scale, self.aspect = convert_wh2scaleaspect(width, height)
        self.state_center = np.asarray(center)
        self.R = measurement_std

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1./N)
        self.particles = self.create_gaussian_particles(center, (200,100), N) #200,100
        ParticleFilterTracker.count += 1
        self.id = ParticleFilterTracker.count

        self.age = 0
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0  # last consecutive count of hit
        self.track_history = {}  # frm_idx => (state_bbox, matched)
        self.neighbors = {}  # [[track_id, x,y]] x,y => relative vector

        # for naive constant motion model
        self.v = 20

    @staticmethod
    def initialize_count():
        ParticleFilterTracker.count = 0

    def create_gaussian_particles(self, mean, std, N):
        particles = np.empty((N, 2))
        particles[:, 0] = mean[0] + (randn(N) * std[0])
        particles[:, 1] = mean[1] + (randn(N) * std[1])
        return particles

    def update_dormant(self, neighbor_constrained_center=None, var=None):
        if self.kf_tracker is not None:
            self.kf_tracker.update_with_optflow(self.get_state_bbox())

        center_prev = self.get_state_center()
        if neighbor_constrained_center is not None:
            dist = np.linalg.norm(self.particles[:, 0:2] - neighbor_constrained_center, axis=1)
            self.weights *= scipy.stats.norm(dist, self.R).pdf(0)
        else:
            return

        self.normalize_weight()
        self.state_center = self.get_state_center()

    def predict(self, optflow):

        if self.kf_tracker is not None:
            self.kf_tracker.predict()
            self.scale, self.aspect = np.squeeze(self.kf_tracker.get_state_z())[2:4]

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        if not np.isnan(optflow).any():
            std = [100 , 30]    #[50, 30] #+ ((width-250)/50)*5 mean = 250, std = 50
            self.particles[:, 0] += optflow[0] + randn(self.N_particle) * std[0]
            self.particles[:, 1] += optflow[1] + randn(self.N_particle) * std[1]

        self.state_center = self.get_state_center()
        return self.get_state_bbox()

    def predict_naive(self):  # predict based on constant velocity model
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1

        if self.kf_tracker is not None:
            v = np.squeeze(self.kf_tracker.get_state_z())[4:6]
            self.particles[:, 0] += v[0]
            self.particles[:, 1] += v[1]
            if np.isnan(self.particles[:, 0]).any():
                print("Nan found!!!")
        self.state_center = self.get_state_center()
        return self.get_state_bbox()

    def update(self, z):
        w_margin = 100

        bbox = z[:4] #, z[4]
        x, y, w, h = util.get_center_wh(bbox)  # z[4] is the detection confidence
        if bbox[0] < w_margin or bbox[2] > (self.ss_h - w_margin) :
            s, a = convert_wh2scaleaspect(w, h)
            self.scale = .2* self.scale + .8* s
            self.aspect =  .2* self.aspect + .8*a
        else:
            if self.kf_tracker is None:
                self.kf_tracker = KalmanBoxTracker(bbox, np.reshape([0, 0, 0], (3,1)), None)
            self.kf_tracker.update(bbox)
            self.scale, self.aspect = np.squeeze(self.kf_tracker.get_state_z())[2:4]#self.convert_wh2scaleaspect(w,h)

        det = (x, y)
        #self.weights.fill(1.)
        dist = np.linalg.norm(self.particles[:, 0:2] - det, axis=1)
        self.weights *= scipy.stats.norm(0, self.R).pdf(dist)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        # for i, landmark in enumerate(self.landmarks):
        #     dist = np.linalg.norm(self.particles[:, 0:2] - landmark, axis=1)
        #     self.weights *= scipy.stats.norm(dist, self.R).pdf(z[i])
        # self.weights /= sum(self.weights)  # normalize
        # resample if too few effective particles
        self.normalize_weight()
        self.state_center = self.get_state_center()

    def normalize_weight(self):
        self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize
        if self.neff(self.weights) < self.N_particle / 2:
            self.resample()

    def estimate(self):
        """returns mean and variance of the weighted particles"""

        pos = self.particles[:, 0:2]
        mean = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mean) ** 2, weights=self.weights, axis=0)
        return mean, var

    def resample(self):

        idx = stratified_resample(self.weights)
        self.resample_from_index(idx)

    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]

        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def add_track_history(self, matched, frame_id):
        self.track_history[frame_id] = (self.get_state_bbox(), matched)

    def get_state_center(self):
        pos = self.particles[:, 0:2]
        state = np.average(pos, weights=self.weights, axis=0)
        return state

    def get_state_bbox(self):
        x,y = self.state_center
        w = np.sqrt(self.scale * self.aspect)
        h = w / self.aspect
        if np.isnan(self.state_center.any()):
            print("Nan found!!!")

        if  self.aspect ==0 :
            print("width or height is zerooooo!!!!!!!!!!")
        return np.asarray([x - w / 2., y - h / 2., x + w / 2., y + h / 2.], dtype=int)

    def draw_particles(self,layer, color=(1,255,1)):
        cv.circle(layer, self.state_center.astype(int), 10 , color , -1)
        for p in self.particles:
            cv.circle(layer, p.astype(int), 1, color, -1 )




