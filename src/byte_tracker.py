import numpy as np
from collections import deque
import os
import os.path as osp
import copy
# import torch
# import torch.nn.functional as F

from kalman_filter import KalmanFilter
import matching
from basetrack import BaseTrack, TrackState
from particle_filter import ParticleFilterTracker
from utils import opticalflow_util as opt_util
import cv2 as cv
from sklearn.neighbors import NearestNeighbors
import relative_position_analyzer as rp_analyzer

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.particle_filter = None

        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.neighbors = {}

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, flow=None):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
        if flow is not None:
            STrack.multi_predict_flow(stracks, flow)
    @staticmethod
    def multi_predict_flow(stracks, dense_flow ):
        for i, trk in enumerate(stracks):
            pf_tracker = trk.particle_filter
            prev_pos = pf_tracker.get_state_center()
            'TODO handle out of window'
            if prev_pos[0] < 0 or prev_pos[0] > 3840 or prev_pos[1] <0 or prev_pos[1]>2160: continue
            if prev_pos[0] < 0 or prev_pos[1] < 0: continue
            flow = opt_util.get_flow_summary(dense_flow, prev_pos)
            bbox = pf_tracker.predict(flow)


    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.particle_filter = ParticleFilterTracker(self.tlwh_to_xyah(self._tlwh)[:2], self._tlwh[2], self._tlwh[3])
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, use_pf=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if use_pf: self.particle_filter.update(self.tlwh_to_tlbr(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, use_pf=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        if use_pf: self.particle_filter.update(self.tlwh_to_tlbr(new_tlwh))
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30, use_pf=False):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.opticalflow_manager = None
        self.use_pf = use_pf
        BaseTrack.init_id()


    def init_opticalflow_manager(self, init_frame):
        self.opticalflow_manager = opt_util.Opticalflow_Manager(init_frame)


    def update(self, output_results, img_info, img_size, frame=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        if self.use_pf:
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            self.dense_flow = self.opticalflow_manager.get_dense_flow(frame_gray)

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        if self.use_pf:
            STrack.multi_predict(strack_pool, self.dense_flow)
        else:
            STrack.multi_predict(strack_pool, None)
        dists = matching.iou_distance(strack_pool, detections, self.use_pf)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id, self.use_pf)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, use_pf=self.use_pf)
                refind_stracks.append(track)
        update_neighbors(matches,strack_pool, self.frame_id, n_neighbor = 5)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second, self.use_pf)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id, self.use_pf)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, use_pf=self.use_pf)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections, self.use_pf)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id, self.use_pf)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, self.use_pf)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        if self.use_pf:
            self.update_lost()
            self.opticalflow_manager.update_old_gray(frame_gray)
        return output_stracks, self.lost_stracks

    def update_lost(self):
        #print('\n',self.frame_id-1)
        tracked_stracks_dict = {t.track_id:t for t in self.tracked_stracks}
        for l_track in self.lost_stracks:
            loc, var = rp_analyzer.get_consistent_loc_linear_in_pos_byte(tracked_stracks_dict,
                                                                    l_track.neighbors, 3)
         #   print(l_track.track_id, end='\t')
            l_track.particle_filter.update_dormant(loc, var)


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, use_pf):
    pdist = matching.iou_distance(stracksa, stracksb, use_pf)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def update_nn_for_matched_tracks( nn, candidate_neigh_idx, strack_pool, frm_idx):
    for i, trk_idx in enumerate(candidate_neigh_idx):
        for j in nn[i][1:]:  # Closest neighbor is itself, so discard that
            neighbor_idx = candidate_neigh_idx[j]
            own_location = strack_pool[trk_idx].particle_filter.state_center
            nn_location = strack_pool[neighbor_idx].particle_filter.state_center
            nn_id = strack_pool[neighbor_idx].track_id
            relative_vec = own_location - nn_location
            #nn_info = [frm_idx, relative_vec[0], relative_vec[1]]
            nn_info = [frm_idx, relative_vec[0], relative_vec[1], nn_location[0], nn_location[1], own_location[0], own_location[1],]
            if nn_id in strack_pool[trk_idx].neighbors:
                strack_pool[trk_idx].neighbors[nn_id].append(nn_info)
            else:
                strack_pool[trk_idx].neighbors[nn_id] = [nn_info]


def update_neighbors(matched, strack_pool,  frm_idx, n_neighbor):
    # update nearest neighbor of matched tracks
    if len(matched) < 2 :return
    candidate_neigh_idx = matched[:,0]
    neighbor_loc = np.asarray([strack_pool[idx].particle_filter.state_center for idx in candidate_neigh_idx])
    n_neighbor = min(n_neighbor, len(neighbor_loc) - 1)

    '''Nearest neighbor'''
    neigh = NearestNeighbors()
    neigh.fit(neighbor_loc)
    nn = neigh.kneighbors(neighbor_loc, n_neighbor + 1, return_distance=False)

    # '''Random Neighbor'''
    # nn = [np.random.choice(len(candidate_neigh_idx), n_neighbor+1) for _ in range(len(neighbor_loc))]
    # nn = np.asarray(nn)

    update_nn_for_matched_tracks(nn, candidate_neigh_idx, strack_pool, frm_idx)



