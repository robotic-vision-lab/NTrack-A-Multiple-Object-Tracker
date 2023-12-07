import argparse
import tqdm
import os
import numpy as np
import motmetrics as mm
import multiprocessing as mp

from byte_tracker import BYTETracker
from utils import metric_calculator as metric
from utils import cotton_dataset as dataset
from utils import draw_util
from utils import  io_util
import cv2 as cv

os.environ['KMP_DUPLICATE_LIB_OK']='True'
fourcc = cv.VideoWriter_fourcc(*'MJPG')
video_writer = None
video_dim =(1080, 720)


def make_parser():
    parser = argparse.ArgumentParser("Test NTrack")
    parser.add_argument("--data_base_dir", help="Base directory to the data")
    parser.add_argument("--data_split", default="test", help="Base directory to the data")
    parser.add_argument("--use_pf", type =bool, help="Use particle filter")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold") ###
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")####30
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_false", help="test mot20.")
    return parser

def get_track_bb(atracks):
    wh = [track.tlwh[2:] for track in atracks]
    center = [track.particle_filter.get_state_center() for track in atracks]
    bb_lost_track = [np.concatenate((center[i] - wh[i] / 2, center[i] + wh[i] / 2)) for i in
              range(len(center))]
    id_lost_track = [t.track_id for t in atracks]
    lost_track = np.insert(bb_lost_track, 4, id_lost_track, axis=1) if len(id_lost_track) > 0 else np.empty((0, 5))
    return lost_track

def do_track(args, seq, save_vid=False, use_pf=True):
    '''
    Track cotton frame by frame in given sequence
    :param args:
    :param seq: A dict with frame as key and frame data as values
    :param save_vid:
    :param use_pf: Whether to use particle filter for motion model
    :return: Tracking results
    '''

    byteTracker = BYTETracker(args, use_pf=use_pf)
    metric_acc = mm.MOTAccumulator(auto_id=True)
    res =[]
    # Go over all frame one by one
    for frame_idx in range(len(seq)):  # seg_length

        """Get all data related to this frame """
        frame_data = seq[frame_idx]
        dets = frame_data['dets']
        gt = frame_data['gt']
        im = frame_data['img']

        if frame_idx == 0:
            byteTracker.init_opticalflow_manager(im)
        dets[:, 2:4] += dets[:, 0:2] # convert from xywh to xyxy
        gt_bb_id = list(gt.keys())
        gt_bb = np.asarray([gt[k] for k in gt_bb_id])

        online_targets, lost_track = byteTracker.update(dets, [1, 1], [1, 1],im)
        '''Handle output from tracker'''
        lost_track = get_track_bb(lost_track)
        matched_trks = get_track_bb(online_targets)
        bb = []
        bbtlbr = []
        bb_id =[]
        for t in online_targets:
            bb.append(t.tlwh.tolist())
            bbtlbr.append(t.tlbr.tolist())
            bb_id.append(t.track_id)

        matched_trks[:, 2:4] -= matched_trks[:, 0:2]
        if use_pf:
            metric.update_metric(metric_acc, gt_bb_id, gt_bb, bb_id, matched_trks[:,:4], 200)
            frm_rst = [[frame_idx + 1, m[4], *m[:4], 1, -1, -1, -1] for m in matched_trks]
        else:
            metric.update_metric(metric_acc, gt_bb_id, gt_bb, bb_id, bb, 200)
            frm_rst = [[frame_idx + 1,id, *m[:4], 1, -1, -1, -1] for m,id in zip(bb, bb_id)]
        res.extend(frm_rst)

        if save_vid:
            mask = np.zeros_like(im)  # mask is to refresh the detection and other disp info
            mask = cv.putText(mask, 'Frame: {}'.format(frame_idx),
                              (200, 200), cv.FONT_HERSHEY_SIMPLEX, fontScale=4,
                              color=(0, 255, 255), thickness=5)
            draw_util.show_output(im, None, np.insert(bbtlbr, 4,bb_id, axis=1),
                                  unmatched_trk=lost_track
                                  ,mask= mask,flow_layer= None,
                                  video_writer= video_writer)
            cv.waitKey(1)
    metric.populate_result(seq.seq_name, metric_acc)

    return [seq.seq_name, metric_acc, res]



if __name__ == '__main__':
    args = make_parser().parse_args()
    run_parallel = False # Set to True to track all sequences in parallel
    seq_names = []
    seq_results = []

    pool = mp.Pool(mp.cpu_count())

    track_seq = dataset.CottonDataset(args.data_base_dir, args.data_split)
    if run_parallel:
        results = pool.starmap(do_track, [(args, seq, not run_parallel, True) for seq in track_seq])
        seq_names = [r[0] for r in results]
        seq_results = [r[1] for r in results]
        for res in results:
            io_util.save_list(res[2], os.path.join('..','output', 'ntrack'), f'{res[0]}.txt')
    else:
        for seq in track_seq:
            video_writer = cv.VideoWriter(f'{seq.seq_name}.avi', fourcc, 10, video_dim)
            res = do_track(args, seq, save_vid= not run_parallel, use_pf=args.use_pf)
            video_writer.release()
        seq_names.append(res[0])
        seq_results.append(res[1])

    metric.populate_combined_results(seq_names, seq_results)
    pool.close()
