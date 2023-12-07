import motmetrics as mm
import numpy as np
import os

def update_metric(metric_acc, gt_id, gt_box, match_id, match_bb):
    margin = 250
    ''''Refine gt and ht - remove boxes which are in margin'''
    l = len(gt_id)
    for i in range(l-1, -1, -1):
        if gt_box[i][0] < margin or gt_box[i][0]> 3840-margin:
            gt_id = np.delete(gt_id,i)
            gt_box = np.delete(gt_box,i,axis=0)
    l =len(match_id)-1
    for i in range(l, -1, -1):
        if match_bb[i][0] < margin or match_bb[i][0] > 3840 - margin:
            match_id = np.delete(match_id, i)
            match_bb = np.delete(match_bb, i, axis=0)

    dis_mat = mm.distances.iou_matrix(gt_box, match_bb, max_iou=0.65)
    metric_acc.update(gt_id, match_id, dis_mat)


def generate_results(seq_names, all_metric):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        all_metric,
        metrics=mm.metrics.motchallenge_metrics,
        names=seq_names,
        generate_overall=True
    )

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    with open(os.path.join('..','..', 'output', 'deepsort_result.txt'), 'w') as f:
        print(strsummary, file= f)


def rearrange(dat):

    n_frm = int(np.max(dat[:, 0]))
    bbox_by_frm = [ {'obj_id':[], 'bbox':[]} for _ in range(n_frm+1)]
    for d in dat:
        frm_id = int(d[0])
        obj_id = int(d[1])
        bbox = [d[2], d[3], d[4], d[5]]
        bbox_by_frm[frm_id]['obj_id'].append(obj_id)
        bbox_by_frm[frm_id]['bbox'].append(bbox)
    return bbox_by_frm



def generate():
    gt_basedir = r'C:\Users\Ahmed\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\MOT_format_for_transtrack\train'
    hypothesis_dir = r"C:\Users\Ahmed\OneDrive - University of Texas at Arlington\PhD\RVL\cotton\data\Evaluation\deep_sort\results"
    train_sequences = ['vid25_01', 'vid09_01', 'vid09_02', #
                       'vid09_03', 'vid25_02','vid25_03', 'vid25_04', 'vid25_05',
                       'vid26_01', 'vid26_03', 'vid26_04', 'vid26_05']

    sequences = [s for s in os.listdir(gt_basedir) if s in train_sequences]
    metrics = []
    seq_names = []
    for seq in sequences:
        mot_metric = mm.MOTAccumulator(auto_id=True)
        seq_gt = os.path.join(gt_basedir,seq, 'gt', 'gt.txt')
        seq_h = os.path.join(hypothesis_dir, f'{seq}.txt')
        gt = np.loadtxt(seq_gt, delimiter=',')
        h = np.loadtxt(seq_h, delimiter=',')
        gt_bbox= rearrange(gt)
        h_bbox = rearrange(h)
        for i in range(1, min(len(h_bbox) ,len(gt_bbox))):
            print(f'frame:{i}')
            update_metric(mot_metric, gt_bbox[i]['obj_id'], gt_bbox[i]['bbox'], h_bbox[i]['obj_id'], h_bbox[i]['bbox'] )
        metrics.append(mot_metric)
        seq_names.append(seq)

    generate_results(seq_names, metrics)

generate()