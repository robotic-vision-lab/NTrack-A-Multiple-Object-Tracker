import motmetrics as mm
import numpy as np

def update_metric(metric_acc, gt_id, gt_box, match_id, match_bb, w_margin):
    ''''Refine gt and ht - remove boxes which are in margin'''
    l = len(gt_id)
    for i in range(l-1, -1, -1):
        if gt_box[i][0] < w_margin or gt_box[i][0]+gt_box[i][2] > 3840-w_margin:
            gt_id = np.delete(gt_id,i)
            gt_box = np.delete(gt_box,i,axis=0)
    l =len(match_id)-1
    w_margin -=30
    for i in range(l, -1, -1):
        if match_bb[i][0] < w_margin or match_bb[i][0] + match_bb[i][2]> 3840 - w_margin:
            match_id = np.delete(match_id, i)
            match_bb = np.delete(match_bb, i, axis=0)

    dis_mat = mm.distances.iou_matrix(gt_box, match_bb, max_iou=0.65)
    metric_acc.update(gt_id, match_id, dis_mat)

def populate_result(seq_name, metric_acc):
    mh = mm.metrics.create()
    summary = mh.compute(metric_acc, metrics=mm.metrics.motchallenge_metrics,
                         name=seq_name)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print('\n', strsummary)

def populate_combined_results(seq_names, all_metric):
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
