import cv2 as cv
import numpy as np
import util

colours = np.random.rand(100, 3)  # used only for display
cvcolor = np.random.randint(0, 255, (100, 3))
fourcc = cv.VideoWriter_fourcc(*'DIVX')
fourcc = cv.VideoWriter_fourcc(*'MJPG')
video_dim =(1080, 720)

def paste_layer(img, layer):
    gray_layer = cv.cvtColor(layer, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray_layer, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    img1_bg = cv.bitwise_and(img, img, mask=mask_inv)
    layer = cv.add(img1_bg, layer)
    return layer

def draw_centers(bbox, layer):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    cv.circle(layer,(int(x),int(y)), 5, (0, 255, 255), 5)  #

    return layer

def draw_rect_with_label(layer, d, label=None, color = None, label_pos='topleft'):
    if label is None:
        label = d[4]
    d = list(map(int, d))
    if color is None:
        color = cvcolor[label % 100].tolist()
    layer = cv.rectangle(layer, (d[0], d[1]), (d[2], d[3]), color, 5)

    if label_pos == 'topleft':
        l_pos = (d[0], d[1])
        font_col = (0, 255, 0, 255)
        font_size = 2
    elif label_pos == 'topright':
        l_pos = (d[2], d[1])
        font_col = (255,215,0, 0)
        font_size = 2
    cv.putText(layer, str(label),  # text
               l_pos,  # position at which writing has to start
               cv.FONT_HERSHEY_SIMPLEX,  # font family
               font_size,  # font size
               font_col,  # font color
               5)  # font stroke
    return layer

def draw_predicted_bbox(layer, predicted_bboxs):
    predicted_bboxs = predicted_bboxs.astype(int)
    if len(predicted_bboxs)>0:
        for  bbox in predicted_bboxs:
            bbox_id = bbox[-1]
            draw_rect_with_label(layer, bbox, str(bbox_id), (225,255,0))

def show_frame(frame, size = (1080, 720), window_name ='Frame', video_writer=None, frame_idx=-1):
    if frame_idx>=0:
        cv.putText(frame, 'Frame: {}'.format(frame_idx),
                      (200, 200), cv.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 255, 255), thickness=5)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.resize(frame, size)
    cv.imshow(window_name, frame)

    if video_writer: video_writer.write(frame)
    return frame

def show_tracks(trackers, frame):
    trk_to_draw ='all'
    trk_img = np.zeros_like(frame)
    trk_img.fill(255)
    overlay = trk_img.copy()
    for trk in trackers.values():
        if trk_to_draw == 'all' or trk.id in trk_to_draw  :
            bbox,_,frm_idx = trk.track_history[0]
            prev_z =  util.convert_bbox_to_z(bbox)
            cv.putText(trk_img, str(trk.id)+','+str(frm_idx), (prev_z[0], prev_z[1]), cv.FONT_HERSHEY_SIMPLEX, fontScale=2,
                       color= cvcolor[trk.id% 100].tolist(),
                       thickness=5)
            for h in trk.track_history:
                bbox, is_live, frame_idx = h
                z = util.convert_bbox_to_z(bbox)
                if is_live:
                    cv.line(trk_img,(prev_z[0], prev_z[1]), (z[0],z[1]), cvcolor[trk.id % 100].tolist(), thickness=7)
                else:
                    cv.line(overlay, (prev_z[0], prev_z[1]), (z[0], z[1]), cvcolor[trk.id % 100].tolist(), thickness=4)
                prev_z = z
    size = (1080, 720)
    alpha = .2
    img = cv.addWeighted(overlay, alpha, trk_img, 1 - alpha, 0)
    trk_img = cv.resize(img, size)  # 1605, 1192, 1800, 1361
    if util.params['save']['save_track']:
        cv.imwrite('../../output/track_img.png', trk_img)
    cv.imshow("Tracks", trk_img)

def draw_bboxs(layer, bboxs, color=(255,0,0)):
    for bbox_idx, bbox in enumerate(bboxs):
        layer = cv.rectangle(layer, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness=5)
        cv.putText(layer, str(bbox_idx),  # text
                   (bbox[0], bbox[1]),  # position at which writing has to start
                   cv.FONT_HERSHEY_SIMPLEX,  # font family
                   2,  # font size
                   (0, 255, 0, 255),  # font color
                   5)  # font stroke
    return layer

def draw_rect_debug(layer, d, label=None):
    layer = cv.rectangle(layer, (d[0], d[1]), (d[2], d[3]), (255,0,0), 10)
    if label is None:
        label = 'Debug'
    cv.putText(layer, label,  # text
               (d[0], d[1]),  # position at which writing has to start
               cv.FONT_HERSHEY_SIMPLEX,  # font family
               2,  # font size
               (0, 255, 0, 255),  # font color
               5)  # font stroke
    return layer

def show_output(img, gt_for_disp, matched_trk, unmatched_trk,
                mask, flow_layer=None, flow_layer_new=None,
                show_optflow=False,
                video_writer = None):
    cv.line(mask, (3840-250, 0), (3840-250, 2160), (255,0,0), 2)
    cv.line(mask, (250, 0), (250, 2160), (255,0,0), 2)

    for d in matched_trk:
        d = d.astype(np.int32)
        mask = draw_rect_with_label(mask, d)

    if show_optflow:
        mask = cv.add(flow_layer, mask)
    cvimg = paste_layer(img, mask)

    show_frame(cvimg)
    if video_writer is not None:
        cvimg = cv.resize(cvimg, video_dim)
        video_writer.write(cv.cvtColor(cvimg, cv.COLOR_RGB2BGR))
    return flow_layer
