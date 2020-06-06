from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import cv2
import numpy as np
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker


class SiamKPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamKPNTracker, self).__init__()
        self.name = 'siamkpn_tracker'
        self.stride = cfg.TRAIN.STRIDE
        self.mink = int(32/self.stride) 
        self.topk = self.mink*4 
        self.threshold = cfg.TRACK.TH  
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.TRAIN.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), 1)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.frame = 1
        self.feature_size = 31
    
    def generate_anchor(self,score_size):
        anchors = np.zeros((2, score_size*score_size), dtype=np.float32)
        for i in range(score_size):
            for j in range(score_size):
                anchors[0][i*score_size+j] =  self.stride*(j-score_size//2)
                anchors[1][i*score_size+j] =  self.stride*(i-score_size//2)
        return anchors

    def _convert_bbox(self, offsets, objsize, topk_rank, num):
        objsize = objsize.contiguous().view(2,-1).data.cpu().numpy()
        objsize = objsize[:,topk_rank]
        anchors = self.anchors[:,topk_rank]
        pre_bbox = np.zeros((4, num))

        if cfg.TRACK.OFFSETS:
            offsets = offsets.contiguous().view(2,-1).data.cpu().numpy()
            offsets = offsets[:,topk_rank]
            pre_bbox[0, :] = offsets[0, :] * 64 + anchors[0, :]
            pre_bbox[1, :] = offsets[1, :] * 64 + anchors[1, :]
        else:
            pre_bbox[0, :] = anchors[0, :]
            pre_bbox[1, :] = anchors[1, :]

        if cfg.TRAIN.NORMWH:
            pre_bbox[2, :] = np.exp(objsize[0, :])*64
            pre_bbox[3, :] = np.exp(objsize[1, :])*64
        else:
            pre_bbox[2, :] = objsize[0, :]
            pre_bbox[3, :] = objsize[1, :]
        return pre_bbox


    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.template = self.model.template(z_crop)
        if self.frame>1:
            self.frame += 5


    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        with torch.no_grad():
            outputs = self.model.track(self.template, x_crop)

        hm_score = outputs['heatmap'][-1].contiguous().view(1,-1)
        score, rank = hm_score.sort(descending=True)
        topk_score = score[:,:self.topk]
        score = torch.sigmoid(topk_score)
        score_threshold = score.gt(self.threshold)
        idx_number = score_threshold.sum().cpu().item()
        num = max(self.mink,min(idx_number,self.topk))
        topk_rank = rank[0,:num].cpu().numpy()

        score = score[:,:num].data[0,:].cpu().numpy()
        if cfg.TRAIN.OFFSETS:
            off = outputs['offsets'][-1]
        else:
            off = 0
        pred_bbox = self._convert_bbox(off, outputs['objsize'][-1],topk_rank,num)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        window = self.window[topk_rank]
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]
        self.frame += 1
        return {
                'bbox': bbox,
                'best_score': best_score,
                'hm': outputs['heatmap'],
                'xcrop': x_crop,
                'scale_z': scale_z
               }
