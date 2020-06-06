from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import IoU, corner2center

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

class KeypointTarget:
    def __init__(self,):
        self.stride = cfg.TRAIN.STRIDE
        self.radius = cfg.TRAIN.OUTPUT_SIZE/8 #8
        self.std = self.radius / 2
        if cfg.TRAIN.OFFSETS:
            self.keypoints = np.zeros((2, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.float32)
            for i in range(cfg.TRAIN.OUTPUT_SIZE):
                for j in range(cfg.TRAIN.OUTPUT_SIZE):
                    self.keypoints[0][i][j] = (cfg.TRAIN.SEARCH_SIZE-1)/2 + self.stride*(j-cfg.TRAIN.OUTPUT_SIZE//2) 
                    self.keypoints[1][i][j] = (cfg.TRAIN.SEARCH_SIZE-1)/2 + self.stride*(i-cfg.TRAIN.OUTPUT_SIZE//2) 

    def __call__(self, target, size, neg=False):
        
        heatmap_label0 = np.zeros((1, size, size), dtype=np.float32)
        if cfg.TRAIN.STACK==0:
            heatmap_label = [heatmap_label0]
        else:
            heatmap_label = [heatmap_label0 for i in range(cfg.TRAIN.STACK)]
        objsize_label = np.zeros((2, size, size), dtype=np.float32)
        if cfg.TRAIN.OFFSETS:
            offsets_label = np.zeros((2, size, size), dtype=np.float32)
        if neg:
            if cfg.TRAIN.OFFSETS:
                offsets_label = np.zeros((2, size, size), dtype=np.float32)
                return heatmap_label, offsets_label, objsize_label
            else:
                return heatmap_label, objsize_label

        tcx, tcy, tw, th = corner2center(target)

        heat_cx = cfg.TRAIN.OUTPUT_SIZE//2 + (tcx-(cfg.TRAIN.SEARCH_SIZE-1)/2)/self.stride
        heat_cy = cfg.TRAIN.OUTPUT_SIZE//2 + (tcy-(cfg.TRAIN.SEARCH_SIZE-1)/2)/self.stride
        pos_x = round(heat_cx)
        pos_y = round(heat_cy)

        if cfg.TRAIN.DIF_STD:
            std = [self.std,self.std*0.9,self.std*0.81]
            radius = [self.radius,self.radius,self.radius]  #/2 /4
        else:
            std = [self.std,self.std,self.std]
            radius = [self.radius,self.radius,self.radius]

        for i in range(cfg.TRAIN.OUTPUT_SIZE):
            for j in range(cfg.TRAIN.OUTPUT_SIZE):
                distance = (i-heat_cy)**2 + (j-heat_cx)**2
                if math.sqrt(distance)<self.radius:
                    for idx,hm in enumerate(heatmap_label):
                        if math.sqrt(distance)<radius[idx]:
                            hm[0,i,j] = np.exp(-distance/(2*std[idx]**2))
                    if cfg.TRAIN.OFFSETS:
                        if cfg.TRAIN.SAMEOFF:
                            offsets_label[0,i,j] = (heat_cx - pos_x) * self.stride / 64
                            offsets_label[1,i,j] = (heat_cy - pos_y) * self.stride / 64
                        else:
                            offsets_label[0,i,j] = (tcx - self.keypoints[0,i,j])/64
                            offsets_label[1,i,j] = (tcy - self.keypoints[1,i,j])/64
                    if cfg.TRAIN.NORMWH:
                        objsize_label[0,i,j] = np.log(tw/64)
                        objsize_label[1,i,j] = np.log(th/64)
                    else:
                        objsize_label[0,i,j] = tw
                        objsize_label[1,i,j] = th
                if i==pos_y and j==pos_x:
                    for idx,hm in enumerate(heatmap_label):
                       hm[0,i,j] = 1
        
        if cfg.TRAIN.OFFSETS:
            return heatmap_label, offsets_label, objsize_label
        else:
            return heatmap_label, objsize_label
