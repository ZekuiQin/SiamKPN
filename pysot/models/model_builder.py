from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import focal_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_kpn_head
from pysot.models.neck import get_neck


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build kpn head
        self.kpn_head = get_kpn_head(cfg.KPN.TYPE,
                                     **cfg.KPN.KWARGS)


    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf
        
        return self.zf

    def extract(self,x):
        return self.neck(self.backbone(x))

    def track(self, z, x):
        xf = self.backbone(x)
        xf = self.neck(xf)
        zfe,xfe = z,xf 
       
        if cfg.TRAIN.OFFSETS:
            heatmap, objsize, offsets = self.kpn_head(zfe, xfe)
            return {
                    'zfeature': zfe,
                    'xfeature': xfe,
                    'heatmap': heatmap,
                    'offsets': offsets,
                    'objsize': objsize,
                }
        else:
            heatmap, objsize = self.kpn_head(zfe, xfe)
            return {
                    'zfeature': zfe,
                    'xfeature': xfe,
                    'heatmap': heatmap,
                    'objsize': objsize,
                }
           

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_heatmap = [hm.cuda() for hm in data['label_heatmap']]
        label_objsize = data['label_objsize'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        if cfg.TRAIN.OFFSETS:
            label_offsets = data['label_offsets'].cuda()
            heatmap, objsize, offsets = self.kpn_head(zf, xf)
        else:
            heatmap, objsize = self.kpn_head(zf, xf)
            
        if cfg.TRAIN.INTER_SUPER:
            heatmap = [torch.sigmoid(hm) for hm in heatmap]
            heatmap_loss = [focal_loss(heatmap[i], label_heatmap[i]) for i in range(cfg.TRAIN.STACK)]
            heatmap_loss = sum(heatmap_loss)/cfg.TRAIN.STACK
            pos_inds = label_heatmap[0].gt(0)
            indx = torch.cat((pos_inds,pos_inds),1)
            if cfg.TRAIN.OFFSETS:
                if cfg.TRAIN.SAMEOFF:
                    offsets_loss = [F.smooth_l1_loss(offsets[i][indx],label_offsets[indx]) for i in range(cfg.TRAIN.STACK)]
                else:
                    weight_offsets = label_heatmap[0][pos_inds]
                    weight_offsets = torch.cat((weight_offsets,weight_offsets),0)
                    offsets_loss = [F.smooth_l1_loss(weight_offsets*offsets[i][indx],weight_offsets*label_offsets[indx]) for i in range(cfg.TRAIN.STACK)]
                offsets_loss = sum(offsets_loss)/cfg.TRAIN.STACK
            objsize_loss = [F.smooth_l1_loss(objsize[i][indx],label_objsize[indx]) for i in range(cfg.TRAIN.STACK)]
            objsize_loss = sum(objsize_loss)/cfg.TRAIN.STACK
        else:
            heatmap = torch.sigmoid(heatmap[-1])
            heatmap_loss = focal_loss(heatmap, label_heatmap[0])
            pos_inds = label_heatmap[0].gt(0)
            indx = torch.cat((pos_inds,pos_inds),1)
            if cfg.TRAIN.OFFSETS:
                if cfg.TRAIN.SAMEOFF:
                    offsets_loss = F.smooth_l1_loss(offsets[-1][indx],label_offsets[indx])
                else:
                    weight_offsets = label_heatmap[0][pos_inds]
                    weight_offsets = torch.cat((weight_offsets,weight_offsets),0)
                    offsets_loss = F.smooth_l1_loss(weight_offsets*offsets[-1][indx],weight_offsets*label_offsets[indx])
            objsize_loss = F.smooth_l1_loss(objsize[-1][indx],label_objsize[indx])

        outputs = {}
        if cfg.TRAIN.OFFSETS:
            outputs['total_loss'] = cfg.TRAIN.HM_WEIGHT * heatmap_loss + \
                cfg.TRAIN.OF_WEIGHT * offsets_loss + cfg.TRAIN.WH_WEIGHT * objsize_loss
            outputs['offsets_loss'] = offsets_loss
        else:
            outputs['total_loss'] = cfg.TRAIN.HM_WEIGHT * heatmap_loss + \
                cfg.TRAIN.WH_WEIGHT * objsize_loss
        outputs['heatmap_loss'] = heatmap_loss
        outputs['objsize_loss'] = objsize_loss

        return outputs
