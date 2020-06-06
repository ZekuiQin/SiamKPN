from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamkpn_tracker import SiamKPNTracker

TRACKS = {
          'SiamKPNTracker': SiamKPNTracker,
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
