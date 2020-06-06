from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.kpn import DepthwiseKPN, MultiKPN


KPNS = {
        'DepthwiseKPN': DepthwiseKPN,
        'MultiKPN': MultiKPN
       }


def get_kpn_head(name, **kwargs):
    return KPNS[name](**kwargs)

