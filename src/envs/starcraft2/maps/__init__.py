from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .smac_maps import get_smac_map_registry


def get_map_params(map_name):
    map_param_registry = get_smac_map_registry()
    return map_param_registry[map_name]