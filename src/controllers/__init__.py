REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .casec_controller import CASECMAC
REGISTRY['casec_mac'] = CASECMAC