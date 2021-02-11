REGISTRY = {}


from .rnn_agent import RNNAgent
from .rnn_agent import PairRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["pair_rnn"] = PairRNNAgent
