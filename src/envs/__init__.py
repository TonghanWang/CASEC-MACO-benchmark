from functools import partial
from smac.env import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
import sys
import os
from .join1 import Join1Env
from .aloha import AlohaEnv
from .fire_fighter import FireFighterEnv
from .prey import PreyEnv
from .sensors import SensorEnv
from .hallway import HallwayEnv
from .dispersion import DispersionEnv
from .climb import ClimbEnv
from .aggregate import AggregateEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["join1"] = partial(env_fn, env=Join1Env)
REGISTRY["aloha"] = partial(env_fn, env=AlohaEnv)
REGISTRY["fire_fighter"] = partial(env_fn, env=FireFighterEnv)
REGISTRY["prey"] = partial(env_fn, env=PreyEnv)
REGISTRY["sensor"] = partial(env_fn, env=SensorEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
REGISTRY["dispersion"] = partial(env_fn, env=DispersionEnv)
REGISTRY["climb"] = partial(env_fn, env=ClimbEnv)
REGISTRY["aggregate"] = partial(env_fn, env=AggregateEnv)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
