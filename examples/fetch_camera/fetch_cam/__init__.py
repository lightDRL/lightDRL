print('in fetch_cam __init__')

from fetch_cam.fetch_pick_place_camera import FetchCameraEnv
from fetch_cam.fetch_discrete 	  import FetchDiscreteEnv
from fetch_cam.fetch_discrete_cam import FetchDiscreteCamEnv

import gym
from gym.envs.registration import registry, make, spec
def register(id,*args,**kvargs):
	if id in registry.env_specs:
		return
	else:
		return gym.envs.registration.register(id,*args,**kvargs)


register(
	id='FetchCameraEnv-v0',
	entry_point='fetch_cam:FetchCameraEnv',
	timestep_limit=1000,
	reward_threshold=950.0,
)


register(
	id='FetchDiscreteEnv-v0',
	entry_point='fetch_cam:FetchDiscreteEnv',
	timestep_limit=1000,
	reward_threshold=950.0,
)