# Light DRL 

## Install

1. Download this package from `git clone git@github.com:lightDRL/lightDRL.git` or download this zip

2. Install python requirements by following command

```
pip install -r requirements.txt

# if fail, try to use sudo 
sudo pip install -r requirements.txt
```

## Run Standalone

Test example which you want to play, for example, play gym

```
python examples/gym_basic/gym_basic.py examples/gym_basic/DDPG.yaml
```

Or you can

```
cd examples/gym_basic
python gym_basic.py DDPG.yaml
```


## Run with Server
```
python server.py
python examples/gym_basic/gym_basic_conn.py examples/gym_basic/DDPG.yaml
```

Or you can

```
cd examples/gym_basic
python ../../server.py
python gym_basic_conn.py DDPG.yaml
```

## Version Note 

### v0.12 (latest)
Support DQN! Now support DQN, DDPG, and Q-learning


Play maze with DQN 
```
python examples/maze/maze.py examples/maze/DQN.yaml
```

### v0.11
Support Q-learning!


```
python examples/gym_basic/gym_basic.py examples/gym_basic/Q-learning_Taxi-v2.yaml
```

### v0.10 
Change framework

Now, support standalone & server-client version !!


### v0.07
Implementation of DDPG (Deep Deterministic Policy Gradient)
and add mobile_avoidance example

1. Run mobile_avoidance.py with DDPG
```
python server.py               config/mobile_avoidance_DDPG.yaml
python examples/mobile_avoidance.py config/mobile_avoidance_DDPG.yaml
```

### v0.06
Implementation Socket.io which replacing ZMQ   
(No ZMQ in the framework now)
Note: Every client use Socket.io link to a new uuid space ->  /[uuid]/

All following app use Socket.io
1. Run two_dof_arm with A3C
```
python server.py               config/two_dof_arm_A3C.yaml
python examples/two_dof_arm.py config/two_dof_arm_A3C.yaml
```
2. Run Gridworld With SARSA
```
python server.py config/gridworld_SARSA.yaml
python examples/gridworld.py config/gridworld_SARSA.yaml
```
3. Run Gridworld With QLearning
```
python server.py config/gridworld_QLearning.yaml
python examples/gridworld.py config/gridworld_QLearning.yaml
```

### v0.05 
1. A3C (continuous) Can Run!!
2. Run two_dof_arm
```
# Run two_dof_arm.py with A3C
python server.py               config/two_dof_arm_A3C.yaml
python examples/two_dof_arm.py config/two_dof_arm_A3C.yaml
```

### v0.04
1. QLearning, SARSA Can Run!!
2. Move example python to examples
```
# Run With SARSA
python server.py config/gridworld_SARSA.yaml
python examples/gridworld.py config/gridworld_SARSA.yaml
# Run with QLearning
python server.py config/gridworld_QLearning.yaml
python examples/gridworld.py config/gridworld_QLearning.yaml
```

### v0.03 
1. Support YAML to config all parameters, including DRL(or RL) methods
2. Parameterize all .py

### v0.02

After run `server.py` and `two_dof_arm.py`

1. Predict:
* `two_dof_arm.py` request env state (7,) to `server.py`
* `server.py` response predict action to  `two_dof_arm.py`

2. Train:
* `two_dof_arm.py` send 5 steps env data to `server.py`

   5  steps env data: state_buf (5,7) action_buf (5,1) reward_buf (5,1), next_state (7,),  done (value)


* `server.py` workers train the parameter to `Main_Net`

