# OpenAI Gym

## Run Standalone

```
python gym_basic.py DDPG_CartPole-v0.yaml
```

## Run with Server

1. Run the Server 

```
python ../../server.py
```

2. Run the gym with DDPG

```
python gym_basic_conn.py DDPG_CartPole-v0.yaml
```


## Other gym game

1. MountainCarContinuous-v0

agent input: continous state

agent output: continous action

```
python gym_basic_conn.py DDPG_MountainCarContinuous-v0.yaml
```

2. CartPole-v0

agent input: continous state

agent output: discrete action

```
python gym_basic_conn.py DDPG_CartPole-v0.yaml
```

3. Taxi-v2

agent input: discrete state

agent output: discrete action

```
python gym_basic_conn.py DDPG_CartPole-v0.yaml
```



