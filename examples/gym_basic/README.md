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

algo. input: continous state

algo. output: continous action

```
python gym_basic_conn.py DDPG_MountainCarContinuous-v0.yaml
```

2. CartPole-v0

algo. input: continous state

algo. output: discrete action

```
python gym_basic_conn.py DDPG_CartPole-v0.yaml
```


