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


### v0.13 (latest)
Support NNcomponent, you can modify network by config.yaml

### v0.121 
Update Q-learning to more general
Play maze with Q-learning 
Add cfg['misc']['ep_max_step'], if done = True and self.ep_use_step >= self.ep_max_step -> train_done = False
```
python examples/maze/maze.py  examples/maze/Q-learning.yaml
```

### v0.12 
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