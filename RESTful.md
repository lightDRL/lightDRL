# Create Session
```
Type: WebSocket
URI: /session
Request: config.yaml
Response: 
      client_id:  [dynamic id]
      predict_url: http://[ip]:[port]/[client_id]/predict
      train_url: http://[ip]:[port]/[client_id]/train
      info_url: http://[ip]:[port]/[client_id]/info
      dashboard: http://[ip]:[port]/[client_id]/dashboard
      EXAMPLE:
        {client_id: 2df12cdtw, 
        predict_url: http://127.0.0.1:5555/2df12cdtw/predict,
        train_url: http://127.0.0.1:5555/2df12cdtw/train,
        info_url: http://127.0.0.1:5555/2df12cdtw/info,
        dashboard: http://[ip]:[port]/[client_id]/dashboard }
```
# Predict
```
Type: WebSocket
URI: /[client_id]/predict
Request: 
      env_id: [id] (Depend on algorithm: Q-Learning, DQN dosen't use it, but A3C use it for logging)
      state: [data]
      EXAMPLE:
        {data: [2,5] , env_id: 3} 
Response:
      action: data
      EXAMPLE:
        {action: 3}
```
# Train
```
Type: WebSocket
URI: /[client_id]/train
Request: 
      state: [data]
      next_state: [data]
      reward: data
      action: data
      done: boolean
      EXAMPLE
        {state : [2,4] , next_state: [4,5], reward: 0.5, action: 3, done: true } 
Response: 
      receive: boolean 
      EXAMPLE
        {receive: true}
```

# Info
```
Type: WebSocket
URI: /[client_id]/info
Request: {type: setting}
Response: config.yaml
```

# Dashboard
```
Type: HTTP
URI: /[client_id]/dashboard
Response: html format for training information

URI: /[client_id]/dashboard/data_pool
Response: html format for all state, reward and done

URI: /[client_id]/dashboard/model_pool
Response: html format for all models
```
