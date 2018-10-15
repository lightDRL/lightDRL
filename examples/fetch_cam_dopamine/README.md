
# This is directory use dopamine's rainbow to test the environment

Download dopamine from https://github.com/google/dopamine/

Note: let dopamine in the PYTHONPATH

# Test dopamine's rainbow

```
python train.py \
  --agent_name=rainbow \
  --base_dir=../../data_pool/fetch_cam_rainbow \
  --gin_files='rainbow.gin'
```

# Acknowledge

Marc G. Bellemare, Pablo Samuel Castro, Carles Gelada, Saurabh Kumar, Subhodeep Moitra. Dopamine, https://github.com/google/dopamine, 2018.