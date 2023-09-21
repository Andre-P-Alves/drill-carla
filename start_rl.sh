
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export WANDB__API_KEY=950ec32be299129719a39dd34583f2b6c64ed9ed
screen -L -S carla_rl .venv/bin/python rl_train.py