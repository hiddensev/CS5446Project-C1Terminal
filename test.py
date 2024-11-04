import numpy as np
import train
from rl_utils import *
from train import load_PPO_model



dict = train.get_trainisition_dict("transition_dict.npy")
print(dict["actions"])
# model = load_PPO_model("ppo_model.pth")
# model.update(dict)

