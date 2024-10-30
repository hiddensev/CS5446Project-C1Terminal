import torch
import json
from models import *
import os

def init_PPO_model(model_pth):
    agent = PPO()
    torch.save(agent.state_dict(), model_pth)


def load_PPO_model(model_pth):
    agent = PPO()
    agent.load_state_dict(torch.load(model_pth))
    return agent


def get_trainisition_dict(file):
    tf = open(file, "r")
    transition_dict = json.load(tf)
    return transition_dict


def train(epochs, run_dir, alg1_dir, alg2_dir, model_dir, model_init = True):
    return_list = []
    if model_init:
        init_PPO_model(model_dir)
    for i in range(epochs):
        os.system(run_dir + " " + alg1_dir + " " + alg2_dir)
        transition_dict = get_trainisition_dict("transition_dict.json")
        agent = load_PPO_model(model_pth=model_dir)
        agent.update(transition_dict)
        torch.save(agent.state_dict(), model_dir)


if __name__ == "__main__":
    epochs = 1500
    run_dir = "/Users/xutao/Desktop/CS5446/C1GamesStarterKit/scripts/run_match.sh"
    alg1_dir = "/Users/xutao/Desktop/CS5446/CS5446Project-C1Terminal/python-algo"
    alg2_dir = "/Users/xutao/Desktop/CS5446/C1GamesStarterKit/python-algo"
    model_dir = "./ppo_model.pth"
    train(epochs, run_dir, alg1_dir, alg2_dir, model_dir, model_init = True)

# Colab





