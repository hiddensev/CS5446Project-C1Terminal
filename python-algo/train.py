import torch
from models import *
from rl_utils import *
import os
from tqdm import tqdm

def init_PPO_model(model_pth):
    agent = PPO()
    torch.save(agent.state_dict(), model_pth)


def load_PPO_model(model_pth):
    agent = PPO()
    agent.load_state_dict(torch.load(model_pth))
    return agent


def get_transition_dict(file):
    load_dict = np.load(file,allow_pickle=True).item()
    return load_dict


def train(epochs, run_dir, alg1_dir, alg2_dir, model_dir, model_init = True):
    return_list = []
    if model_init:
        init_PPO_model(model_dir)
    for i in tqdm(range(epochs), desc="Training Progress"):
        print(f"\nEpoch {i+1}/{epochs}")
        
        os.system(run_dir + " " + alg1_dir + " " + alg2_dir)
        transition_dict = get_transition_dict("transition_dict.npy")
        agent = load_PPO_model(model_pth=model_dir)
        
        # Get training metrics
        episode_return = agent.update(transition_dict)
        return_list.append(episode_return)

        # Print current episode return
        print(f"Episode Return: {episode_return:.4f}")
        if len(return_list) > 1:
            print(f"Average Return (last 5): {np.mean(return_list[-5:]):.4f}")
        
        torch.save(agent.state_dict(), model_dir)


if __name__ == "__main__":
    epochs = 1
    run_dir = "/Users/hiddensev/Documents/NUS/5446-AI-planning/Project/original/scripts/run_match.sh"
    alg1_dir = "/Users/hiddensev/Documents/GitHub/hiddensev/CS5446Project-C1Terminal/python-algo"
    alg2_dir = "/Users/hiddensev/Documents/NUS/5446-AI-planning/Project/original/python-algo"
    model_dir = "./ppo_model.pth"
    train(epochs, run_dir, alg1_dir, alg2_dir, model_dir, model_init = True)

# Colab





