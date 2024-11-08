import numpy as np

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
from models import *
from train import *
import torch
import pickle

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        #gamelib.debug_write('Random seed: {}'.format(seed))

        # initialize the agent
        self.agent = load_PPO_model("./ppo_model.pth").eval()

        self.state = np.zeros(426)
        self.next_state = np.zeros(426)
        self.action = np.zeros(210 * 8)
        self.reward = 0
        self.my_health = 30
        self.en_health = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize the transition dict
        self.transition_dict = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_states": [],
        }

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        #gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP, UNIT_TYPE_TO_INDEX

        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        UNIT_TYPE_TO_INDEX = {
            WALL: 0,
            SUPPORT: 1,
            TURRET: 2,
            SCOUT: 3,
            DEMOLISHER: 4,
            INTERCEPTOR: 5,
        }
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        #gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        # for the first turn, we have no idea of next_state, rewards, just update the state
        if game_state.turn_number == 0:
            self.state = self.get_state(game_state)
        # for the second turn and turns after, we can record state, next_state, action and reward
        else:
            self.next_state = self.get_state(game_state)
            self.reward = self.get_reward(game_state)

            self.transition_dict['states'].append(self.state)
            self.transition_dict['actions'].append(self.action)
            self.transition_dict['rewards'].append(self.reward)
            self.transition_dict['next_states'].append(self.next_state)
            self.transition_dict['dones'].append(0)

            self.state = self.next_state

        self.update_health(game_state)
        action = self.get_actions(self.state)
        self.action = action.detach().numpy()
        self.take_action(action, game_state)

        game_state.submit_turn()

    def get_state(self, game_state):
        res = np.zeros(426)-1
        # get resources
        my_points = game_state.get_resources(0)
        en_points = game_state.get_resources(1)
        res[211], res[212] = my_points[0], my_points[1]
        res[424], res[425] = en_points[0], en_points[1]

        # get health
        my_health = game_state.my_health
        en_health = game_state.enemy_health
        res[210] = my_health
        res[423] = en_health

        # get constructions:
        for i in range(28):
            for j in range(28):
                if game_state.game_map.in_arena_bounds((i, j)):
                    state_pos = self.game_pos_map(i, j)
                    units = game_state.game_map[i, j]
                    if units:
                        res[state_pos] = UNIT_TYPE_TO_INDEX[units[0].unit_type]
        return res



    def update_health(self, game_state):
        self.my_health = game_state.my_health
        self.en_health = game_state.enemy_health

    def get_reward(self, game_state):
        # main objective: reduce enemy health and reserve my health
        health_diff = 50*(self.en_health - game_state.enemy_health) - 100*(self.my_health - game_state.my_health)
        # invalid actions are punished in take_action() for quick learning
        return self.reward + health_diff

    # state dim: 426
    # return dim: 8 * 210
    def get_actions(self, states):
        input_state = torch.tensor(np.array([states]), dtype=torch.float).to(self.device)
        actions = self.agent.actor(input_state)
        actions = torch.reshape(actions, (8, -1))
        #gamelib.debug_write(f"Actions shape: {actions.shape}")
        return actions

    #
    def take_action(self, actions, game_state):
        # 0, 1, 2: Construction
        # 3, 4, 5: Attack
        # 6: Upgrade
        # 7: Remove
        for idx, action in enumerate(actions):
            # # normalize the probability map of this action
            # action,_ = torch.sort(F.softmax(action, dim=0))
            # filter out locations with probability less than 0.0005
            selected_idx = torch.nonzero(action>0.5).squeeze().numpy()
            print(selected_idx)
            # get the locations
            locations = []
            for i in selected_idx:
                x, y = self.state_pos_map(i)
                locations.append([x, y])

            # filter out invalid locations and reward valid actions
            self.reward = 0 # reset from last turn
            ori_loc_num = len(locations)
            if idx in [0, 1, 2]: # construction only place at empty spots
                locations = [loc for loc in locations if len(game_state.game_map[loc[0], loc[1]]) == 0]
                self.reward += len(locations)
            elif idx in [3, 4, 5]: # attack only place at borders
                borders = game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)
                locations = [loc for loc in locations if loc in borders]
                self.reward += 5*len(locations) # encourage attacking
            elif idx in [6, 7]: # only remove/upgrade if there is a unit
                locations = [loc for loc in locations if len(game_state.game_map[loc[0], loc[1]]) > 0]
                self.reward += len(locations)
            # self.reward -= 0.5*(ori_loc_num - len(locations))

            # WALL
            if idx == 0 and locations:
                game_state.attempt_spawn(WALL, locations)
            # SUPPORT
            elif idx == 1 and locations:
                game_state.attempt_spawn(SUPPORT, locations)
            # TURRET
            elif idx == 2 and locations:
                game_state.attempt_spawn(TURRET, locations)
            # SCOUT
            elif idx == 3 and locations:
                game_state.attempt_spawn(SCOUT, locations)
            # DEMOLISHER
            elif idx == 4 and locations:
                game_state.attempt_spawn(DEMOLISHER, locations)
            # INTERCEPTOR
            elif idx == 5 and locations:
                game_state.attempt_spawn(INTERCEPTOR, locations)
            # REMOVE
            elif idx == 6 and locations:
                game_state.attempt_remove(locations)
            # UPGRADE
            elif idx == 7 and locations:
                game_state.attempt_upgrade(locations)


    # pos range from 0 - 209
    def state_pos_map(self, pos):
        x, y = 13, 0
        count = 28
        while pos >= count:
            x -= 1
            y += 1
            pos -= count
            count -= 2
        return x, y + pos


    def game_pos_map(self, x, y):
        count = 28
        res = 0
        if x >= 14:
            while x > 14:
                res += count
                count -= 2
                x -= 1
            res += y - (28 - count) // 2 + 213
        else:
            while x < 13:
                res += count
                x += 1
                count -= 2
            res += y - (28 - count) // 2
        return res


    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]

        # if meet the end frame, save all the states into json file
        if int(state.get("turnInfo")[0]) == 2:
            self.transition_dict['dones'] = 0
            np.save('transition_dict.npy', self.transition_dict)

        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                #gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                #gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
