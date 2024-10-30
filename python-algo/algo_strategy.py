import numpy as np

import gamelib
import random
import math
import warnings
from sys import maxsize
import json
from models import *
from train import *

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
        gamelib.debug_write('Random seed: {}'.format(seed))

        # initialize the agent
        self.agent = load_PPO_model("./ppo_model.pth")

        self.state = np.zeros(213)
        self.next_state = np.zeros(213)
        self.action = np.zeros(210 * 8)
        self.reward = 0

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
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
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
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        if game_state.turn_number == 0:
            self.state = self.get_state(game_state)

        else:
            self.next_state = self.get_state(game_state)
            self.rewards = self.get_rewards(game_state)

            self.transition_dict['states'].append(self.state)
            self.transition_dict['actions'].append(self.action)
            self.transition_dict['rewards'].append(self.reward)
            self.transition_dict['next_states'].append(self.next_state)
            self.transition_dict['dones'].append(0)

            self.state = self.next_state

        self.action = self.get_actions(self.state)
        # store all the things
        self.take_action(self.action)

        game_state.submit_turn()

    def get_state(self, game_state):
        pass
    #
    def get_rewards(self, game_state):
        pass

    # state dim: 213
    def get_actions(self, states):
        return self.agent.take_action(states)

    #
    def take_action(self, actions):
        pass

    # pos range from 0 - 209
    def position_map(pos):
        x, y = 13, 0
        count = 28
        while pos >= count:
            x -= 1
            y += 1
            pos -= count
            count -= 2
        return x, y + pos

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
        if "endStats" in state:
            json_file = open("transition_dict.json", "w")
            json.dump(self.transition_dict, json_file)
            json_file.close()

        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
