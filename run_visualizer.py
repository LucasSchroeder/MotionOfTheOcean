import time
import os
import inspect
import json
from learning.rl_world import RLWorld
from ppo_example import build_world, update_world

from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
import sys
import random
from custom_reward import getRewardCustom

update_timestep = 1. / 240.
animating = True
step = False
total_reward = 0
steps = 0


args = sys.argv[1:]


# env= gym.make("CartPole-v0")
# low = env.observation_space.low
# high = env.observation_space.high




if __name__ == '__main__':

  world = build_world(args, True)
  env = world.env
  state = env._humanoid.getState()
  agentoo7 = world.agents[0]

  while (world.env._pybullet_client.isConnected()):

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      animating = not animating
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
    if (animating or step):
      action = agentoo7.act(state)
      agentoo7._apply_action(action)
      state, reward, done = update_world(world)

      #   total_reward_count += reward
      step = False
