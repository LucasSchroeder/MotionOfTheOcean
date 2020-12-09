import numpy as np
import copy
import os
import time
import sys

from enum import Enum

from pybullet_utils.logger import Logger
import pybullet_utils.mpi_util as MPIUtil


class RLAgent():

  class Mode(Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_END = 2

  DISCOUNT_KEY = "Discount"
  TEST_EPISODES_KEY = "TestEpisodes"

  def __init__(self, json_data):
    self._mode = self.Mode.TRAIN

    self._enable_training = True
    self.discount = 0.95

    self.test_episodes = int(0)
    self.test_episode_count = int(0)

    self._load_params(json_data)
    self._build_bounds()

    return

  def set_enable_training(self, enable):
    print("set_enable_training=", enable)
    self._enable_training = enable
    return

  def enable_testing(self):
    return self.test_episodes > 0

  def _build_bounds(self):
    self.a_bound_min = self.world.env.build_action_bound_min(self.id)
    self.a_bound_max = self.world.env.build_action_bound_max(self.id)
    return

  def _load_params(self, json_data):

    if (self.DISCOUNT_KEY in json_data):
      self.discount = json_data[self.DISCOUNT_KEY]

    if (self.TEST_EPISODES_KEY in json_data):
      self.test_episodes = int(json_data[self.TEST_EPISODES_KEY])

    return

  def _update_mode(self):
    if (self._mode == self.Mode.TRAIN):
      self._update_mode_train()
    elif (self._mode == self.Mode.TRAIN_END):
      self._update_mode_train_end()
    elif (self._mode == self.Mode.TEST):
      self._update_mode_test()
    else:
      assert False, Logger.print2("Unsupported RL agent mode" + str(self._mode))
    return

  def _update_mode_train_end(self):
    self._init_mode_test()
    return

  def _update_mode_test(self):
    if (self.test_episode_count * MPIUtil.get_num_procs() >= self.test_episodes):

      if self.enable_training:
        self._init_mode_train()
    return

  def _init_mode_train(self):
    self._mode = self.Mode.TRAIN
    self.world.env.set_mode(self._mode)
    return

  def _init_mode_train_end(self):
    self._mode = self.Mode.TRAIN_END
    return

  def _init_mode_test(self):
    self._mode = self.Mode.TEST
    self.test_episode_count = 0
    self.world.env.set_mode(self._mode)
    return

  def _calc_val_bounds(self, discount):
    r_min = self.world.env.get_reward_min(self.id)
    r_max = self.world.env.get_reward_max(self.id)
    assert (r_min <= r_max)

    val_min = r_min / (1.0 - discount)
    val_max = r_max / (1.0 - discount)
    return val_min, val_max

  def _calc_val_offset_scale(self, discount):
    val_min, val_max = self._calc_val_bounds(discount)
    val_offset = 0
    val_scale = 1

    if (np.isfinite(val_min) and np.isfinite(val_max)):
      val_offset = -0.5 * (val_max + val_min)
      val_scale = 2 / (val_max - val_min)

    return val_offset, val_scale