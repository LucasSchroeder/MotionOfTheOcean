import numpy as np
import copy
import os
import time
import sys

from enum import Enum

from pybullet_envs.deep_mimic.learning.path import *
from pybullet_envs.deep_mimic.learning.exp_params import ExpParams
from pybullet_envs.deep_mimic.learning.normalizer import Normalizer
from pybullet_envs.deep_mimic.learning.replay_buffer import ReplayBuffer
from pybullet_utils.logger import Logger
import pybullet_utils.mpi_util as MPIUtil
import pybullet_utils.math_util as MathUtil


class RLAgent():

  class Mode(Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_END = 2

  NAME = "None"

  UPDATE_PERIOD_KEY = "UpdatePeriod"
  ITERS_PER_UPDATE = "ItersPerUpdate"
  DISCOUNT_KEY = "Discount"
  MINI_BATCH_SIZE_KEY = "MiniBatchSize"
  INIT_SAMPLES_KEY = "InitSamples"
  NORMALIZER_SAMPLES_KEY = "NormalizerSamples"

  OUTPUT_ITERS_KEY = "OutputIters"
  INT_OUTPUT_ITERS_KEY = "IntOutputIters"
  TEST_EPISODES_KEY = "TestEpisodes"

  EXP_ANNEAL_SAMPLES_KEY = "ExpAnnealSamples"
  EXP_PARAM_BEG_KEY = "ExpParamsBeg"
  EXP_PARAM_END_KEY = "ExpParamsEnd"

  def __init__(self, world, id, json_data):
    self.world = world
    self.id = id
    self.logger = Logger()
    self._mode = self.Mode.TRAIN

    self._enable_training = True
    self.iter = int(0)
    self.start_time = time.time()
    self._update_counter = 0

    self.update_period = 1.0  # simulated time (seconds) before each training update
    self.iters_per_update = int(1)
    self.discount = 0.95
    self.mini_batch_size = int(32)
    self.init_samples = int(1000)
    self.normalizer_samples = np.inf
    self._local_mini_batch_size = self.mini_batch_size  # batch size for each work for multiprocessing
    self._need_normalizer_update = True
    self._total_sample_count = 0

    self.output_iters = 100
    self.int_output_iters = 100

    self.train_return = 0.0
    self.test_episodes = int(0)
    self.test_episode_count = int(0)
    self.test_return = 0.0
    self.avg_test_return = 0.0

    self.exp_anneal_samples = 320000
    self.exp_params_beg = ExpParams()
    self.exp_params_end = ExpParams()
    self.exp_params_curr = ExpParams()

    self._load_params(json_data)
    # self._build_normalizers()
    self._build_bounds()

    return

  def has_goal(self):
    return self.get_goal_size() > 0

  def set_enable_training(self, enable):
    print("set_enable_training=", enable)
    self._enable_training = enable
    if (self._enable_training):
      self.reset()
    return

  def enable_testing(self):
    return self.test_episodes > 0

  def need_new_action(self):
    return self.world.env.need_new_action(self.id)

  def _build_bounds(self):
    self.a_bound_min = self.world.env.build_action_bound_min(self.id)
    self.a_bound_max = self.world.env.build_action_bound_max(self.id)
    return

  def _load_params(self, json_data):
    if (self.UPDATE_PERIOD_KEY in json_data):
      self.update_period = int(json_data[self.UPDATE_PERIOD_KEY])

    if (self.ITERS_PER_UPDATE in json_data):
      self.iters_per_update = int(json_data[self.ITERS_PER_UPDATE])

    if (self.DISCOUNT_KEY in json_data):
      self.discount = json_data[self.DISCOUNT_KEY]

    if (self.MINI_BATCH_SIZE_KEY in json_data):
      self.mini_batch_size = int(json_data[self.MINI_BATCH_SIZE_KEY])

    if (self.INIT_SAMPLES_KEY in json_data):
      self.init_samples = int(json_data[self.INIT_SAMPLES_KEY])

    if (self.NORMALIZER_SAMPLES_KEY in json_data):
      self.normalizer_samples = int(json_data[self.NORMALIZER_SAMPLES_KEY])

    if (self.OUTPUT_ITERS_KEY in json_data):
      self.output_iters = json_data[self.OUTPUT_ITERS_KEY]

    if (self.INT_OUTPUT_ITERS_KEY in json_data):
      self.int_output_iters = json_data[self.INT_OUTPUT_ITERS_KEY]

    if (self.TEST_EPISODES_KEY in json_data):
      self.test_episodes = int(json_data[self.TEST_EPISODES_KEY])

    if (self.EXP_ANNEAL_SAMPLES_KEY in json_data):
      self.exp_anneal_samples = json_data[self.EXP_ANNEAL_SAMPLES_KEY]

    if (self.EXP_PARAM_BEG_KEY in json_data):
      self.exp_params_beg.load(json_data[self.EXP_PARAM_BEG_KEY])

    if (self.EXP_PARAM_END_KEY in json_data):
      self.exp_params_end.load(json_data[self.EXP_PARAM_END_KEY])

    num_procs = MPIUtil.get_num_procs()
    self._local_mini_batch_size = int(np.ceil(self.mini_batch_size / num_procs))
    self._local_mini_batch_size = np.maximum(self._local_mini_batch_size, 1)
    self.mini_batch_size = self._local_mini_batch_size * num_procs

    assert (self.exp_params_beg.noise == self.exp_params_end.noise)  # noise std should not change
    self.exp_params_curr = copy.deepcopy(self.exp_params_beg)
    self.exp_params_end.noise = self.exp_params_beg.noise

    self._need_normalizer_update = self.normalizer_samples > 0

    return

  def _record_state(self):
    s = self.world.env.record_state(self.id)
    return s

  def _record_goal(self):
    g = self.world.env.record_goal(self.id)
    return g

  def _record_reward(self):
    r = self.world.env.calc_reward(self.id)
    return r

  def _apply_action(self, a):
    self.world.env.set_action(self.id, a)
    return

  def _end_path(self):
    s = self._record_state()
    g = self._record_goal()
    r = self._record_reward()

    self.path.rewards.append(r)
    self.path.states.append(s)
    self.path.goals.append(g)
    self.path.terminate = self.world.env.check_terminate(self.id)

    return

  def _update_test_return(self, path):
    path_reward = path.calc_return()
    self.test_return += path_reward
    self.test_episode_count += 1
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

  def _update_mode_train(self):
    return

  def _update_mode_train_end(self):
    self._init_mode_test()
    return

  def _update_mode_test(self):
    if (self.test_episode_count * MPIUtil.get_num_procs() >= self.test_episodes):
      global_return = MPIUtil.reduce_sum(self.test_return)
      global_count = MPIUtil.reduce_sum(self.test_episode_count)
      avg_return = global_return / global_count
      self.avg_test_return = avg_return

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
    self.test_return = 0.0
    self.test_episode_count = 0
    self.world.env.set_mode(self._mode)
    return

  def _enable_output(self):
    return MPIUtil.is_root_proc() and self.output_dir != ""

  def _enable_int_output(self):
    return MPIUtil.is_root_proc() and self.int_output_dir != ""

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

  def _enable_draw(self):
    return self.world.env.enable_draw


  def _get_iters_per_update(self):
    return MPIUtil.get_num_procs() * self.iters_per_update