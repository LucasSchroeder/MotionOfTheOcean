import os
import sys
import inspect
from enum import Enum

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from pybullet_envs.deep_mimic.env.env import Env
from custom_reward import getRewardCustom
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger

import os
import json
import csv

save_name = ''


class RLWorld(object):

    def __init__(self, env, arg_parser):
        # TFUtil.disable_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        self.env = env
        self.arg_parser = arg_parser
        self._enable_training = True
        self.build_agents()

        return

    def get_enable_training(self):
        return self._enable_training

    def set_enable_training(self, enable):
        self._enable_training = enable
        
        self.world_agent.enable_training = self.enable_training

        if (self._enable_training):
            self.env.set_mode(CustomAgent.Mode.TRAIN)
        else:
            self.env.set_mode(CustomAgent.Mode.TEST)

        return

    enable_training = property(get_enable_training, set_enable_training)

    def shutdown(self):
        self.env.shutdown()
        return

    def build_agents(self):
        self.world_agent = CustomAgent(self, 0)

        model_files = self.arg_parser.parse_strings('model_files')
        assert (len(model_files) == 1 or len(model_files) == 0)

        if (len(model_files) > 0):
                curr_model_file = model_files[0]
                if curr_model_file != 'none':
                    self.world_agent.load_model(os.getcwd() + "/" + curr_model_file)

        self.set_enable_training(self.enable_training)

        return

    def update(self, timestep):
        self.env.update(timestep)

        # compute next state
        next_state = self.env._humanoid.getState()

        # compute reward
        kinPose = self.env._humanoid.computePose(self.env._humanoid._frameFraction)
        reward = getRewardCustom(kinPose, self.env._humanoid)

        # compute whether episode is done
        is_done = self.env.is_episode_end()
        return next_state, reward, is_done

    def reset(self):
        self.env.reset()
        return


class custom_critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(512, activation='relu')
        # TODO: there should be another layer here 
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        layer2 = self.d2(x)
        v = self.v(layer2)
        return v


class custom_actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.num_actions = 36
        self.d1 = tf.keras.layers.Dense(1024, activation='relu')
        self.d2 = tf.keras.layers.Dense(512, activation='relu')
        self.a = tf.keras.layers.Dense(self.num_actions)

    def call(self, input_data):
        layer1 = self.d1(input_data)
        layer2 = self.d2(layer1)
        a = self.a(layer2)
        return a


class CustomAgent():
    class Mode(Enum):
        TRAIN = 0
        TEST = 1
    def __init__(self, world, id, gamma=0.95):
        self.world = world
        self.id = id
        self.gamma = gamma
        self.discount_factor = 0.95

        self.a_opt = tf.keras.optimizers.SGD(learning_rate=0.00005,
                                             momentum=0.9, clipnorm=1.0)  # policy step size of α(π) = 5 × 10^(−5)
        self.c_opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)  # value stepsize of α(v) = 10^(−2)
        self.actor = custom_actor()
        self.critic = custom_critic()
        self.clip_pram = 0.2

        self.num_actions = 36

        self.action_bound_min = self.world.env.build_action_bound_min(self.id)
        self.action_bound_max = self.world.env.build_action_bound_max(self.id)

    def _apply_action(self, a):
        self.world.env.set_action(self.id, a)
        return

    def load_model(self, in_path):
        print("LOADED MODEL")
        self.actor = keras.models.load_model(in_path)

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        mu = tf.math.reduce_mean(prob)
        std = tf.math.reduce_std(prob)

        # the mean should be a sample of distributions (whatever the actor returned)
        dist = tfp.distributions.Normal(loc=prob, scale=std)
        action = dist.sample()
        return action.numpy()[0]

    def actor_loss(self, probability, actions, adv, old_probs, closs):
        # normalize probability
        max_pb = tf.math.reduce_max(probability)
        min_pb = tf.math.reduce_min(probability)
        probability = [(x - min_pb + .0000001) / (max_pb - min_pb) for x in probability]
        probability = tf.convert_to_tensor(probability)

        # normalize old probability
        max_old_pb = tf.math.reduce_max(old_probs)
        min_old_prob = tf.math.reduce_min(old_probs)
        old_probs = [(x - min_old_prob + .0000001) / (max_old_pb - min_old_prob) for x in old_probs]
        old_probs = tf.convert_to_tensor(old_probs)
        
        ratio = tf.math.exp(tf.math.log(probability + 1e-10) - tf.math.log(old_probs + 1e-10))
        
        non_clipped = tf.math.multiply(tf.transpose(ratio), adv)
        clipped = tf.math.multiply(tf.clip_by_value(tf.transpose(ratio), 1.0 - self.clip_pram, 1.0 + self.clip_pram), adv)

        aloss = -tf.reduce_mean(tf.math.minimum(non_clipped, clipped))
        total_loss = 0.5 * closs + aloss - 0.001 * tf.reduce_mean(-(probability * tf.math.log(probability + 1e-10)))

        return total_loss
    
    def learn(self, states, actions, adv, old_probs, discnt_rewards):
        # discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        # adv = tf.reshape(adv, (len(adv),))

        # old_p = old_probs

        # old_p = tf.reshape(old_p, (len(old_p), self.num_actions))
        with tf.GradientTape() as tape1:
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            c_loss = kls.mean_squared_error(discnt_rewards, v)

        with tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)


        grads1 = tape2.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape1.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


def test_reward(env):
    steps_update_world = 0
    total_reward = 0
    world.reset()
    total_reward_count = 0
    state = env._humanoid.getState()
    done = False

    while not done:
        action = ppo_agent.act(state)

        # take a step with the environment 
        ppo_agent._apply_action(action)
        next_state, reward, done = update_world(world)

        state = next_state
        total_reward_count += reward
        done = world.env.is_episode_end()

    return total_reward_count


# This is the GAE function. It measures how much better off the model can be by taking 
# a particular action when in a particular state. It uses the rewards that we collected
# at each time step and calculates how much of an advantage we were able to obtain by 
# taking the action that we took. 
# 
# So if we took a good action, we want to calculate how much better off we were by taking 
# that action, not only in the short run but also over a longer period of time. This way, 
# even if we do not get good rewards in the next time step after the action we tool, 
# we still look at few time steps after that action into the longer future to see how 
# out model performed in the long run
def advantage_estimation(rewards, done, values, discount_factor, gamma):
    # initialize advantage and empty list
    gae = 0
    returns = []
    # loop backwards through the rewards
    for i in reversed(range(len(rewards))):
        # Define delta
        # The "done" variable can be thought of as a mask value. If the episode is over then
        # the next state in the batch will be from a newly restarted game so we do not want 
        # to consider that and therefore mask value is taken as 0
        delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
        # update gae
        gae = delta + gamma * discount_factor * dones[i] * gae
        # append the return to the list of returns
        returns.append(gae + values[i])

    # reverse the list of returns to restore the original order
    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    returns = np.array(returns, dtype=np.float32)

    return returns, adv


total_reward = 0
steps_update_world = 0
update_timestep = 1. / 240.


def update_world(world):
    next_state, reward, is_done = world.update(update_timestep)

    global total_reward
    total_reward += reward
    global steps_update_world
    steps_update_world += 1

    end_episode = world.env.is_episode_end()
    if (end_episode or steps_update_world >= 1000):
        print("total_reward =", total_reward)
        total_reward = 0
        steps_update_world = 0
        world.reset()
    return next_state, reward, is_done


def build_arg_parser(args, training):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if arg_file == '':
        if training == True:
            arg_file = "train_humanoid3d_backflip_args.txt"
        else: 
            arg_file = "run_humanoid3d_backflip_args.txt"
    if (arg_file != ''):
        path = os.getcwd() + "/args/" + arg_file
        succ = arg_parser.load_file(path)
        Logger.print2(arg_file)
        assert succ, Logger.print2('Failed to load args from: ' + arg_file)

    # update global variable to use when saving the models
    global save_name
    save_name = arg_file.replace('run_humanoid3d_', '').replace('train_humanoid3d_', '').replace('_args.txt', '')
    return arg_parser

def process_motion_data(path_to_motion_file, env):
    with open(path_to_motion_file, 'r') as f:
      motion_data = json.load(f)
    env._mocapData = motion_data


def build_world(args, enable_draw, training = True):
    arg_parser = build_arg_parser(args,training)
    env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
    
    # Process the motion capture data:
    path_to_motion_file = arg_parser.parse_strings('motion_file')[0]
    process_motion_data(path_to_motion_file,env)

    world = RLWorld(env, arg_parser)
    world.reset()

    return world


if __name__ == '__main__':
    args = sys.argv[1:]
    enable_draw = False
    world = build_world(args, enable_draw)

    env = world.env

    tf.random.set_seed(336699)
    ppo_agent = world.world_agent
    ppo_steps = 4096
    mini_batches = 256
    batch_size = ppo_steps/mini_batches
    ep_reward = []
    total_avgr = []
    target_reached = False
    best_reward = 0
    avg_rewards_list = []
    samples_count = 0

    # Delete contents from rewards_log.csv if it exists
    with open(r'rewards_log.csv', 'w+') as f:
        writer = csv.writer(f)
        fields=['# samples trained on','avg_reward']
        writer.writerow(fields)
        f.close()

    while not target_reached:

        done = False
        state = env._humanoid.getState()
        all_aloss = []
        all_closs = []
        rewards = []
        states = []
        actions = []
        probs = []
        dones = []
        values = []
        print("STARTING A NEW EPISODE")

        # for s in range(ppo_steps): #### CHANGE THIS BACK
        for s in range(32):
            print(s)
            action = ppo_agent.act(state)
            value = ppo_agent.critic(np.array([state])).numpy()

            # take a step with the environment 
            ppo_agent._apply_action(action)
            next_state, reward, done = update_world(world)

            dones.append(1 - done)
            rewards.append(reward)
            states.append(state)
            actions.append(action)

            ########################### THIS LINE THAT CALLS PROB IS WRONG ###########
            # The action from the policy specifies target orientations for PD controllers
            # at each joint. IT DOES NOT SPECIFY PROBABILITIES!
            probs.append(ppo_agent.actor(np.array([state])))
            # probs.append(action)
            values.append(value[0][0])
            state = next_state
            samples_count = samples_count + 1

        value = ppo_agent.critic(np.array([state])).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs), ppo_agent.num_actions))
        probs = np.stack(probs, axis=0)

        returns, adv = advantage_estimation(rewards, dones, values,
                                                             ppo_agent.discount_factor, ppo_agent.gamma)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = tf.reshape(returns, (len(returns),))
        adv = tf.reshape(adv, (len(adv),))
        probs = tf.reshape(probs, (len(probs), ppo_agent.num_actions))

        
        for learn_step in range(mini_batches):
            start_index = int(learn_step * batch_size)
            stop_index = int(start_index + batch_size)

            curr_states_batch = states[start_index:stop_index]
            curr_action_batch = actions[start_index:stop_index]
            curr_adv_batch = adv[start_index:stop_index]
            curr_prolicy_batch = probs[start_index:stop_index]
            curr_returns_batch = returns[start_index:stop_index]

            ## Update the gradients
            print("Learning/ Updating Gradients")
            al, cl = ppo_agent.learn(curr_states_batch, curr_action_batch, curr_adv_batch, curr_prolicy_batch, curr_returns_batch)

        avg_reward = np.mean([test_reward(env) for _ in range(32)])
        print(f"TEST REWARD is {avg_reward}")
        avg_rewards_list.append(avg_reward)
        if avg_reward > best_reward:
            print('Saving Model -- reward improved to: ' + str(avg_reward))
            ppo_agent.actor.save('Saved_Models/{}_model_actor'.format(save_name), save_format='tf')
            ppo_agent.critic.save('Saved_Models/{}_model_critic'.format(save_name), save_format='tf')
            best_reward = avg_reward
        if best_reward == 200:
            target_reached = True
        # Reset the environment and the humanoid
        total_reward = 0
        steps_update_world = 0
        world.reset()

        # SAVE CSV FILE
        with open(r'rewards_log.csv', 'a') as f:
            writer = csv.writer(f)
            fields=[samples_count,avg_reward]
            writer.writerow(fields)
            f.close()

    env.close()
