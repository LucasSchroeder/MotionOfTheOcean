import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import tensorflow.keras.losses as kls
from rl_agent import RLAgent
import pybullet_utils.mpi_util as MPIUtil
from pybullet_envs.deep_mimic.env.env import Env
from custom_reward import getRewardCustom
from pybullet_envs.deep_mimic.env.action_space import ActionSpace
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv

from pybullet_utils.arg_parser import ArgParser
from pybullet_utils.logger import Logger

import os
import json

save_name = ''


class RLWorld(object):

    def __init__(self, env, arg_parser):
        # TFUtil.disable_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        self.env = env
        self.arg_parser = arg_parser
        self._enable_training = True
        self.train_agents = []
        self.parse_args(arg_parser)

        self.build_agents()

        return

    def get_enable_training(self):
        return self._enable_training

    def set_enable_training(self, enable):
        self._enable_training = enable
        for i in range(len(self.agents)):
            curr_agent = self.agents[i]
            if curr_agent is not None:
                enable_curr_train = self.train_agents[i] if (len(self.train_agents) > 0) else True
                curr_agent.enable_training = self.enable_training and enable_curr_train

        if (self._enable_training):
            self.env.set_mode(RLAgent.Mode.TRAIN)
        else:
            self.env.set_mode(RLAgent.Mode.TEST)

        return

    enable_training = property(get_enable_training, set_enable_training)

    def parse_args(self, arg_parser):
        self.train_agents = self.arg_parser.parse_bools('train_agents')
        assert (len(self.train_agents) == 1 or len(self.train_agents) == 0)

        return

    def shutdown(self):
        self.env.shutdown()
        return

    def build_agents(self):
        self.agents = []

        Logger.print2('')
        Logger.print2('Num Agents: {:d}'.format(1))

        agent_files = self.arg_parser.parse_strings('agent_files')
        print("len(agent_files)=", len(agent_files))
        assert (len(agent_files) == 1 or len(agent_files) == 0)

        model_files = self.arg_parser.parse_strings('model_files')
        assert (len(model_files) == 1 or len(model_files) == 0)

        curr_file = agent_files[0]
        curr_agent = self._build_agent(0, curr_file)

        if curr_agent is not None:
            Logger.print2(str(curr_agent))

            if (len(model_files) > 0):
                curr_model_file = model_files[0]
                if curr_model_file != 'none':
                    curr_agent.load_model(os.getcwd() + "/" + curr_model_file)

        self.agents.append(curr_agent)
        Logger.print2('')

        self.set_enable_training(self.enable_training)

        return

    def update(self, timestep):
        self._update_env(timestep)

        # compute next state
        next_state = self.env._humanoid.getState()

        # compute reward
        kinPose = self.env._humanoid.computePose(self.env._humanoid._frameFraction)
        reward = getRewardCustom(kinPose, self.env._humanoid)

        # compute whether episode is done
        is_done = self.env.is_episode_end()
        return next_state, reward, is_done

    def reset(self):
        self._reset_env()
        return

    def _update_env(self, timestep):
        self.env.update(timestep)
        return

    def _reset_env(self):
        self.env.reset()
        return

    def _build_agent(self, id, agent_file):
        if (agent_file == 'none'):
            agent = None
        else:
            agent = None
            with open(os.getcwd() + "/" + agent_file) as data_file:
                json_data = json.load(data_file)

                agent = CustomAgent(self, id, json_data)

            assert (agent != None), 'Failed to build agent {:d} from: {}'.format(id, agent_file)
        return agent


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


class CustomAgent(RLAgent):
    def __init__(self, world, id, json_data, gamma=0.95):
        super().__init__(world, id, json_data)
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

        self.state_size = 197
        self.num_actions = 36

    def load_model(self, in_path):
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
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p), self.num_actions))
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


def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if arg_file == '':
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


def build_world(args, enable_draw):
    arg_parser = build_arg_parser(args)
    env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
    world = RLWorld(env, arg_parser)

    agent_files = os.getcwd() + "/" + arg_parser.parse_string("agent_files")

    with open(agent_files) as data_file:
        json_data = json.load(data_file)
        print("json_data=", json_data)

        # build the custom agent
        agent = CustomAgent(world, id, json_data)

        agent.set_enable_training(False)
        world.reset()

    return world


if __name__ == '__main__':
    args = sys.argv[1:]
    enable_draw = False
    world = build_world(args, enable_draw)

    env = world.env

    tf.random.set_seed(336699)
    ppo_agent = world.agents[0]
    ppo_steps = 4096
    mini_batches = 256
    ep_reward = []
    total_avgr = []
    target_reached = False
    best_reward = 0
    avg_rewards_list = []
    test_iter = 1

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

        for s in range(ppo_steps):
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
            prob = ppo_agent.actor(np.array([state]))
            probs.append(action)
            values.append(value[0][0])
            state = next_state

        value = ppo_agent.critic(np.array([state])).numpy()
        values.append(value[0][0])
        np.reshape(probs, (len(probs), ppo_agent.num_actions))
        probs = np.stack(probs, axis=0)

        returns, adv = advantage_estimation(rewards, dones, values,
                                                             ppo_agent.discount_factor, ppo_agent.gamma)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)

        ## Update the gradients
        print("Learning/ Updating Gradients")
        al, cl = ppo_agent.learn(states, actions, adv, probs, returns)

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
        file = open('rewards_log.csv','a')
        file.write(str(test_iter)+','+str(avg_reward))
        file.close()

        test_iter = test_iter + 1

    env.close()
