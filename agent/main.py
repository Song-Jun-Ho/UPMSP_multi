import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Flatten
from agent.VDN_agent import QAgent
from agent.VDN_net import VDN
from utils.memory import RandomMemory
from utils.policy import EpsGreedyQPolicy
import copy
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import pandas as pd

from collections import deque
from environment.Multi_Environment import ENVIRONMENT



class Duel_DRQN(tf.keras.Model):
    def __init__(self, action_size, input_shape, h_size):
        super(Duel_DRQN, self).__init__()
        self.h_size = h_size
        self.fc1 = Dense(64, activation='relu', input_shape=input_shape)
        self.fc2 = Dense(64, activation='relu', input_shape=input_shape)
        self.rnn = GRU(units=self.h_size, return_sequences=True, return_state=True)
        self.flatten = Flatten()

        self.fc4 = Dense(32, activation='relu')
        self.out = Dense(action_size)

        self.ADV = Dense(32, activation='relu')
        self.V = Dense(32, activation='relu')
        self.adv_out = Dense(action_size)
        self.val_out = Dense(1)

        self.initial_state = None
        self.initialize_state = None
        self.batch_size = None
        self.trace_len = None

    def call(self, obs, other_obs, batch_size, trace_len):
        self.batch_size = batch_size
        self.trace_len = trace_len
        obs_out = self.fc1(obs)
        other_obs_out = self.fc2(other_obs)
        concat_out = tf.concat([obs_out, other_obs_out], axis=-1)
        concat_out = tf.reshape(concat_out, [self.batch_size, self.trace_len, -1])
        rnn_out, h_state = self.rnn(concat_out, initial_state=self.initial_state)
        rnn_out = tf.reshape(rnn_out, shape=[-1, self.h_size])
        self.initial_state = h_state
        interim_out = self.flatten(rnn_out)

        interim_out = self.fc4(interim_out)
        q = self.out(interim_out)

        # adv = self.ADV(interim_out)
        # v = self.V(interim_out)
        #
        # adv = self.adv_out(adv)
        # v = self.val_out(v)
        # q = v + adv - tf.reduce_mean(adv, axis= -1, keepdims=True)

        return q


agent_num = 2
max_episodes = 50000
# train_start = 1000
a_size = 5
# a_size = 2
trajectory_len = 1
observation_dim = 106
num_sample_episodes = 32
trace_size = 8
h_size = 64
# observation_dim = 3

mixing_model_path = '../model/multi_agent/mixing-%d' % a_size
agent_net_model_path = '../model/multi_agent/agent-%d' % a_size
summary_path = '../summary/multi_agent/queue-%d' % a_size
result_path = '../result/multi_agent/queue-%d' % a_size
event_path = '../simulation/multi_agent/queue-%d' % a_size

if not os.path.exists(mixing_model_path):
    os.makedirs(mixing_model_path)

if not os.path.exists(agent_net_model_path):
    os.makedirs(agent_net_model_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)

if not os.path.exists(result_path):
    os.makedirs(result_path)

if not os.path.exists(event_path):
    os.makedirs(event_path)

env = ENVIRONMENT()
writer = tf.summary.create_file_writer(summary_path)

avg_max_q_list = []
reward_list = []
make_span_list = []
mean_weighted_tardiness_list = []
loss_list = []

memory = RandomMemory(limit=2000)
policy = EpsGreedyQPolicy()
learning_rate = 1e-4
# global_step = tf.Variable(0, trainable=False)
# decayed_lr = tf.compat.v1.train.exponential_decay(learning_rate, global_step, 100000, 0.95, staircase=True)

# agent_input_shape = (trajectory_len, observation_dim)
agent_input_shape = (observation_dim, )

model = Duel_DRQN(a_size, agent_input_shape, h_size=h_size)
target_model = Duel_DRQN(a_size, agent_input_shape, h_size=h_size)

mixing_model_path = '../model/multi_agent/mixing-%d' % a_size

agents = []
for aid in range(agent_num):
    # model_path = mixing_model_path + '_{0}'.format(aid)
    # model = Duel_DRQN(a_size, agent_input_shape, h_size=h_size)
    # model.load_weights(model_path)
    # target_model = Duel_DRQN(a_size, agent_input_shape, h_size=h_size)
    # target_model.load_weights(model_path)
    agent = QAgent(
        aid=aid,
        policy=policy,
        model=model,
        target_model=target_model, action_num=a_size)
    agent.target_model.set_weights(model.get_weights())
    init_state = tf.one_hot(0, observation_dim)
    agents.append(agent)

# initialize states
init_states = env.reset()


loss_fn = tf.keras.losses.MeanSquaredError()
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-4, rho=0.99)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
batch_size = 256

vdn = VDN(agents=agents, memory=memory, batch_size=batch_size, num_sample_episodes=num_sample_episodes,
            trace_size=trace_size, loss_fn=loss_fn, optimizer=optimizer, eps=1.0, eps_decay_rate=.99992, min_eps=.01)
episode_reward_history = []
loss_history = []
episode_reward_mean = 0
loss_mean = 0

# result = qmix.get_qmix_output()
# print(result)

for episode in range(max_episodes):
    done = False
    step = 0
    avg_loss = 0.0
    rewards = []
    dones = []
    episode_actions = []
    episode_max_actions = []
    episode_q_val = []
    episodeBuffer = []

    # initialize states and observations
    observation, state, dispatching_possible = env.reset()

    # print(init_states)
    # print(init_states.shape)
    # print(init_states[0])
    # print(init_states[1])
    # print(qmix.trainable_variables)
    for aid, agent in enumerate(agents):
        initial_one_hot = tf.one_hot(a_size - 1, depth=a_size)
        obs = [observation[aid], observation[1 - aid]]
        obs = np.array(obs)
        agent.reset(obs)
        # print(agent.model.trainable_variables)

    # initialize rnn hidden state
    rnn_state = {}
    for idx in range(agent_num):
        rnn_state[idx] = None
        vdn.agents[idx].model.initial_state = None
        vdn.agents[idx].target_model.initial_state = None

    vdn.batch_rnn_state = None

    while not done:
        step += 1
        actions = []
        prev_actions = []
        max_actions = []
        q_values = []
        for aid, (agent, dispatching) in enumerate(zip(agents, dispatching_possible)):
            prev_action = agent.prev_action
            vdn.agents[aid].model.initial_state = copy.deepcopy(rnn_state[aid])
            action, max_action, q_val = agent.act(dispatching, eps=vdn.eps, batch_size=1, trace_len=1)
            rnn_state[aid] = copy.deepcopy(vdn.agents[aid].model.initial_state)
            # print("TRAJECCT: ", agent.trajectory)
            actions.append(action)
            max_actions.append(max_action)
            q_values.append(q_val)
            prev_actions.append(prev_action)

        next_observation, next_state, reward, done, agent_done, current_agent_done, next_dispatching_possible = env.step(actions)
        rewards.append(reward)
        dones.append(done)
        episode_actions.append(actions)
        episode_max_actions.append(max_actions)
        episode_q_val.append(q_values)

        one_hot_actions = []
        prev_one_hot_actions = []
        for action in actions:
            action = tf.one_hot(action, depth=a_size)
            one_hot_actions.append(action)
        for prev_action in prev_actions:
            prev_action = tf.one_hot(prev_action, depth=a_size)
            prev_one_hot_actions.append(prev_action)

        # trajectory : observations history of each agents (partially observable)
        # state : global state of current time step (not history)
        next_trajectories = []
        trajectories = []
        for aid, agent in enumerate(agents):
            trajectory = agent.trajectory
            trajectories.append(trajectory)
            next_obs = [next_observation[aid], next_observation[1 - aid]]
            agent.observe(next_obs)
            next_trajectory = agent.trajectory
            next_trajectories.append(next_trajectory)

        episodeBuffer.append([state, trajectories, one_hot_actions, prev_one_hot_actions, reward, next_state, next_trajectories,
                    done, agent_done, current_agent_done, next_dispatching_possible])

        state = next_state
        dispatching_possible = next_dispatching_possible
        trajectories = next_trajectories

        # if step > batch_size:
        loss = vdn.train(num_sample_episodes, trace_size)
        loss_history.append(loss)
        avg_loss += loss

        if done:
            vdn.save(episodeBuffer)
            mean_weighted_tardiness = env.mean_weighted_tardiness
            make_span = env.make_span

            episode_reward = np.sum(rewards)
            episode_reward_history.append(episode_reward)

            mean_weighted_tardiness = env.mean_weighted_tardiness
            make_span = env.make_span

            with writer.as_default():
                tf.summary.scalar('Loss/Average Loss', avg_loss / float(step), step=episode)
                # tf.summary.scalar('Performance/Average Max Q', agent.avg_q_max / float(step), step=episode)
                tf.summary.scalar('Performance/Reward', episode_reward, step=episode)
                tf.summary.scalar("Perf/Mean weighted tardiness", mean_weighted_tardiness, step=episode)

            if episode % 250 == 0:
                for aid, agent in enumerate(vdn.agents):
                    agent.model.save_weights(mixing_model_path + '_{0}'.format(aid), save_format='tf')
                print("Saved Model at episode %d" % episode)

            # agent.avg_q_max, agent.avg_loss = 0, 0
            avg_loss = avg_loss / float(step)

            print("episode: {:4d} | score_avg: {:5.4f} | memory_length: {:4d} | epsilon: {:.4f} | loss_avg: {:5.5f}".format(
                episode, episode_reward, len(vdn.memory.experiences), vdn.eps, avg_loss
            ))
            print(episode_actions)
            print(episode_max_actions)
            print(episode_q_val)
            # print(dones)

            avg_loss = 0.0
