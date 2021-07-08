import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Flatten
from agent.qmix_mixing_net import QMix
from utils.memory import RandomMemory
from utils.policy import EpsGreedyQPolicy
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from collections import deque
from environment.Multi_Environment import ENVIRONMENT


class Duel_DRQN(tf.keras.Model):
    def __init__(self, action_size, input_shape, h_size):
        super(Duel_DRQN, self).__init__()
        self.h_size = h_size
        self.fc1 = Dense(128, activation='relu', input_shape=input_shape)
        self.fc2 = Dense(128, activation='relu', input_shape=input_shape)
        self.fc3 = Dense(64, activation='relu')
        self.rnn = GRU(units=self.h_size, return_sequences=True, return_state=True)
        self.flatten = Flatten()

        self.fc4 = Dense(32, activation='relu')
        self.fc5 = Dense(16, activation='relu')
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
        # concat = np.concatenate((obs, other_obs), axis=-1)
        concat_out = tf.concat([obs_out, other_obs_out], axis=-1)
        # concat_out = self.fc2(concat)
        concat_out = self.fc3(concat_out)
        concat_out = tf.reshape(concat_out, [self.batch_size, self.trace_len, -1])
        rnn_out, h_state = self.rnn(concat_out, initial_state=self.initial_state)
        rnn_out = tf.reshape(rnn_out, shape=[-1, self.h_size])
        self.initial_state = h_state
        interim_out = self.flatten(rnn_out)

        interim_out = self.fc4(interim_out)
        interim_out = self.fc5(interim_out)
        q = self.out(interim_out)

        # adv = self.ADV(interim_out)
        # v = self.V(interim_out)
        #
        # adv = self.adv_out(adv)
        # v = self.val_out(v)
        # q = v + adv - tf.reduce_mean(adv, axis= -1, keepdims=True)

        return q


agent_num = 2
max_episodes = 10
a_size = 5
trajectory_len = 1
observation_dim = 106
num_sample_episodes = 4
trace_size = 8
h_size = 64
agent_input_shape = (observation_dim, )

mixing_model_path = '../model/multi_agent/mixing-%d' % a_size

class QAgent:
    def __init__(self, aid, policy, model, action_num, mixing_model_path, timesteps=1):
        self.aid = aid
        self.policy = policy
        self.model = model
        self.prev_action = 0
        self.recent_action_id = [0, 0]
        self.timesteps = timesteps
        self.trajectory = None
        self.last_q_values = None
        self.action_num = action_num
        self.model_path = mixing_model_path + '_{0}'.format(self.aid)
        self.model.load_weights(self.model_path)

    def _init_deque(self, observation):
        trajectory = deque(maxlen=self.timesteps)
        for i in range(self.timesteps):
            trajectory.append(observation)
        return trajectory

    def act(self, dispatching, eps, batch_size, trace_len):
        action, max_action, q_values = self._forward(dispatching, eps, batch_size, trace_len)
        self.prev_action = action
        return action, max_action, q_values

    def observe(self, observation):
        self.state = observation
        self.trajectory.append(observation)

    def _forward(self, dispatching, eps, batch_size, trace_len):
        q_values = self._compute_q_values(self.trajectory, self.recent_action_id, batch_size, trace_len)
        self.last_q_values = q_values
        if dispatching:
            action_id = self.policy.select_action(q_values=q_values[:-1], eps=eps, is_training=False)
            max_action = np.argmax(q_values[:-1])
            q_values = np.max(q_values[:-1])
        else:
            action_id = self.action_num - 1
            max_action = self.action_num - 1
            q_values = q_values[-1]

        return action_id, max_action, q_values

    def _compute_q_values(self, trajectory, prev_action_is, batch_size, trace_len):
        prev_masks = []
        # other_prev_mask = tf.one_hot(other_prev_action, depth=self.action_num)
        # other_prev_mask = np.expand_dims(other_prev_mask, axis=0)
        traject_1 = np.array(trajectory)[:, 0, :]
        traject_2 = np.array(trajectory)[:, 1, :]
        inputs_1 = tf.convert_to_tensor(list(traject_1))
        inputs_2 = tf.convert_to_tensor(list(traject_2))
        q_values = self.model(inputs_1, inputs_2, batch_size, trace_len)
        return q_values[0]

    def reset(self, observation):
        self.observation = observation
        self.prev_observation = observation
        self.trajectory = self._init_deque(observation)


policy = EpsGreedyQPolicy()
model = Duel_DRQN(a_size, agent_input_shape, h_size=h_size)
agents = []
for aid in range(agent_num):
    agent = QAgent(
        aid=aid,
        policy=policy,
        model=model, action_num=a_size, mixing_model_path=mixing_model_path)
    agents.append(agent)

# env = ENVIRONMENT()

mean_weighted_tardiness_list = []
for episode in range(max_episodes):
    # np.random.seed(episode)
    env = ENVIRONMENT()
    done = False
    rewards = []
    episode_actions = []
    episode_max_actions = []
    episode_q_val = []
    # initialize states and observations
    observation, state, dispatching_possible = env.reset()

    for aid, agent in enumerate(agents):
        obs = [observation[aid], observation[1-aid]]
        obs = np.array(obs)
        agent.reset(obs)
        # print(agent.model.trainable_variables)

    # initialize rnn hidden state
    rnn_state = {}
    for idx in range(agent_num):
        rnn_state[idx] = None
        agents[idx].model.initial_state = copy.deepcopy(rnn_state[idx])

    while not done:
        actions = []
        # actions = [0, 0]
        prev_actions = []
        max_actions = []
        q_values = []
        for aid, (agent, dispatching) in enumerate(zip(agents, dispatching_possible)):
            prev_action = agent.prev_action
            agents[aid].model.initial_state = copy.deepcopy(rnn_state[aid])
            action, max_action, q_val = agent.act(dispatching, eps=0.0, batch_size=1, trace_len=1)
            rnn_state[aid] = copy.deepcopy(agents[aid].model.initial_state)

            actions.append(action)
            max_actions.append(max_action)
            q_values.append(q_val)
            prev_actions.append(prev_action)

        next_observation, next_state, reward, done, agent_done, current_agent_done, next_dispatching_possible = env.step(actions)
        rewards.append(reward)
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
            next_obs = np.array([next_observation[aid], next_observation[1-aid]])
            agent.observe(next_obs)
            next_trajectory = agent.trajectory
            next_trajectories.append(next_trajectory)

        state = next_state
        dispatching_possible = next_dispatching_possible
        trajectories = next_trajectories

        if done:
            mean_weighted_tardiness = env.mean_weighted_tardiness
            make_span = env.make_span

            episode_reward = np.sum(rewards)

            mean_weighted_tardiness = env.mean_weighted_tardiness
            mean_weighted_tardiness_list.append(mean_weighted_tardiness)

            print("episode: {:4d} | score_avg: {:5.4f} | mean_weighted_tardiness: {:5.5f}".format(
                episode, episode_reward, mean_weighted_tardiness
            ))
            print(episode_actions)
            # print(episode_max_actions)
            # print(episode_q_val)

mean_weighted_tardiness_list = np.array(mean_weighted_tardiness_list)
avg_mean_weighted_tardiness = np.average(mean_weighted_tardiness_list)
std_mean_weighred_tardiness = np.std(mean_weighted_tardiness_list)
print('Average mean weighted tardiness : ', avg_mean_weighted_tardiness)
print('Std mean weighted tardiness : ', std_mean_weighred_tardiness)


# (RL, RL) : Average mean weighted tardiness :  11.765802524748478, Std mean weighted tardiness :  7.541663678867745
# (0, 0) : Average mean weighted tardiness :  9.995724746083459
# Std mean weighted tardiness :  6.591132448904124

# (2, 0)