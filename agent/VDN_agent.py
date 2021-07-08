import tensorflow as tf
from collections import deque, namedtuple
import numpy as np
from agent.VDN_net import *

'''
    agent_net 및 target agent_net 의 get_action 등 여러 내장함수가 있는 class
'''


class QAgent:
    def __init__(self, aid, policy, model, target_model, action_num, timesteps=1):
        self.aid = aid
        self.policy = policy
        self.model = model
        self.target_model = target_model
        self.prev_action = action_num - 1
        self.recent_action_id = [0, 0]
        self.timesteps = timesteps
        self.trajectory = None
        self.last_q_values = None
        self.update_interval = 200   # episode 기준
        self.action_num = action_num

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
            action_id = self.policy.select_action(q_values=q_values[:-1], eps=eps, is_training=True)
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
        # inputs_1 = tf.expand_dims(inputs_1, 0)
        # inputs_2 = tf.expand_dims(inputs_2, 0)
        q_values = self.model(inputs_1, inputs_2, batch_size, trace_len)
        return q_values[0]

    def _hard_update_target_model(self):
        """ for hard update """
        self.target_model.set_weights(self.model.get_weights())

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights(), dtype=object)
        model_weights = np.array(self.model.get_weights(), dtype=object)
        new_weight = (1. - self.update_interval) * target_model_weights + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)

    def reset(self, observation):
        self.observation = observation
        self.prev_observation = observation
        self.trajectory = self._init_deque(observation)
