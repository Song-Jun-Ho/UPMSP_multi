import tensorflow as tf
from tensorflow.keras.layers import Dense
import copy
import numpy as np
import copy



class MixingNet():
    def __init__(self, agent_nets, embed_shape):
        self.agent_nets = agent_nets
        self.agent_num = len(agent_nets)
        self.embed_shape = embed_shape
        self.timesteps = 1
        self.agent_output_dim = 5

    def call(self, inputs):
        agents_obs = inputs[0]
        prev_masks = inputs[1]
        # agents_inputs = np.concatenate((agent_obs, agent_prev_action), axis=-1)
        # agents_inputs = inputs[0]
        states = inputs[2]
        masks = inputs[3]
        batch_size = states.shape[0]
        # print(states.shape)
        states = np.reshape(states, (batch_size, 1, -1))

        agents_outputs = []
        # prev_masks = np.reshape(prev_masks, [-1, self.agent_output_dim])
        # prev_masks = np.expand_dims(prev_masks, axis=1)
        for agent_net, agent_obs, mask in zip(self.agent_nets, agents_obs, masks):
            # agent_input = np.concatenate((agent_obs, prev_masks), axis=-1)
            agent_out = agent_net(agent_obs[:, :, 0, :], agent_obs[:, :, 1, :])
            agent_out = tf.multiply(agent_out, mask)
            agent_out = tf.reduce_sum(agent_out, axis=-1, keepdims=True)
            # print(agent_out.shape)
            agents_outputs.append(agent_out)

        # agents_outputs = np.array(agents_outputs)
        # q_tot = np.sum(agents_outputs)
        agents_outputs = tf.concat(agents_outputs, 1)
        y = tf.reduce_sum(agents_outputs, axis=-1, keepdims=True)
        # agents_outputs = tf.expand_dims(agents_outputs, 1)

        q_tot = tf.reshape(y, [-1, 1])
        return q_tot

class VDN:
    def __init__(self, agents=None, memory=None, gamma=0.99, batch_size=100, num_sample_episodes=4,
                 trace_size=8, loss_fn=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.RMSprop(), is_ddqn=False, update_interval=0.001, embed_shape=60, lr=1e-5,
                 agent_action_num=5, eps=1., eps_decay_rate=0.999, min_eps=0.01):
        self.agents = agents
        self.memory = memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_sample_episodes = num_sample_episodes
        self.trace_size = trace_size
        self.is_ddqn = is_ddqn
        self.update_interval = update_interval
        self.step = 0
        self.train_interval = 1
        self.warmup_steps = 50
        self.prev_state = None
        self.prev_observations = None
        self.agent_action_num = agent_action_num
        self.last_q_values = [0]
        self.last_targets = [0]
        self.eps = eps
        self.eps_decay_rate = eps_decay_rate
        self.min_eps = min_eps

        models = []
        target_models = []
        self.trainable_variables = None
        self.target_trainable_variables = None

        # rnn hidden state for training
        self.batch_rnn_state = None

        # trainable variables of network and target network of agent_net
        for agent in self.agents:
            models.append(agent.model)
            target_models.append(agent.target_model)
            if self.trainable_variables is None:
                self.trainable_variables = agent.model.trainable_variables
                self.target_trainable_variables = agent.target_model.trainable_variables
            else:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

        # network and target network of MixingNet
        self.model = MixingNet(models, embed_shape)
        self.target_model = MixingNet(target_models, embed_shape)
        # trainable variables of network and target network of MixingNet
        # self.trainable_variables += self.model.trainable_variables
        # self.target_trainable_variables += self.target_model.trainable_variables

        self.loss_fn = loss_fn
        self.optimizer = optimizer

    # save sample in replay memory
    # state : 모든
    def save(self, episodeBuffer):
        self.memory.append(episodeBuffer)
        self.step += 1

    def train(self, batch_size, trace_len):
        self.agents[0].model.initial_state = None
        self.agents[0].target_model.initial_state = None
        loss = self._experience_replay(batch_size, trace_len)
        self.batch_rnn_state = None
        return loss

    def _experience_replay(self, batch_size, trace_len):
        loss = 0
        if self.step > self.warmup_steps and self.step % self.train_interval == 0:
            self.eps = self.eps * self.eps_decay_rate
            if self.eps < self.min_eps:
                self.eps = self.min_eps
            states, observations, actions, prev_actions, rewards, next_states, next_observations, terminals, \
            agent_dones, current_agent_dones, next_dispatchings = self.memory.sample(self.num_sample_episodes, self.trace_size)

            rewards = np.array(rewards).reshape(-1, 1)
            terminals = np.array(terminals).reshape(-1, 1)
            next_observations = np.array(next_observations)
            next_states = np.array(next_states)

            masks, target_masks = [], []
            prev_masks = []
            agent_actions = np.reshape(actions, [-1, self.agent_action_num*2])
            agent_actions = np.expand_dims(agent_actions, axis=1)
            for idx, (agent, next_observation, next_dispatching) in enumerate(zip(self.agents, next_observations, next_dispatchings)):
                agent.target_model.initial_state = None
                agent_outs = agent.target_model(next_observation[:, 0, :], next_observation[:, 1, :],
                                                batch_size=batch_size, trace_len=trace_len)
                # agent_out = agent.target_model(next_observation)
                agent_outs = np.array(agent_outs).reshape(self.batch_size, -1)
                argmax_actions = []
                # agent_out = agent.target_model(next_observation)
                for i, (next_dsp, agent_out) in enumerate(zip(next_dispatching, agent_outs)):
                    if next_dsp:
                        argmax_actions.append(np.argmax(agent_out[:-1]))
                    else:
                        argmax_actions.append(self.agent_action_num - 1)
                target_mask = tf.one_hot(argmax_actions, depth=self.agent_action_num)
                target_masks.append(target_mask)  # idx 번째 agent의 one-hot actions (target agent_net)
                masks.append(actions[:, idx, :])  # idx 번째 agent의 one-hot actions (agent_net)
                prev_masks.append(prev_actions[:, idx, :])  # idx 번째 agent의 one-hot prev actions (agent_net)

            masks = np.array(masks)
            prev_masks = np.array(prev_masks)
            target_masks = tf.convert_to_tensor(target_masks)   # one hot actions of target agent_nets

            target_q_values = self._predict_on_batch(next_states, next_observations, masks, target_masks,
                                                     agent_dones, self.target_model, batch_size, trace_len)
            masks = tf.convert_to_tensor(masks)  # one hot actions of agent_nets
            discounted_reward_batch = self.gamma * target_q_values * (1 - terminals)
            targets = rewards + discounted_reward_batch

            observations = np.array(observations)
            states = np.array(states)
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            observations = tf.convert_to_tensor(observations, dtype=tf.float32)

            loss = self._train_on_batch(states, observations, prev_masks, masks, targets, current_agent_dones,
                                        batch_size, trace_len)

            if self.update_interval > 1:
                # hard update
                self._hard_update_target_model()
            else:
                # soft update
                self._soft_update_target_model()
        # self.step += 1

        return loss


    def _predict_on_batch(self, states, observations, prev_masks, masks, agent_dones, model, batch_size, trace_len):
        # q_values = model.call([observations, prev_masks, states, masks])  # object of MixingNet
        agents_outputs = []

        # initialize target network rnn hidden states
        for agent_net, agent_obs, mask, agent_done in zip(self.agents, observations, masks, agent_dones):
            agent_net.target_model.initial_state = None
            agent_out = agent_net.target_model(agent_obs[:, 0, :], agent_obs[:, 1, :],
                                               batch_size=batch_size, trace_len=trace_len)
            agent_out = tf.multiply(agent_out, mask)
            agent_out = tf.reduce_sum(agent_out, axis=-1, keepdims=True)
            # agent_out = agent_out * (1 - agent_done)
            # print(agent_out.shape)
            agents_outputs.append(agent_out)

        # agents_outputs = np.array(agents_outputs)
        # q_tot = np.sum(agents_outputs)
        agents_outputs = tf.concat(agents_outputs, 1)
        y = tf.reduce_sum(agents_outputs, axis=-1, keepdims=True)
        # agents_outputs = tf.expand_dims(agents_outputs, 1)

        q_values = tf.reshape(y, [-1, 1])
        return q_values  # q_tot

    def _train_on_batch(self, states, observations, prev_masks, masks, targets, agent_dones, batch_size, trace_len):
        # for agent in self.agents:
        with tf.GradientTape() as tape:
            # tape.watch(observations)
            # tape.watch(states)
            targets = tf.stop_gradient(targets)
            agents_outputs = []
            for agent_net, agent_obs, mask, agent_done in zip(self.agents, observations, masks, agent_dones):
                agent_net.model.initial_state = None
                agent_out = agent_net.model(agent_obs[:, 0, :], agent_obs[:, 1, :],
                                            batch_size=batch_size, trace_len=trace_len)
                agent_out = tf.multiply(agent_out, mask)
                agent_out = tf.reduce_sum(agent_out, axis=-1, keepdims=True)
                # agent_out = agent_out * (1 - agent_done)
                # print(agent_out.shape)
                agents_outputs.append(agent_out)

            # agents_outputs = np.array(agents_outputs)
            # q_tot = np.sum(agents_outputs)
            agents_outputs = tf.concat(agents_outputs, 1)
            y = tf.reduce_sum(agents_outputs, axis=-1, keepdims=True)
            # agents_outputs = tf.expand_dims(agents_outputs, 1)

            y_preds = tf.reshape(y, [-1, 1])

            loss_value = self.loss_fn(targets, y_preds)
            # error = tf.abs(targets - y_preds)
            # quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            # linear_part = error - quadratic_part
            # loss_value = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        self.last_q_values = y_preds
        self.last_targets = targets
        # print('y_pred : ', y_preds)
        # print('targets : ', targets)
        # print('error : ', (targets - y_preds)/targets * 100)

        grads = tape.gradient(loss_value, self.agents[0].model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 10.)
        # print(y_preds)

        self.optimizer.apply_gradients(zip(grads, self.agents[0].model.trainable_variables))
        # print(len(self.trainable_variables))

        return loss_value.numpy()

    def _hard_update_target_model(self):
        """for hard update"""
        if self.step != 0 and self.step % self.update_interval == 0:
            # update target agent_net
            for agent in self.agents:
                agent._hard_update_target_model()

    def _soft_update_target_model(self):
        # update target agent_net
        for agent in self.agents:
            agent._soft_update_target_model()

    def get_qmix_output(self):
        """
            for debug
        """
        obs = np.array([[[[0., 0., 1.]]], [[[0., 0., 1.]]]])
        st = np.array([[0., 0., 1.]])
        mk = np.array([[[1., 0.]], [[1., 0.]]])

        obs = tf.convert_to_tensor(obs, dtype=np.float32)
        st = tf.convert_to_tensor(st, dtype=np.float32)
        mk = tf.convert_to_tensor(mk, dtype=np.float32)

        result = {}
        result[(0, 0)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[1., 0.]], [[0., 1.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(0, 1)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[0., 1.]], [[1., 0.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(1, 0)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[0., 1.]], [[0., 1.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(1, 1)] = round(self.model([obs, st, mk]).numpy()[0][0], 2)

        return result