import random
from collections import deque, namedtuple
import numpy as np

Experience = namedtuple('Experience', 'state0, observations0, action, reward, observations1, state1, terminal')

def sample_batch_indexes(low, high, size):
    r = range(low, high)
    batch_idxs = random.sample(r, size)

    return batch_idxs

class Memory(object):
    def sample(self, **kwargs):
        raise NotImplementedError()

    def append(self, **kwargs):
        raise NotImplementedError()

class RandomMemory(Memory):
    def __init__(self, limit, agent_num=2):
        super(Memory, self).__init__()
        self.experiences = deque(maxlen=limit)
        self.agent_num = agent_num

    def sample(self, batch_size, trace_length):
        assert batch_size > 1, "batch_size must be positive integer"
        batch_size = min(batch_size, len(self.experiences))
        sampled_episodes = random.sample(self.experiences, batch_size)
        # mini_batch = random.sample(self.experiences, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces, dtype=object)
        sampledTraces = np.reshape(sampledTraces, [batch_size * trace_length, -1])
        state_batch = []
        observation_batch = [[] for _ in range(self.agent_num)]
        action_batch = []
        prev_action_batch = []
        reward_batch = []
        next_state_batch = []
        next_observation_batch = [[] for _ in range(self.agent_num)]
        terminal_batch = []
        dispatching_batch = [[] for _ in range(self.agent_num)]
        agent_done_batch = [[] for _ in range(self.agent_num)]
        current_agent_done_batch = [[] for _ in range(self.agent_num)]
        for trace in sampledTraces:
            state_batch.append(trace[0])
            for i in range(self.agent_num):
                observation_batch[i].append(trace[1][i][0])
                next_observation_batch[i].append(trace[6][i][0])
                dispatching_batch[i].append(trace[10][i])
                agent_done_batch[i].append(trace[8][i])
                current_agent_done_batch[i].append(trace[9][i])
            action_batch.append(trace[2])
            prev_action_batch.append(trace[3])
            reward_batch.append(trace[4])
            next_state_batch.append(trace[5])
            terminal_batch.append(1. if trace[7] else 0.)
        state_batch = np.array(state_batch)
        observation_batch = np.array(observation_batch)
        action_batch = np.array(action_batch)
        prev_action_batch = np.array(prev_action_batch)
        reward_batch = np.array(reward_batch)
        next_observation_batch = np.array(next_observation_batch)
        dispatching_batch = np.array(dispatching_batch)
        agent_done_batch = np.array(agent_done_batch)
        current_agent_done_batch = np.array(current_agent_done_batch)
        next_state_batch = np.array(next_state_batch)
        terminal_batch = np.array(terminal_batch)

        assert len(state_batch) == batch_size * trace_length

        return state_batch, observation_batch, action_batch, prev_action_batch, reward_batch, next_state_batch, next_observation_batch, terminal_batch, \
               agent_done_batch, current_agent_done_batch, dispatching_batch

    def append(self, episodeBuffer):
        self.experiences.append(episodeBuffer)
