import simpy
import random
import os
import functools

import numpy as np

from environment.RL_SimComponent import *


class ENVIRONMENT(object):
    def __init__(self):
        self.job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # self.weight = np.random.uniform(0, 5, len(self.job_type))
        self.weight = [0.6824156, 1.94436401, 4.16835236, 4.98060503, 1.3044099, 1.1492061,
                       1.91089353, 3.99533121, 3.9909476, 4.12363243]

        self.machine_num = {'BH': 8, 'LH': 8}

        # job type 별 average process time
        # self.p_ij = {'BH': np.random.uniform(1/60, 20/60, size=(len(self.job_type), self.machine_num['BH'])),
        #              'LH': np.random.uniform(5/60, 25/60, size=(len(self.job_type), self.machine_num['LH']))}
        self.p_ij = {'BH': [[0.0475769 , 0.28345901, 0.23134256, 0.02536312, 0.02895502,
        0.22328141, 0.03726922, 0.32319657],
       [0.2399417 , 0.2400061 , 0.28399284, 0.11620812, 0.19142107,
        0.18470976, 0.2019121 , 0.0640591 ],
       [0.31379431, 0.11245655, 0.1641372 , 0.18922426, 0.23837775,
        0.14602297, 0.1060729 , 0.12615988],
       [0.09044639, 0.17608172, 0.06488684, 0.25332603, 0.24005649,
        0.12337443, 0.16461656, 0.32751977],
       [0.31979514, 0.30126702, 0.20464691, 0.08268538, 0.11863086,
        0.24570244, 0.07532015, 0.06204494],
       [0.3299028 , 0.09503863, 0.15045407, 0.25428424, 0.0817669 ,
        0.20346872, 0.28080275, 0.06727106],
       [0.19124216, 0.09208689, 0.19362802, 0.19767289, 0.09402526,
        0.30067483, 0.22299395, 0.2564082 ],
       [0.01874433, 0.10641512, 0.28838857, 0.11334244, 0.09541337,
        0.24090806, 0.33288681, 0.2597504 ],
       [0.22806796, 0.07527788, 0.03417118, 0.05092743, 0.32389313,
        0.28833886, 0.3333021 , 0.18983926],
       [0.09192716, 0.23528543, 0.04003422, 0.18723643, 0.11708069,
        0.30529505, 0.29345386, 0.10489848]], 'LH': [[0.22310386, 0.26439182, 0.18219778, 0.30787039, 0.32009981,
        0.29492099, 0.16389799, 0.28358519],
       [0.28993501, 0.31026674, 0.25133252, 0.32543637, 0.29496944,
        0.28710504, 0.30584589, 0.26024802],
       [0.20178157, 0.32249392, 0.31492638, 0.20560182, 0.2745658 ,
        0.20606701, 0.13362029, 0.26625894],
       [0.14101235, 0.16772292, 0.08691617, 0.32414129, 0.14040794,
        0.33255302, 0.13205028, 0.30216985],
       [0.15979066, 0.1349698 , 0.30910287, 0.17579561, 0.15133105,
        0.15875274, 0.15735525, 0.28021894],
       [0.13955715, 0.2135363 , 0.161433  , 0.29666641, 0.15633622,
        0.29680304, 0.29696446, 0.30312013],
       [0.27140286, 0.13508065, 0.13993424, 0.09932373, 0.21391676,
        0.29912141, 0.28444951, 0.19424855],
       [0.2423934 , 0.14828647, 0.24096303, 0.17937817, 0.30241866,
        0.09811648, 0.13997243, 0.23747486],
       [0.22326694, 0.25650319, 0.10351381, 0.11284559, 0.15327647,
        0.1971764 , 0.1333118 , 0.25920199],
       [0.08389998, 0.28465521, 0.27098741, 0.15743965, 0.25035588,
        0.23028704, 0.25868046, 0.1307697 ]]}

        self.p_j = {'BH': np.average(self.p_ij['BH'], axis=1), 'LH': np.average(self.p_ij['LH'], axis=1)}
        self.process_list = ['BH', 'LH', 'Sink']
        self.process_all = ['BH', 'LH']
        self.priority = {'BH': [1, 2, 3, 4, 5, 6, 7, 8], 'LH': [1, 2, 3, 4, 5, 6, 7, 8]}
        self.part_num = 60

        self.arrival_rate = self.machine_num['LH'] / np.average(self.p_j['LH'])
        self.iat = len(self.job_type)*0.5 / self.arrival_rate
        # self.c_e = 1    # variability coefficient
        # self.iat_list = np.random.gamma(shape=1 / pow(self.c_e, 2), scale=pow(self.c_e, 2) * self.iat, size=self.part_num)
        self.iat_list = np.random.exponential(scale=self.iat, size=[len(self.job_type), self.part_num])

        self.K = {'BH': 1, 'LH': 1}
        self.global_step = 0

        self.env, self.model, self.source = self._modeling()
        self.done = False

        self.mean_weighted_tardiness = 0
        self.make_span = 0


    def _modeling(self):
        # Make model
        env = simpy.Environment()
        model = dict()

        self.iat_list = np.random.exponential(scale=self.iat, size=[len(self.job_type), self.part_num])

        source = Source(env, self.iat, self.iat_list, self.weight, self.job_type, self.p_ij, model,
                        self.process_list, self.machine_num, self.part_num, self.K)

        for i in range(len(self.process_all) + 1):
            if i == len(self.process_all):
                model['Sink'] = Sink(env)
            else:
                model[self.process_all[i]] = Process(env, self.process_all[i], self.machine_num[self.process_all[i]],
                                                     self.priority[self.process_all[i]],
                                                     model, self.process_list)

        self.global_step += 1

        return env, model, source

    # Decision time step : parts_arriving_event 또는 idle_machine_event 발생 시기로 한다.
    def step(self, actions):
        # Decision time step : if there is any process that satisfies routing possible condition
        # step until no more process left that satisfies routing possible condition
        self.done = False
        agent_done = [False, False]
        current_agent_done = [False, False]
        dispatching_possible = []
        next_dispatching_possible = []
        reward = 0.0
        next_observations = []
        next_global_states = []

        for i, process in enumerate(self.process_all):
            if (self.model[process].parts_routed == self.part_num) and (len(self.model[process].machine_store.items)
                                                                        == self.model[process].machine_num):
                current_agent_done[i] = True

        for i, process in enumerate(self.process_all):
            action = actions[i]
            if len(self.model[process].buffer_to_machine.items) != 0 and len(
                    self.model[process].machine_store.items) != 0:
                self.model[process].action = action
                dispatching_possible.append(True)
            else:
                self.model[process].action = action
                dispatching_possible.append(False)
            #     self.model[process].action = 4 # action of doing nothing

        current_time_step = self.env.now
        parts_in_buffer = {}
        parts_in_machine = {}
        idle_machine = {}
        for process in self.process_all:
            parts_in_buffer[process] = self.model[process].buffer_to_machine.items[:]
            parts_in_machine[process] = []
            idle_machine[process] = self.model[process].machine_store.items[:]
            for i, machine in enumerate(self.model[process].machines):
                if len(machine.part_in_machine) != 0:
                    parts_in_machine[process].append(machine.part_in_machine[0])

        current_new_idles = {}
        current_new_arrivals = {}
        for process in self.process_all:
            current_new_idles[process] = self.model[process].new_idles[:]
            current_new_arrivals[process] = self.model[process].new_arrivals[:]

        while True:
            # Go to next time step
            # any new arrival event or new idle machine event
            new_conditions = []
            for process in self.process_all:
                new_conditions.append(len(current_new_arrivals[process]) != len(self.model[process].new_arrivals)
                                      or len(current_new_idles[process]) != len(self.model[process].new_idles))
            new_time_step_possible = any(new_conditions)  # any and all
            if new_time_step_possible:
                break

            # 중간에 시뮬레이션이 종료되는 경우 break
            if self.model['LH'].parts_routed == self.part_num:
                self.done = True
                self.env.run()
                break

            self.env.step()

        if self.model['LH'].parts_routed == self.part_num:
            self.done = True

        if self.done:
            self.env.run()

        for i, process in enumerate(self.process_all):
            if (self.model[process].parts_routed == self.part_num) and (len(self.model[process].machine_store.items)
                                                                        == self.model[process].machine_num):
                agent_done[i] = True

        for i, process in enumerate(self.process_all):
            if len(self.model[process].buffer_to_machine.items) != 0 and len(
                    self.model[process].machine_store.items) != 0:
                next_dispatching_possible.append(True)
            else:
                next_dispatching_possible.append(False)

        next_time_step = self.env.now

        for i, (process, action, dsp_poss) in enumerate(zip(self.process_all, actions, dispatching_possible)):
            next_observation, next_global_state = self._get_state(process, i)
            next_observations.append(next_observation)
            if i == len(self.process_all) - 1:
                next_global_states.append(next_global_state)
            reward += self._calculate_reward(process, current_time_step, next_time_step, dsp_poss, action,
                                                parts_in_machine[process],
                                                parts_in_buffer[process])

        next_observations = np.array(next_observations)
        next_global_states = np.array(next_global_states)

        return next_observations, next_global_states, reward, self.done, agent_done, current_agent_done, next_dispatching_possible

    def _get_state(self, process, agent_idx):
        # f_1 (feature 1)
        # f_2 (feature 2)
        # f_3 (feature 3)
        # f_4 (feature 4)
        # aid = np.array([i])
        f_0 = np.zeros(2)

        # if len(self.model[process].buffer_to_machine.items) != 0 and len(self.model[process].machine_store.items) != 0:
        #     f_0[0] = 1      # dispatching possible
        # else:
        #     f_0[1] = 1      # dispatching not possible

        f_0[agent_idx] = 1

        f_1 = np.zeros(len(self.job_type))
        NJ = np.zeros(len(self.job_type))

        f_2 = np.zeros(len(self.model['BH'].machines))

        z = np.zeros(len(self.model['BH'].machines))
        f_3 = np.zeros(len(self.model['BH'].machines))

        f_4 = np.zeros(len(self.model['BH'].machines))

        machine_list = []
        # for process in self.process_all:
        for i, machine in enumerate(self.model[process].machines):
            machine_list.append(machine)
        # for i, machine in enumerate(self.model[process].machines):
        #     machine_list.append(machine)

        for i, machine in enumerate(machine_list):
            if len(machine.part_in_machine) != 0:  # if the machine is not idle(working)
                # print(machine.machine.items)
                step = machine.part_in_machine[0].step

                # feature 2
                f_2[i] = machine.part_in_machine[0].type / len(self.job_type)

                # feature 4
                f_4[i] = (machine.part_in_machine[0].due_date[process] - self.env.now) / machine.part_in_machine[0].process_time[
                             machine.process_name]
                # z_i : remaining process time of part in machine i
                if machine.part_in_machine[0].real_proc_time > self.env.now - machine.start_work:
                    z[i] = machine.part_in_machine[0].real_proc_time - (self.env.now - machine.start_work)
                    # feature 3
                    f_3[i] = z[i] / machine.part_in_machine[0].process_time[machine.process_name]

        # features to represent the tightness of due date allowance of the waiting jobs
        # f_5 (feature 5)
        # f_6 (feature 6)
        # f_7 (feature 7)
        # f_8 (feature 8)
        # f_9 (feature 9)
        f = [[] for _ in range(len(self.job_type))]
        f_5 = np.zeros(len(self.job_type))
        f_6 = np.zeros(len(self.job_type))
        f_7 = np.zeros(len(self.job_type))

        f_8 = np.zeros((len(self.job_type), 4))
        f_9 = np.zeros(len(self.job_type))
        # f_9 = np.zeros((len(self.job_type), 4))

        if len(self.model[process].buffer_to_machine.items) == 0:
            f_5 = np.zeros(len(self.job_type))
            f_6 = np.zeros(len(self.job_type))
            f_7 = np.zeros(len(self.job_type))

            f_8 = f_8.flatten()
            # f_9 = np.zeros(len(self.job_type)*4)

        else:
            # interval number indicating the tightness of the due date allowance
            g = np.zeros((len(self.job_type), 4))
            # interval number indicating the process time
            h = np.zeros((len(self.job_type), 4))

            for i, part in enumerate(self.model[process].buffer_to_machine.items):

                NJ[part.type] += 1
                f[part.type].append((part.due_date[process] - self.env.now) / part.process_time[part.process_list[part.step]])
                # print('hi ', part.due_date - self.env.now)
                # print(part.due_date)
                # print(self.env.now)
                # print('ho ',part.process_time)

                # case for interval number g
                if (part.due_date[process] - self.env.now) >= part.max_process_time[part.process_list[part.step]]:
                    g[part.type][0] += 1
                elif (part.due_date[process] - self.env.now) >= part.min_process_time[part.process_list[part.step]] \
                        and (part.due_date[process] - self.env.now) < part.max_process_time[part.process_list[part.step]]:
                    g[part.type][1] += 1
                elif (part.due_date[process] - self.env.now) >= 0 and (part.due_date[process] - self.env.now) < \
                        part.min_process_time[part.process_list[part.step]]:
                    g[part.type][2] += 1
                elif (part.due_date[process] - self.env.now) < 0:
                    g[part.type][3] += 1

            # feature 1
            f_1 = np.array([2 ** (-1 / nj) if nj > 0 else 0 for nj in NJ])

            # feature 8
            for j in self.job_type:
                for _g in range(4):
                    f_8[j][_g] = 2 ** (-1 / g[j][_g]) if g[j][_g] != 0 else 0
            f_8 = f_8.flatten()

            for i in range(len(self.job_type)):
                if len(f[i]) != 0:
                    min_tightness = np.min(np.array(f[i]))
                    max_tightness = np.max(np.array(f[i]))
                    avg_tightness = np.average(np.array(f[i]))
                else:
                    min_tightness = 0
                    max_tightness = 0
                    avg_tightness = 0

                # feature 5
                f_5[i] = min_tightness
                # feature 6
                f_6[i] = max_tightness
                # feature 7
                f_7[i] = avg_tightness

        # Calculating mean-weighted-tardiness
        # Calculating make span
        if len(self.model['Sink'].sink) != 0:
            mean_w_tardiness = 0
            for part in self.model['Sink'].sink:
                for process in self.process_all:
                    filter_list = list(filter(lambda x: x.type == part.type, self.model['Sink'].sink))
                    f_9[part.type] += part.weight * min(0, part.due_date['LH'] - part.completion_time[process]) / len(
                        filter_list)
                    mean_w_tardiness += part.weight * max(0, part.completion_time[process] - part.due_date[process])

            self.mean_weighted_tardiness = mean_w_tardiness / len(self.model['Sink'].sink)
        f_9 = f_9 / 100
        observation = np.concatenate((f_0, f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8), axis=None)
        global_state = f_9

        return observation, global_state

    def reset(self):
        self.env, self.model, self.source = self._modeling()
        self.done = False
        dispatching_possible = []

        parts_in_buffer = {}
        parts_in_machine = {}
        idle_machine = {}
        for process in self.process_all:
            parts_in_buffer[process] = self.model[process].buffer_to_machine.items[:]
            parts_in_machine[process] = []
            idle_machine[process] = self.model[process].machine_store.items[:]
            for i, machine in enumerate(self.model[process].machines):
                if len(machine.part_in_machine) != 0:
                    parts_in_machine[process].append(machine.part_in_machine[0])

        current_new_idles = {}
        current_new_arrivals = {}
        for process in self.process_all:
            current_new_idles[process] = self.model[process].new_idles[:]
            current_new_arrivals[process] = self.model[process].new_arrivals[:]

        while True:
            # Go to next time step
            # any new arrival event or new idle machine event
            new_conditions = []
            for process in self.process_all:
                new_conditions.append(len(current_new_arrivals[process]) != len(self.model[process].new_arrivals)
                                      or len(current_new_idles[process]) != len(self.model[process].new_idles))
            new_time_step_possible = any(new_conditions)  # any and all
            if new_time_step_possible:
                break

            self.env.step()

        next_global_states = []
        next_observations = []
        for i, process in enumerate(self.process_all):
            next_observation, next_global_state = self._get_state(process, i)
            if i == len(self.process_all) - 1:
                next_global_states.append(next_global_state)
            next_observations.append(next_observation)
        next_global_states = np.array(next_global_states)
        next_observations = np.array(next_observations)

        for i, process in enumerate(self.process_all):
            if len(self.model[process].buffer_to_machine.items) != 0 and len(
                    self.model[process].machine_store.items) != 0:
                dispatching_possible.append(True)
            else:
                dispatching_possible.append(False)

        return next_observations, next_global_states, dispatching_possible


    def _calculate_reward(self, process, current_time_step, next_time_step, dispatching_possible, action, parts_in_machine,
                          parts_in_buffer):
        # calculate reward for parts in waiting queue
        sum_reward_for_tardiness = 0
        sum_reward_for_makespan = 0
        total_weight_sum = 0

        for part in parts_in_buffer:
            if part.completion_time[process] == None:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - part.due_date[process])
            else:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (part.completion_time[process] - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * min(0, part.due_date[process] - part.completion_time[process])
        for part in parts_in_machine:
            if part.completion_time[process] == None:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (next_time_step - part.due_date[process])
            else:
                if part.due_date[process] < current_time_step:
                    sum_reward_for_tardiness += part.weight * (-1) * (part.completion_time[process] - current_time_step)
                elif part.due_date[process] >= current_time_step and part.due_date[process] < next_time_step:
                    sum_reward_for_tardiness += part.weight * min(0, part.due_date[process] - part.completion_time[process])

        # if dispatching_possible:
        #     if action == 4:
        #         sum_reward_for_tardiness += -50
        # else:
        #     if action != 4:
        #         sum_reward_for_tardiness += -50

        if len(self.model['Sink'].sink) != 0:
            mean_w_tardiness = 0
            make_span = self.model['Sink'].sink[0].completion_time['LH']
            for part in self.model['Sink'].sink:
                # mean_w_tardiness += part.weight * max(0, part.completion_time - part.due_date)
                # mean_w_tardiness += part.weight * min(0, part.due_date - part.completion_time)
                if part.completion_time['LH'] > make_span:
                    make_span = part.completion_time['LH']

            # sum_reward_for_tardiness = mean_w_tardiness / len(self.model['Sink'].sink)
            sum_reward_for_makespan = make_span
        sum_reward_for_tardiness = sum_reward_for_tardiness

        # if self.done == True:
        #     max_completion_time = self.model['Sink'].sink[0].completion_time
        #     for part in self.model['Sink'].sink:
        #         if part.completion_time > max_completion_time:
        #             max_completion_time = part.completion_time
        #         # total_weight_sum += part.weight
        #         # sum_reward_for_tardiness += part.weight * (min(0, part.due_date - part.completion_time))
        #
        #     sum_reward_for_makespan = 1 / max_completion_time
        # sum_reward_for_tardiness = sum_reward_for_tardiness / total_weight_sum
        # sum_reward_for_tardiness = sum_reward_for_tardiness / 15

        # 선형적으로 더할 시 각각에 대한 coefficient가 필요
        sum_reward = sum_reward_for_makespan + sum_reward_for_tardiness

        return sum_reward_for_tardiness


if __name__ == '__main__':
    forming_shop = ENVIRONMENT()

    _, _, _ = forming_shop.reset()
    # print('BH (buffer to machine)  : ', len(forming_shop.model['BH'].buffer_to_machine.items))
    # print('BH (machine store)  : ', len(forming_shop.model['BH'].machine_store.items))
    # print('BH (routed) : ', forming_shop.model['BH'].parts_routed)
    # print('LH (buffer to machine)  : ', len(forming_shop.model['LH'].buffer_to_machine.items))
    # print('LH (machine store)  : ', len(forming_shop.model['LH'].machine_store.items))
    # print('LH (routed) : ', forming_shop.model['LH'].parts_routed)

    for i in range(180):
        next_observations, next_state1, reward1, done1, agent_done, _, routing = forming_shop.step([0, 0])

        print('BH (buffer to machine)  : ', len(forming_shop.model['BH'].buffer_to_machine.items))
        print('BH (machine store)  : ', len(forming_shop.model['BH'].machine_store.items))
        print('BH (routed) : ', forming_shop.model['BH'].parts_routed)
        print('LH (buffer to machine)  : ', len(forming_shop.model['LH'].buffer_to_machine.items))
        print('LH (machine store)  : ', len(forming_shop.model['LH'].machine_store.items))
        print('LH (routed) : ', forming_shop.model['LH'].parts_routed)
        # print(len(forming_shop.model['Sink'].sink))
        # print(forming_shop.model['LH'].parts_routed)
        print(next_observations)
        # print(next_state1)
        print(reward1)
        print(done1)
        print(routing)

    print(forming_shop.mean_weighted_tardiness)
    # print(forming_shop.model['Sink'].sink[0].completion_time)
    # LH가 직전 공정인 BH보다 capacity가 크기 때문에 LH가 병목공정이 되고
    # buffer to machine 에는 part가 많이 쌓이는 반면 idle machine은 하나씩 발생하여
    # idle machine 하나가 발생하면 어떤 part를 먼저 해당 idle machine에 투입할 지 agent가 결정해준다.
