from environment.RL_SimComponent import *

job_type = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
weight = np.random.uniform(0, 5, len(job_type))
machine_num = {'BH': 5, 'LH': 8}
# job type 별 average process time
p_ij = {'BH': np.random.uniform(1, 20, size=(len(job_type), machine_num['BH'])),
        'LH': np.random.uniform(10, 40, size=(len(job_type), machine_num['LH']))}
p_j = {'BH': np.average(p_ij['BH'], axis=1), 'LH': np.average(p_ij['LH'], axis=1)}
# LH : average process time (20.8), BH : average process time (4.45)
# LH : machine 8 대 , BH : machine 2 대
process_list = ['BH', 'LH', 'Sink']
process_all = ['BH', 'LH']
priority = {'BH': [1, 2, 3, 4, 5], 'LH': [1, 2, 3, 4, 5, 6, 7, 8]}
arrival_rate = machine_num['BH'] / np.average(p_j['BH'])
# IAT = 1 / arrival_rate * 0.8
IAT = 15
part_num = 300
# due date generating factor
K = 1

env = simpy.Environment()
model = dict()

source = Source(env, IAT, weight, job_type, p_ij, model, process_list,
                machine_num,
                part_num, K)

for i in range(len(process_all) + 1):
    if i == len(process_all):
        model['Sink'] = Sink(env)
    else:
        model[process_all[i]] = Process(env, process_all[i], machine_num[process_all[i]],
                                             priority[process_all[i]],
                                             model, process_list)
# for process in process_all:
#     model[process].action = 3

env.run()
print(len(model['BH'].buffer_to_machine.items))
print(len(model['BH'].machine_store.items))
print(model['BH'].buffer_to_process.items)
print(model['LH'].buffer_to_machine.items)
print(model['LH'].buffer_to_process.items)
print(model['BH'].parts_sent)
print(model['LH'].parts_sent)