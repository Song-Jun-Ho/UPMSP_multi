from SimComponent import *
import simpy
import copy
import tensorflow as tf


# v = tf.constant([[1], [3], [4]])
# a = tf.constant([[1, 2, 3], [4, 5, 6], [1, 1, 1]])
# avg = tf.reduce_mean(a, axis= -1, keepdims=True)
# print(v+a)
# print(avg)
# print(v+a - avg)
#
# p_ij = {'BH': np.random.uniform(1, 20, size=(10, 8)),
#                      'LH': np.random.uniform(1, 20, size=(10, 8))}
#
# print(p_ij)
#
# weight = np.random.uniform(0, 5, 10)
# print(weight)
#
# job_type_list = np.random.randint(low=0, high=10, size=(1, 200))
# print(job_type_list)

test_list = []

# for i in range(1000):
#     test_list.append(i)

test_list = [i for i in range(1000)]

print(test_list)