import numpy as np

episode = [[[1,2,3,4], [1,0,0], 3],[[1,2,3,4], [1,0,0], 3], [[1,2,3,4], [1,0,0], 3]]
buffer = []
for _ in range(3):
    buffer.append(episode)

buffer = np.array(buffer, dtype=object)
sample = np.reshape(buffer, [9, -1])

print(sample[0][0])

a = [[[1,2,3]], [[4,5,6]], [[7,8,9]]]
a = np.vstack(a)
print(a)

weight = np.random.uniform(0, 5, 10)
print(weight)

p_ij = {'BH': np.random.uniform(1/60, 20/60, size=(10, 8)),
                     'LH': np.random.uniform(5/60, 20/60, size=(10, 8))}

print(p_ij)