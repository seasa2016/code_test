import time
import numpy as np

arr = np.array(list(range(10000)))
t = np.random.rand(10000)

start = time.time()
for i in range(10000):
    t[arr[i:]] = 0
    len(t[arr[i:]])
print(time.time()-start)


start = time.time()
for i in range(10000):
    t[arr[i:]] = 0
    len(arr) - i//10
print(time.time()-start)
