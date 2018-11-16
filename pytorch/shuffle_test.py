import numpy as np
import random
import time

arr1 = list(range(100))
arr2 = np.array(arr1)

start = time.time()
for i in range(1000):
    random.shuffle(arr1)
    a = [0]+arr1+[0]
print(time.time()-start)


start = time.time()
for i in range(1000):
    np.random.shuffle(arr2)
    a = np.concatenate([[0]+arr2+[0]])
    
print(time.time()-start)