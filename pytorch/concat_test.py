import numpy as np
import time

arr = np.array([1,2,3])

zero = np.array([[0]*100])


start = time.time()
for i in range(100):
    out = np.concatenate([[0]+arr+[0]])
print(time.time()-start)


temp =np.stack([arr for _ in range(100)])
start = time.time()
for i in range(100):
    out = np.concatenate([[0]+arr+[0]])
print(time.time()-start)



