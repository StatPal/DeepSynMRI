from joblib import Parallel, delayed
def process(i):
    return i * i
    
results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
print(results)


import numpy as np

def process(i, vec):
    return [i ** 0.3, i * vec[i]]

vec = np.asarray(range(100))

process(1, vec)

results = Parallel(n_jobs=2)(delayed(process)(i, vec) for i in range(10))
print(results)






def big_fn(n, n_job):
    def process(i):
        return [i ** 0.3];
    results = Parallel(n_jobs=n_job)(
        delayed(process)(i) for i in range(n))
    return results





import time

start = time.time()
print(big_fn(4, 2))
end = time.time()
print(end - start)
