
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        t = time.time()
        r = func(*args, **kwargs)
        print(f'{func} run time: {time.time() - t}')
        return r
    return wrapper
