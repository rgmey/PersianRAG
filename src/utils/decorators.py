
import time
import logging
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logging.getLogger(__name__).debug(
            "%s executed in %.4f seconds", func.__name__, duration
        )
        print(func.__name__, f'Executed in {round(duration, 2)} Secodns')
        return result
    return wrapper
