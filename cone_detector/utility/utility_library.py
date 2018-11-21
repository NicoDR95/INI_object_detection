import time
from datetime import datetime, timedelta
import logging
log = logging.getLogger()

def measure_time(method):
    def timed(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        #elapsed_time = str(timedelta(seconds=end_time - start_time)) #.split('.', 2)[0]
        elapsed_time = end_time - start_time
        log.info("{} run completed in {:.3f}s".format(method.__name__, elapsed_time))

        return result

    return timed

