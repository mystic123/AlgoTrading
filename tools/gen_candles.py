import pandas as pd
import numpy as np
import os
import sys
from Queue import Queue
from threading import Thread
import datetime

THREADS = 8


def datetime_converter(x):
    if len(x) == len('2016-01-01 12:34:56.123'):
        x = x[:-4]
    if len(x) == len('2016-01-01 12:34:56.123456789'):
        val = datetime.datetime.strptime(x[:-3], '%Y-%m-%d %H:%M:%S.%f')
    else:
        val = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return val - datetime.timedelta(hours=17)


def read_file(file):
    try:
        df = pd.read_csv(file,
                         # nrows=1000,
                         names=['', '', '', 'Date', 'Bid', 'Ask'],
                         converters={
                             'Date': datetime_converter
                         },
                         dtype={
                             'Bid': np.float64,
                             'Ask': np.float64
                         },
                         usecols=['Date', 'Bid', 'Ask'],
                         error_bad_lines=False)
    except ValueError:
        df = pd.read_csv(file,
                         # nrows=1000,
                         names=['', '', 'Date', 'Bid', 'Ask', ''],
                         converters={
                             'Date': datetime_converter
                         },
                         dtype={
                             'Bid': np.float64,
                             'Ask': np.float64
                         },
                         usecols=['Date', 'Bid', 'Ask'],
                         error_bad_lines=False)

    df = df.set_index('Date')

    ask = df['Ask'].resample('1min', label='right').ohlc().dropna()
    bid = df['Bid'].resample('1min', label='right').ohlc().dropna()

    candles = pd.concat([ask, bid], axis=1, keys=['Ask', 'Bid'])

    return candles


def worker():
    while not q.empty():
        item = q.get()
        print 'processing: ' + item
        try:
            candles = read_file(os.path.join(input_dir, item))
            candles.to_csv(path_or_buf=output_dir + item, header=False, index_label=False)
        except IOError as e:
            print 'exception', item, e
        except ValueError as e:
            print 'exception', item, e
        q.task_done()


input_dir = '../data/eur_usd/'
output_dir = '../candlesticks/1min/eur_usd/'
files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv") and os.path.isfile(os.path.join(input_dir, f))])
files = files[35:]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

q = Queue()

for item in files:
    q.put(item)

for i in range(THREADS):
    t = Thread(target=worker)
    t.daemon = True
    t.start()

q.join()
