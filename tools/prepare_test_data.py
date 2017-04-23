#!/usr/local/bin/python

import pandas as pd
from datetime import *
import numpy as np
import os
import io
import sys
from Queue import Queue
from threading import Thread


def datetime_converter(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


input_dir = '../candlesticks/10min/eur_usd/'
output_dir = '../train_data/eur_usd/'
files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv") and os.path.isfile(os.path.join(input_dir, f))])

THREADS = 8
INPUT_SIZE = 10


def dt64_to_datetime(x):
    return pd.to_datetime(str(x)).replace(tzinfo=None).to_pydatetime()


def worker():
    while not q.empty():
        try:
            item = q.get()
            print 'processing: ' + item
            df = pd.read_csv(os.path.join(input_dir, item),
                             # nrows=50,
                             names=['Date', 'AskOpen', 'AskHigh', 'AskLow', 'AskClose', 'BidOpen', 'BidHigh', 'BidLow',
                                    'BidClose'],
                             converters={
                                 'Date': datetime_converter
                             },
                             dtype={
                                 'AskOpen': np.float64,
                                 'AskHigh': np.float64,
                                 'AskLow': np.float64,
                                 'AskClose': np.float64,
                                 'BidOpen': np.float64,
                                 'BidHigh': np.float64,
                                 'BidLow': np.float64,
                                 'BidClose': np.float64
                             },
                             error_bad_lines=False)
            dates = df['Date'].values
            df = df.set_index('Date')

            ahlhigh = (df['AskHigh'] + df['BidHigh']) / 2.
            ahllow = (df['AskLow'] + df['BidLow']) / 2.

            ahl = (ahlhigh + ahllow) / 2.

            vals = ahl.values

            result = []
            for i in xrange(0, vals.size - INPUT_SIZE - 1):
                # date = dates[i + INPUT_SIZE - 1]
                # date = dt64_to_datetime(date)
                # day = int(date.strftime('%w'))
                # hour = int(date.strftime('%H'))
                # minute = int(date.strftime('%M'))
                # result.append([day, hour, minute] + vals[i:i + INPUT_SIZE].tolist() + [vals[i + INPUT_SIZE]])
                result.append(vals[i:i + INPUT_SIZE].tolist() + [vals[i + INPUT_SIZE]])

            with io.FileIO(output_dir + 'train_' + item, "w") as newFile:
                for line in result:
                    newFile.write(str(line)[1:-1] + '\n')
        except ValueError as e:
            print 'exception', item, e
        except IOError as e:
            print 'exception', item, e
        except KeyError as e:
            print 'exception', item, e
        except TypeError as e:
            print 'exception', item, e
        except AttributeError as e:
            print 'exception', item, e
        q.task_done()


q = Queue()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for item in files:
    q.put(item)

for i in range(THREADS):
    t = Thread(target=worker)
    t.daemon = True
    t.start()

q.join()
