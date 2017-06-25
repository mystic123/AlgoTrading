#!/usr/bin/env python

import Queue
import argparse
import datetime as dt
import io
import multiprocessing
import os
import sys
from io import BytesIO
import threading
from urllib import urlopen
from zipfile import ZipFile

from dateutil.relativedelta import relativedelta

URL = 'http://ratedata.gaincapital.com/'


def _get_parser():
    parser = argparse.ArgumentParser(description='Tool for downlaoding forex data from http://ratedata.gaincapital.com')
    parser.add_argument('pair', type=str, help='currency pair with underscore eg. EUR_USD')
    parser.add_argument('date_from', type=str, help='date from in format Y-m eg. 2015-01')
    parser.add_argument('date_to', type=str, default='', help='date to in format Y-m eg. 2016-01')
    parser.add_argument('-o', '--dir', type=str, default='./', help='dir where save files')
    parser.add_argument('--threads', type=int, default=multiprocessing.cpu_count(), help='number of worker threads')
    return parser


class ProcessThread(threading.Thread):
    def __init__(self, queue, dest_dir):
        threading.Thread.__init__(self)
        self.queue = queue
        self.dest_dir = dest_dir

    def run(self):
        while not self.queue.empty():
            url, date_prefix, download_file, week = self.queue.get()
            try:
                resp = urlopen(url)
                if resp.getcode() == 200:
                    sys.stderr.write('Downloading: ' + url + '\n')
                    with ZipFile(BytesIO(resp.read())) as zfile:
                        file_name = date_prefix + download_file + str(week) + '.csv'
                        with io.FileIO(os.path.join(self.dest_dir, file_name), 'w') as new_file:
                            if len(zfile.namelist()) > 1:
                                sys.stderr.write('More than one file in archive: {}\n'.format(file_name))
                            zip_content = zfile.read(zfile.namelist()[0])
                            new_file.write(zip_content)
            except Exception as e:
                print type(e)
                print e

            self.queue.task_done()


def main():
    parser = _get_parser()
    args = parser.parse_args()

    save_dir = args.dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    date_from = dt.datetime.strptime(args.date_from, '%Y-%m')
    date_to = dt.datetime.strptime(args.date_to, '%Y-%m')

    date = date_from

    queue = Queue.Queue()

    while date < date_to:
        download_file = args.pair + '_Week'
        date_prefix = date.strftime('%Y-%m-')
        for week in xrange(1, 6):
            url = URL + date.strftime('%Y/%m%%20%B/') + download_file + str(week) + '.zip'
            queue.put([url, date_prefix, download_file, week])
        date += relativedelta(months=1)

    for i in range(args.threads):
        t = ProcessThread(queue, args.dir)
        t.start()

    queue.join()


if __name__ == '__main__':
    main()
