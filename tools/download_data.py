#!/usr/local/bin/python
import datetime as dt
from dateutil.relativedelta import relativedelta
import sys
import io
from zipfile import ZipFile
from io import BytesIO, FileIO, StringIO
from urllib import urlopen
import os

if len(sys.argv) < 4:
	print 'Arguments: pair date-from(Y-m) date-to(Y-m) dir\nExample: ' + sys.argv[0] + ' EUR_USD 2015-01 2016-01 data'
	sys.exit()

if len(sys.argv) >= 4:
    pair = sys.argv[1].upper()
    dateFrom = dt.datetime.strptime(sys.argv[2], '%Y-%m')
    dateTo = dt.datetime.strptime(sys.argv[3], '%Y-%m')

path = '.'

if len(sys.argv) >= 5:
	path = sys.argv[4]

if not os.path.exists(path):
	os.makedirs(path)

source = 'http://ratedata.gaincapital.com/'
date = dateFrom
while date < dateTo:
	downloadFile = pair + '_Week'
	datePrefix = date.strftime('%Y-%m-')
	for week in xrange(1,6):
		url = source + date.strftime('%Y/%m%%20%B/') + downloadFile + str(week) + '.zip'
		try:
			resp = urlopen(url)
			if resp.getcode() == 200:
				sys.stderr.write('Downloading: ' + url + '\n')
				with ZipFile(BytesIO(resp.read())) as zfile:
					fileName = datePrefix + downloadFile + str(week) + '.csv'
					with io.FileIO(os.path.join(path,fileName), 'w') as newfile:
						if len(zfile.namelist()) > 1:
							sys.stderr.write('More than one file in archive: ' + fileName)
						zipcontent = zfile.read(zfile.namelist()[0])
						newfile.write(zipcontent)
		except Exception as e:
			print type(e)
			print e
	date += relativedelta(months=1)
