#!/usr/bin/python

import os
import io
import sys
import re

if len(sys.argv) < 3:
    sys.stderr.write('Arguments: input_dir output_dir\nExample: ' + sys.argv[0] + ' data merged\n')
    sys.exit()

input_dir = '.'
output_dir = '.'
if len(sys.argv) >= 2:
    input_dir = sys.argv[1]

if len(sys.argv) >= 3:
    output_dir = sys.argv[2]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')])
processed = set()

for f in files:
    datePrefix = f[:len('2001-01')]
    output_file = datePrefix + '.csv'
    if re.match('\d{4}-\d{2}.csv',f) is None:
        if f not in processed:
            processed.add(f)
            to_merge = [x for x in files if x[:-5] == f[:-5] and x not in processed]
            with io.FileIO(os.path.join(output_dir, output_file), "w") as outFile:
                print output_file, f, to_merge
                for nextFile in to_merge:
                    processed.add(nextFile)
                    with io.FileIO(os.path.join(input_dir,nextFile), "r") as nextF:
                        for line in nextF:
                            if line[0] != 'l':
                                outFile.write(nextF.read())

        """
    if f not in processed:
        processed.add(f)
        datePrefix = f[:7]
        print '------------'
        for nextFile in [x for x in files if x.startswith(datePrefix)]:
            print f, datePrefix
            processed.add(nextFile)
        """
        """
            if re.match('\d{4}-\d{2}',datePrefix) is not None:
                pass

                with io.FileIO(os.path.join(output_dir, datePrefix + '.csv'), "w") as newFile:
                    for nextFile in sorted([x for x in files if x.startswith(datePrefix) and x != datePrefix+'.csv']):
                        with io.FileIO(os.path.join(input_dir,nextFile), "r") as nextF:
                            line = nextF.read()
                            print nextFile, datePrefix
                            if line[0] != 'l':
                                newFile.write(nextF.read())
                        processed.add(nextFile)
        """
