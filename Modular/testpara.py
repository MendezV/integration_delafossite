import concurrent.futures as cf
from time import time
from traceback import print_exc

def _findmatch(nmin, nmax, number):
    '''Function to find the occurrence of number in range nmin to nmax and return
       the found occurrences in a list.'''
    print('\n def _findmatch', nmin, nmax, number)
    start = time()
    match=[]
    for n in range(nmin, nmax):
        if number in str(n):
            match.append(n)
    end = time() - start
    print("found {0} in {1:.4f}sec".format(len(match),end))
    return match

def _concurrent_submit(nmax, number, workers):
    '''Function that utilises concurrent.futures.ProcessPoolExecutor.submit to
       find the occurences of a given number in a number range in a parallelised
       manner.'''
    # 1. Local variables
    start = time()
    chunk = nmax // workers
    futures = []
    found =[]
    #2. Parallelization
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        # 2.1. Discretise workload and submit to worker pool
        for i in range(workers):
            cstart = chunk * i
            cstop = chunk * (i + 1) if i != workers - 1 else nmax
            futures.append(executor.submit(_findmatch, cstart, cstop, number))
        # 2.2. Instruct workers to process results as they come, when all are
        #      completed or .....
        cf.as_completed(futures) # faster than cf.wait()
        # 2.3. Consolidate result as a list and return this list.
        for future in futures:
            for f in future.result():
                try:
                    found.append(f)
                except:
                    print_exc()
        foundsize = len(found)
        end = time() - start
        print('within statement of def _concurrent_submit():')
        print("found {0} in {1:.4f}sec".format(foundsize, end))
    return found

if __name__ == '__main__':
    nmax = int(1E8) # Number range maximum.
    number = str(5) # Number to be found in number range.
    workers = 6     # Pool of workers

    start = time()
    a = _concurrent_submit(nmax, number, workers)
    end = time() - start
    print('\n main')
    print('workers = ', workers)
    print("found {0} in {1:.4f}sec".format(len(a),end))


#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
print("MAP")
import concurrent.futures as cf
import itertools
from time import time
from traceback import print_exc

def _findmatch(listnumber, number):    
    '''Function to find the occurrence of number in another number and return
       a string value.'''
    #print('def _findmatch(listnumber, number):')
    #print('listnumber = {0} and ref = {1}'.format(listnumber, number))
    if number in str(listnumber):
        x = listnumber
        #print('x = {0}'.format(x))
        return x 

def _concurrent_map(nmax, number, workers):
    '''Function that utilises concurrent.futures.ProcessPoolExecutor.map to
       find the occurrences of a given number in a number range in a parallelised
       manner.'''
    # 1. Local variables
    start = time()
    chunk = nmax // workers
    futures = []
    found =[]
    #2. Parallelization
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        # 2.1. Discretise workload and submit to worker pool
        for i in range(workers):
            cstart = chunk * i
            cstop = chunk * (i + 1) if i != workers - 1 else nmax
            numberlist = range(cstart, cstop)
            futures.append(executor.map(_findmatch, numberlist,
                                        itertools.repeat(number),
                                        chunksize=10000))
        # 2.3. Consolidate result as a list and return this list.
        for future in futures:
            for f in future:
                if f:
                    try:
                        found.append(f)
                    except:
                        print_exc()
        foundsize = len(found)
        end = time() - start
        print('within statement of def _concurrent(nmax, number):')
        print("found {0} in {1:.4f}sec".format(foundsize, end))
    return found

if __name__ == '__main__':
    nmax = int(1E8) # Number range maximum.
    number = str(5) # Number to be found in number range.
    workers = 6     # Pool of workers

    start = time()
    a = _concurrent_map(nmax, number, workers)
    end = time() - start
    print('\n main')
    print('workers = ', workers)