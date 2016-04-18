# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:46:20 2016

@author: joe
"""

import sys
sys.path.insert(0, '/home/joe/github/caffe-4-17/caffe/python/')

import caffe

#net = caffe.Net('euclid.prototxt', caffe.TEST)

solver = caffe.SGDSolver('solver.prototxt')

solver.step(100)