import sys
sys.path.insert(0, '/home/joe/github/caffe-4-17/caffe/python/')
import caffe
from utils import surgery, score, layers

import numpy as np
import os

# import setproctitle
# setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'vgg16fc.caffemodel'

# init
# caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

for epoch in range(10):
    solver.step(4000)
    solver.net.save('epoch_'+str(epoch)+'.caffemodel')
'''
# scoring
val = np.loadtxt('../data/segvalid11.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
'''