# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:37:40 2016

@author: rbodo
"""

import os
import caffe
import numpy as np
from google.protobuf import text_format

path = '/home/path_to_caffemodel'
filename = 'model_name'

prototxt_filepath = os.path.join(path, filename + '.prototxt')
caffemodel_filepath = os.path.join(path, filename + '.caffemodel')
caffemodel = caffe.Net(prototxt_filepath, caffemodel_filepath, caffe.TEST)
model_protobuf = caffe.proto.caffe_pb2.NetParameter()
text_format.Merge(open(prototxt_filepath).read(), model_protobuf)

parameters = []
for layer in model_protobuf.layer:
    W = caffemodel.params[layer.name][0].data
    b = caffemodel.params[layer.name][1].data
    parameters.append([W, b])

out_path = os.path.join(path, 'parameters')
np.save(out_path, np.array(parameters, dtype='float32'))
