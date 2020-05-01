# -*- coding: utf-8 -*-
"""
Used to work on the well-trained ANN model.

@author: qinyu
"""

from snntoolbox.bin.run import main
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# main('/home/qinche/PycharmProjects/snn_resnet/snn_toolbox/temp_cifar/1587585638.3050525/config')
main('/home/qinche/PycharmProjects/snn_resnet/snn_toolbox/temp_imagenet/1588344826.070348/config')