import multiprocessing
import sys
import os
from multiprocessing import Pool
from time import time
import numpy as np

def EvaluateModel(t_stim, testing_examples, normalisation):
    current = multiprocessing.current_process()
    print('Started {}'.format(current))
    f_name = "errorlog/" + current.name +"_stdout.txt"
    g_name = "errorlog/" + current.name + "_stderror.txt"
    f = open(f_name, 'w')
    g = open(g_name, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = g
        
    from spinn_front_end_common.utilities import globals_variables
    globals_variables.unset_simulator()

    extra_args = ['lenet_dense_dt_01_normalised_' +str(normalisation)+ '_serialised', '--t_stim', str(t_stim), '--testing_examples',\
                  str(testing_examples), '--result_filename', 'output_data_dt_01_normalised_'+str(normalisation), '--result_dir', 'results',\
                  '--chunk_size', '20']
    import pynn_object_serialisation.experiments.mnist_testing.mnist_testing as mnist_testing
    from pynn_object_serialisation.experiments.mnist_testing.mnist_argparser import parser
    new_args = parser.parse_args(extra_args) 
    sys.stderr.write("Input {}, Output {} \n".format(new_args.model, new_args.result_filename))
    mnist_testing.run(new_args)
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    print("Run for {} completed".format(normalisation))
    
    return

po = Pool(7)
range_input = np.array([90])
input_data = [(1000,100, i) for i in range_input]
output = po.starmap(EvaluateModel, input_data)
print('Done!')
