# -*- coding: utf-8 -*-
"""

The modules in ``target_simulators`` package allow building a spiking network
and exporting it for use in a spiking simulator.

This particular module offers functionality for Brian2 simulator. Adding
another simulator requires implementing the class ``SNN_compiled`` with its
methods tailored to the specific simulator.

Created on Thu May 19 15:00:02 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
from builtins import int, range

import numpy as np
import os
import sys
import subprocess
from abc import ABCMeta, abstractmethod
from random import randint

from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator

standard_library.install_aliases()


#TODO This is ugly, i can use the sim object to store the megasim path
MEGASIM_PATH = "/Users/Evangelos/Programming/NPP/megasim/megasim/bin/"

#TODO there is a lot of duplicate code in these classes, maybe i can create a base class and use inheritance


class Megasim_base(metaclass=ABCMeta):
    """
        Class that holds the common attributes and methods for the MegaSim modules.

        Parameters
        ----------

        Attributes
        ----------
        Attributes set to -1 must be set by each subclass, the rest can be used as default values

        Attributes common to all MegaSim modules
        n_in_ports: int
            Number of input ports
        n_out_ports: int
            Number of output ports
        delay_to_process: int
            Delay to process an input event
        delay_to_ack: int
            Delay to acknoldege an event
        fifo_depth: int
            Depth of input fifo
        n_repeat: int

        delay_to_repeat: int

        #Parameters for the convolutional module and avgerage pooling module
        Nx_array: int
            X dimensions of the feature map
        Ny_array: int
            Y dimensions of the feature map
        Xmin: int
            start counting from Xmin (=0)
        Ymin: int
            start counting from Ymin (=0)
        THplus:
            Positive threshold
        THplusInfo:
            Flag to enable spikes when reaching the positive threshold
        THminus:
            Negative threshold
        THminusInfo:
            Flag to enable spikes with negative polarity when reaching the negative threshold
        Reset_to_reminder:
            After reaching the threshold if set it will set the membrane to the difference
        MembReset: int
            Resting potential (=0)
        TLplus: int
            Linear leakage slope from the positive threshold
        TLminus: int
            Linear leakage slope from the negative threshold
        Tmin: int
            minimum time between 2 spikes
        T_Refract: int
            Refractory period

        # Parameters for the output
        crop_xmin: int
            Xmin crop of the feature map
        crop_xmax: int

        crop_ymin: int

        crop_ymax: int

        xshift_pre: int
            X shift before subsampling
        yshift_pre: int
            Y shift before subsampling
        x_subsmp: int
            Subsampling (=1 if none)
        y_subsmp: int

        xshift_pos: int
            X shift after subsampling
        yshift_pos: int

        rectify: int
            Flag that if set will force all spikes to have positive polarity

        # The fully connected module has population_size instead of Nx_array
        population_size: int
            Number of neurons in the fully connected module
        Nx_array_pre: int
            Number of neurons in the previous layer

        # Needed by the state file
        time_busy_initial: int
            Initial state of the module (=0)

        # Scaling factor
        scaling_factor: int
            Scaling factor for the parameters since MegaSim works with integers

        Methods
        -------
        build_state_file:
            Input parameters: a string with the full path to the megasim SNN directory

            This method is similar to all MegaSim modules. It generates an initial state file
            per module based on the time_busy_initial.

        build_parameter_file:
            Input parameters: a string with the full path to the megasim SNN directory

            This method generates the module's parameter file based on its attributes set by
            the sub-class.
            This method depends on the MegaSim module and will raise error if not implemented.
    """
    # Attributes common to all MegaSim modules
    n_in_ports  = -1
    n_out_ports = 1
    delay_to_process = 0
    delay_to_ack = 0
    fifo_depth = 0
    n_repeat = 1
    delay_to_repeat = 15

    #Parameters for the conv module and avg pooling
    Nx_array = -1
    Ny_array = 1
    Xmin = 0
    Ymin = 0
    THplus = 0
    THplusInfo = 1
    THminus = -2147483646
    THminusInfo = 0
    Reset_to_reminder = 0
    MembReset = 0
    TLplus = 0
    TLminus = 0
    Tmin = 0
    T_Refract = 0

    # Parameters for the output
    crop_xmin = -1
    crop_xmax = -1
    crop_ymin = -1
    crop_ymax = -1
    xshift_pre = 0
    yshift_pre = 0
    x_subsmp = 1
    y_subsmp = 1
    xshift_pos = 0
    yshift_pos = 0
    rectify = 0

    # The fully connected module has population_size instead of Nx_array
    population_size = -1
    Nx_array_pre = -1

    # Needed by the state file
    time_busy_initial = 0

    # Scaling factor
    scaling_factor = 1

    def __init__(self):
        pass

    def build_state_file(self, dirname):
        '''
        dirname = the full path of the
        '''
        f = open(dirname + self.label + ".stt", "w")
        f.write(".integers\n")
        f.write("time_busy_initial %d\n" % self.time_busy_initial)
        f.write(".floats\n")
        f.close()

    @abstractmethod
    def build_parameter_file(self, dirname):
        pass


class module_input_stimulus():
    '''
    A dummy class for the input stimulus.

    Parameters
    ----------

    label: string
        String to hold the module's name.

    pop_size: int
        Integer to store the population size.


    Attributes
    ----------

    label: string

    pop_size: int

    input_stimulus_file: string
        String to hold the filename of the input stimulus

    module_string: string
        String that holds the module name for megasim

    evs_files: list
        List of strings of the event filenames that will generated when a megasim simulation is over.

    '''

    def __init__(self, label, pop_size):
        self.label = label
        self.pop_size = pop_size
        self.input_stimulus_file = "input_events.stim"
        self.module_string = "source"
        self.evs_files =[]

class module_flatten(Megasim_base):
    '''
    A class for the flatten megasim module. The flatten module is used to connect a 3D population to a
    1D population. eg A convolutional layer to a fully connected one.

    Parameters
    ----------

    layer_params: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    input_ports: int
        Number of input ports (eg feature maps from the previous layer)

    fm_size: tuple
        Tuple of integers that holds the size of the feature maps from the previous layer

    Attributes
    ----------

    module_string: string
        String that holds the module name for megasim

    output_shapes: tuple
        Tuple that holds the shape of the output of the module. Used for the plotting.

    evs_files: list
        List of strings of the event filenames that will generated when a megasim simulation is over.

    '''
    def __init__(self, layer_params, input_ports, fm_size):
        self.module_string = "module_flatten"
        self.label=layer_params["label"]
        self.output_shapes = layer_params["output_shape"]
        self.evs_files = []

        self.n_in_ports = input_ports

        self.Nx_array = fm_size[0]
        self.Ny_array = fm_size[1]

    def build_parameter_file(self, dirname):
        """

        """
        param1=(
    """.integers
n_in_ports %d
n_out_ports %d
delay_to_process %d
delay_to_ack %d
fifo_depth %d
n_repeat %d
delay_to_repeat %d
"""%(self.n_in_ports, self.n_out_ports,self.delay_to_process, self.delay_to_ack, self.fifo_depth, self.n_repeat,
     self.delay_to_repeat))

        param_k = (
         """Nx_array %d
Ny_array %d
""" % (self.Nx_array,
           self.Ny_array))


        q=open(dirname+self.label+'.prm',"w")
        q.write(param1)

        for k in range(self.n_in_ports):
            q.write(param_k)

        q.write(".floats\n")
        q.close()


class Module_average_pooling(Megasim_base):
    '''

    duplicate code with the module_conv class - TODO: merge them
    layer_params
    dict_keys(['label', 'layer_num', 'border_mode', 'layer_type', 'strides', 'input_shape', 'output_shape', 'get_activ', 'pool_size'])
    '''

    def __init__(self, layer_params, neuron_params, scaling_factor=10000000):
        self.module_string = 'module_conv'
        self.layer_type = layer_params['layer_type']
        self.output_shapes = layer_params['output_shape'] #(none, 32, 26, 26) last two
        self.label = layer_params['label']
        self.evs_files = []

        self.n_in_ports = 1 # one average pooling layer per conv layer
        #self.in_ports = 1 # one average pooling layer per conv layer
        self.num_of_FMs = layer_params['input_shape'][1]

        self.fm_size = layer_params['output_shape'][2:]
        self.Nx_array, self.Ny_array = self.fm_size[1] * 2, self.fm_size[0] * 2
        self.Dx, self.Dy = 0, 0
        self.crop_xmin, self.crop_xmax = 0, self.fm_size[0] * 2 - 1
        self.crop_ymin, self.crop_ymax = 0, self.fm_size[1] * 2 - 1
        self.xshift_pre, self.yshift_pre = 0, 0

        self.strides = layer_params["strides"]

        self.num_pre_modules = layer_params['input_shape'][1]

        self.scaling_factor = int(scaling_factor)

        self.THplus = neuron_params["v_thresh"] * self.scaling_factor
        self.THminus = -2147483646
        self.refractory = neuron_params["tau_refrac"]
        self.MembReset = neuron_params["v_reset"] * self.scaling_factor
        self.TLplus = 0
        self.TLminus = 0

        self.kernel_size = (1,1)#layer_params['pool_size']

        self.Reset_to_reminder = 0
        if neuron_params["reset"] == 'Reset to zero':
            self.Reset_to_reminder = 0
        else:
            self.Reset_to_reminder = 1

        self.pre_shapes = layer_params['input_shape'] # (none, 1, 28 28) # last 2

        self.border_mode = layer_params['border_mode']
        if self.border_mode != 'valid':
            echo("Not implemented yet!")
            sys.exit(88)



    def build_parameter_file(self, dirname):
        sc = self.scaling_factor
        fm_size = self.fm_size
        num_FMs = self.num_pre_modules

        print("building %s with %d FM receiving input from %d pre pops. FM size is %d,%d"%(
            self.label, self.output_shapes[1],self.pre_shapes[1], self.output_shapes[2],self.output_shapes[3]))

        kernel = np.ones(self.kernel_size, dtype="float") * sc
        kernel *= ((1.0 / np.sum(self.kernel_size)))#/(np.sum(self.kernel_size)))
        kernel = kernel.astype("int")

        for f in range(num_FMs):
            fm_filename = self.label+"_"+str(f)
            self.__build_single_fm(self.n_in_ports,self.n_out_ports,fm_size,kernel,dirname,fm_filename)
        pass

    def __build_single_fm(self, num_in_ports, num_out_ports, fm_size, kernel, dirname, fprmname):
        '''

        Parameters
        ----------
        num_in_ports
        num_out_ports
        fm_size
        kernel
        dirname
        fprmname

        Returns
        -------

        '''
        param1 = (
        """.integers
n_in_ports %d
n_out_ports %d
delay_to_process %d
delay_to_ack %d
fifo_depth %d
n_repeat %d
delay_to_repeat %d
Nx_array %d
Ny_array %d
Xmin %d
Ymin %d
THplus %d
THplusInfo %d
THminus %d
THminusInfo %d
Reset_to_reminder %d
MembReset %d
TLplus %d
TLminus %d
Tmin %d
T_Refract %d
""" %( num_in_ports, num_out_ports, self.delay_to_process, self.delay_to_ack, self.fifo_depth,
       self.n_repeat, self.delay_to_repeat,
       self.Nx_array, self.Ny_array, self.Xmin, self.Ymin,
       self.THplus, self.THplusInfo, self.THminus, self.THminusInfo,
       self.Reset_to_reminder, self.MembReset, self.TLplus, self.TLminus, self.Tmin, self.T_Refract))

        param_k= (
        """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (self.kernel_size[0],
               self.kernel_size[1],
               self.Dx, self.Dy))

        kernels_list =[]
        for k in range(1):
            # scale the weights
            w = kernel

            np.savetxt(dirname + "w.txt", w, delimiter=" ", fmt="%d")
            q = open(dirname + "w.txt")
            param2 = q.readlines()
            q.close()
            os.remove(dirname + "w.txt")
            kernels_list.append(param2)

        param5 = (
            """crop_xmin %d
crop_xmax %d
crop_ymin %d
crop_ymax %d
xshift_pre %d
yshift_pre %d
x_subsmp %d
y_subsmp %d
xshift_pos %d
yshift_pos %d
rectify %d
.floats
""" % (self.crop_xmin, self.crop_xmax,
       self.crop_ymin, self.crop_ymax,
       self.xshift_pre, self.yshift_pre,
       2,2,#self.kernel_size[0], self.kernel_size[1],
       0, 0,
       0)
        )
        '''
         % (0, (fm_size[0] *self.kernel_size[0])-1,
       0, (fm_size[1] *self.kernel_size [1]) -1,
       0, 0,
       self.kernel_size[0], self.kernel_size[1],
       0, 0,
       0)
        )
        '''
        q = open(dirname + fprmname + '.prm', "w")
        q.write(param1)
        for k in range(len(kernels_list)):
            q.write(param_k)
            for i in param2:
                q.write(i)
        q.write(param5)
        q.close()




class Module_conv(Megasim_base):
    '''
    A class for the convolutional megasim module.

    Parameters
    ----------

    layer_params: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    neuron_params: dictionary
        This is the settings dictionary that is set in the config.py module

    flip_kernels: boolean
        If set will flip the kernels upside down.

    scaling_factor: int
        An integer that will be used to scale all parameters.

    Attributes
    ----------

    module_string: string
        String that holds the module name for megasim

    output_shapes: tuple
        Tuple that holds the shape of the output of the module. Used for the plotting.

    evs_files: list
        List of strings of the event filenames that will generated when a megasim simulation is over.

    num_of_FMs: int
        Number of feature maps in this layer

    w: list
        list of weights

    border_mode: string
        String with the border mode used for the convolutional layer. So far only the valid mode is implemented


    layer_params
    dict_keys(['nb_col', 'activation', 'layer_type', 'layer_num', 'nb_filter', 'output_shape', 'input_shape', 'nb_row', 'label', 'parameters', 'border_mode'])
    '''

    def __init__(self, layer_params, neuron_params, flip_kernels = True,scaling_factor=10000000):
        self.module_string = 'module_conv'
        self.layer_type = layer_params["layer_type"]
        self.output_shapes = layer_params['output_shape'] #(none, 32, 26, 26) last two
        self.label = layer_params['label']
        self.evs_files = []

        #self.size_of_FM = 0
        self.num_of_FMs = layer_params['parameters'][0].shape[0]
        self.kernel_size = layer_params['parameters'][0].shape[2:] #(kx, ky)
        self.w = layer_params['parameters'][0]

        self.n_in_ports = self.w.shape[1]
        self.pre_shapes = layer_params['input_shape'] # (none, 1, 28 28) # last 2
        self.fm_size = self.output_shapes[2:]
        self.Nx_array = self.output_shapes[2:][1]
        self.Ny_array = self.output_shapes[2:][0]

        self.border_mode = layer_params["border_mode"] # 'same', 'valid',

        self.Reset_to_reminder = 0
        if neuron_params["reset"] == 'Reset to zero':
            self.Reset_to_reminder = 0
        else:
            self.Reset_to_reminder = 1


        if self.border_mode == 'valid':
            # if its valid mode
            self.Nx_array = self.output_shapes[2:][1] + self.kernel_size[1] - 1
            self.Ny_array = self.output_shapes[2:][0] + self.kernel_size[0] - 1
            self.xshift_pre, self.yshift_pre = -int(self.kernel_size[1]/2), -int(self.kernel_size[0]/2)
            self.crop_xmin, self.crop_xmax = int(self.kernel_size[1]/2), (self.Nx_array-self.kernel_size[1]+1)
            self.crop_ymin, self.crop_ymax = int(self.kernel_size[0]/2), (self.Ny_array-self.kernel_size[0]+1)
        else:
            echo("Not implemented yet!")
            self.Nx_array = self.output_shapes[2:][1]
            self.Ny_array = self.output_shapes[2:][0]
            self.xshift_pre, self.yshift_pre = (0, 0)
            self.crop_xmin, self.crop_xmax = (0, self.Nx_array)
            self.crop_ymin, self.crop_ymax = (0, self.Ny_array)

        self.scaling_factor = int(scaling_factor)

        self.flip_kernels = flip_kernels

        self.THplus = neuron_params["v_thresh"] * self.scaling_factor
        self.T_Refract = neuron_params["tau_refrac"]
        self.MembReset = neuron_params["v_reset"]
        self.TLplus = 0
        self.TLminus = 0


    def build_parameter_file(self, dirname):
        sc = self.scaling_factor
        fm_size = self.output_shapes[2:]
        pre_num_ports = self.pre_shapes[1]
        num_FMs = self.output_shapes[1]
        print("building %s with %d FM receiving input from %d pre pops. FM size is %d,%d"%(
            self.label, self.output_shapes[1],self.pre_shapes[1], self.output_shapes[2],self.output_shapes[3]))

        for f in range(num_FMs):
            fm_filename = self.label+"_"+str(f)
            kernel = self.w[f]

            self.__build_single_fm(pre_num_ports,1,fm_size,kernel,dirname,fm_filename)
        pass

    def __build_single_fm(self, num_in_ports, num_out_ports, fm_size, kernel, dirname, fprmname):
        '''
        Helper method to create a single feature map

        Parameters
        ----------
        num_in_ports: int
            number of input ports
        num_out_ports: int
            number of output ports
        fm_size: tuple
            A tuple with the X, Y dimensions of the feature map
        kernel: numpy array
            A numpy array of X,Y dimensions with the kernel of the feature map
        dirname: string
            String with the full path of the megasim simulation folder
        fprmname: string
            Filename of the parameter file

        Returns
        -------

        '''
        sc = self.scaling_factor

        param1 = (
        """.integers
n_in_ports %d
n_out_ports %d
delay_to_process %d
delay_to_ack %d
fifo_depth %d
n_repeat %d
delay_to_repeat %d
Nx_array %d
Ny_array %d
Xmin %d
Ymin %d
THplus %d
THplusInfo %d
THminus %d
THminusInfo %d
Reset_to_reminder %d
MembReset %d
TLplus %d
TLminus %d
Tmin %d
T_Refract %d
""" %( self.n_in_ports, self.n_out_ports,
       self.delay_to_process, self.delay_to_ack, self.fifo_depth, self.n_repeat,
       self.delay_to_repeat,
       self.Nx_array, self.Ny_array,
       self.Xmin, self.Ymin,
       self.THplus, self.THplusInfo,
       self.THminus, self.THminusInfo,
       self.Reset_to_reminder, self.MembReset,
       self.TLplus, self.TLminus, self.Tmin, self.T_Refract))

        param_k= (
        """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (self.kernel_size[0],
               self.kernel_size[1],
               -int(self.kernel_size[0]/2), -int(self.kernel_size[1]/2)
       ))

        kernels_list =[]
        for k in range(kernel.shape[0]):
            w = kernel[k] * sc

            if self.flip_kernels:
                '''
                After tests i did in zurich we only need to flip the kernels upside down
                '''
                w = np.flipud(w)


            np.savetxt(dirname + "w.txt", w, delimiter=" ", fmt="%d")
            q = open(dirname + "w.txt")
            param2 = q.readlines()
            q.close()
            os.remove(dirname + "w.txt")
            kernels_list.append(param2)

        # if self.label == "02Convolution2D_32x24x24":
        #     import pdb
        #     pdb.set_trace()
        param5 = (
            """crop_xmin %d
crop_xmax %d
crop_ymin %d
crop_ymax %d
xshift_pre %d
yshift_pre %d
x_subsmp %d
y_subsmp %d
xshift_pos %d
yshift_pos %d
rectify %d
.floats
""" % (self.crop_xmin, self.crop_xmax,
       self.crop_ymin, self.crop_ymax,
       self.xshift_pre, self.yshift_pre,
       self.x_subsmp, self.y_subsmp,
       self.xshift_pos, self.yshift_pos,
       self.rectify)
        )

        q = open(dirname + fprmname + '.prm', "w")
        q.write(param1)
        for k in range(len(kernels_list)):
            q.write(param_k)
            for i in kernels_list[k]:#param2:
                q.write(i)
        q.write(param5)
        q.close()


class Module_fully_connected(Megasim_base):
    '''
    A class for the fully connected megasim module.

    Parameters
    ----------

    layer_params: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    neuron_params: dictionary
        This is the settings dictionary that is set in the config.py module

    scaling_factor: int
        An integer that will be used to scale all parameters.

    enable_softmax: Boolean
        A flag that if set will use (if the ann uses it) softmax for the output layer. If not set
        a population of LIF neurons will be used instead.

    Attributes
    ----------

    module_string: string
        String that holds the module name for megasim

    output_shapes: tuple
        Tuple that holds the shape of the output of the module. Used for the plotting.

    evs_files: list
        List of strings of the event filenames that will generated when a megasim simulation is over.

    num_of_FMs: int
        Number of feature maps in this layer

    w: list
        list of weights

    border_mode: string
        String with the border mode used for the convolutional layer. So far only the valid mode is implemented


    layer_params
    dict_keys(['nb_col', 'activation', 'layer_type', 'layer_num', 'nb_filter', 'output_shape', 'input_shape', 'nb_row', 'label', 'parameters', 'border_mode'])
    '''
    def __init__(self, layer_params, neuron_params, scaling_factor=10000000, enable_softmax=True):
        self.module_string = 'module_fully_connected'
        self.label= layer_params["label"]
        self.output_shapes= layer_params['output_shape']
        self.evs_files = []

        self.population_size = layer_params['output_shape'][1]
        self.scaling_factor = int(scaling_factor)
        self.w = layer_params["parameters"][0]
        self.Nx_array_pre = len(self.w)

        self.enable_softmax = enable_softmax


        self.THplus = neuron_params["v_thresh"] * self.scaling_factor
        self.T_Refract = neuron_params["tau_refrac"]
        self.MembReset = neuron_params["v_reset"]
        self.TLplus = 0
        self.TLminus = 0

        self.crop_xmin, self.crop_ymin = 0, 0
        self.crop_xmax, self.crop_ymax = self.population_size, self.population_size

        # Reset type
        if neuron_params["reset"] == 'Reset to zero':
            self.Reset_to_reminder = 0
        else:
            self.Reset_to_reminder = 1

        # If its the output layer choose the type
        # either population of LIF neurons or softmax
        if layer_params["activation"] == 'softmax' and self.enable_softmax == True:
            print("Using softmax for the output layer")
            self.module_string = 'module_softmax'
            self.n_in_ports = 2
        else:
            print("Using LIF")
            self.n_in_ports = 1


    def build_parameter_file(self, dirname):
        sc = self.scaling_factor

        param1=(
    """.integers
n_in_ports %d
n_out_ports %d
delay_to_process %d
delay_to_ack %d
fifo_depth %d
n_repeat %d
delay_to_repeat %d
population_size %d
Ny_array %d
Xmin %d
Ymin %d
THplus %d
THplusInfo %d
THminus %d
THminusInfo %d
Reset_to_reminder %d
MembReset %d
TLplus %d
TLminus %d
Tmin %d
T_Refract %d
Nx_array_pre %d
Ny_array_pre 1
"""%(self.n_in_ports,self.n_out_ports,
    self.delay_to_process,
    self.delay_to_ack, self.fifo_depth, self.n_repeat, self.delay_to_repeat,
    self.population_size, 1, self.Xmin, self.Ymin,
     self.THplus, self.THplusInfo,
    self.THminus, self. THminusInfo,
     self.Reset_to_reminder, self.MembReset,
     self.TLplus, self.TLminus, self.Tmin, self.T_Refract, self.Nx_array_pre))

        w = self.w * sc

        # TODO: change these lines
        np.savetxt(dirname+"w.txt",w,delimiter=" ",fmt="%d")
        q=open(dirname+"w.txt")
        param2=q.readlines()
        q.close()
        os.remove(dirname+"w.txt")

        # if the output activation is softmax add one more input for the control in events
        if self.module_string == 'module_softmax':
            param_softmax2 = " ".join([str(x) for x in [0]*self.population_size])
            param_softmax1=(
                """Nx_array_pre 1
Ny_array_pre 1
"""
            )

        param5=(
    """crop_xmin %d
crop_xmax %d
crop_ymin %d
crop_ymax %d
xshift_pre %d
yshift_pre %d
x_subsmp %d
y_subsmp %d
xshift_pos %d
yshift_pos %d
rectify %d
.floats
    """%(self.crop_xmin, self.crop_xmax, self.crop_ymin, self.crop_ymax, self.xshift_pre,
         self.yshift_pre, self.x_subsmp, self.y_subsmp, self.xshift_pos, self.yshift_pos, self.rectify))

        q=open(dirname+self.label+'.prm',"w")
        q.write(param1)
        for i in param2:
            q.write(i)

        # if we use a softmax use 0 weights for the control events
        if self.module_string == 'module_softmax':
            q.write(param_softmax1)
            q.write(param_softmax2)
            q.write("\n")
        q.write(param5)
        q.close()

    def build_softmax_conrol_events(self, megadirname):
        print("Generating control events for the softmax module")
        softmax_in_events = []
        for t in range(0,int(settings['duration'] / settings['dt'])):
            rnd = np.random.uniform(0,settings["input_rate"])
            if rnd< settings["softmax_clockrate"]:
                softmax_in_events.append([t, -1, -1, 0, -1, -1])
        print(megadirname+"softmax_input.stim")
        softmax_in_events = np.asarray(softmax_in_events)
        np.savetxt(megadirname+"softmax_input.stim",softmax_in_events,delimiter=" ",fmt="%d")


#----------------------------------------------------------------------------------------------------------------------#


class SNN_compiled():
    """
    Class to hold the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Parameters
    ----------

    ann: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    Attributes
    ----------

    ann: dict
        Parsed input model; result of applying ``model_lib.extract(in)`` to the
        input model ``in``.

    sim: Simulator
        Module containing utility functions of spiking simulator. Result of
        calling ``snntoolbox.config.initialize_simulator()``. For instance, if
        using Brian simulator, this initialization would be equivalent to
        ``import pyNN.brian as sim``.

    layers: list
        Each entry represents a layer

    spikemonitors: list
        A list of numpy arrays of the generated events of each module. Len depends on the topology
        and not on the architecture.

    output_shapes: list
        The output shapes of each layer. During conversion, all layers are
        flattened. Need output shapes to reshape the output of layers back to
        original form when plotting results later.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons in
        pyNN simulators.

    megaschematic: string
        String that holds megasim main schmatic file that is needed to run a simulation

    megadirname: string
        String that holds the full path where the generated files for a megasim simulation are stored.
        These files include the stimulus, parameter, state and schematic files. The event files will
        be generated in the same folder.

    Methods
    -------

    add_input_layer:

    check_megasim_output:
        A method that checks the prints of MegaSim for errors

    poisson_spike_generator_megasim_flatten:
        Method that converts an mnist digit to spike trains and stores it in the megadirname folder as a
        MegaSim stimulus file.

    build_schematic_updated:
        This method builds the main schematic file for running a megasim simulation

    clean_megasim_sim_data:
        Method that cleans the data generated from and for a megasim simulation. eg stimulus files and event
        files

    get_spikes:
        Method that opens all generated event files from a megasim simulation

    spike_count_histogram: Numpy array, pop_size
        Method that receives a a numpy array of events from a megasim module and the size of that population,
        creates a histogram of spike counts and returns the argmax. Returns -1 if no spikes were generated.

    build:
        Convert an ANN to a spiking neural network, using layers derived from
        Keras base classes.
    run:
        Simulate a spiking network.
    save:
        Write model architecture and parameters to disk.
    load:
        Load model architecture and parameters from disk.
    end_sim:
        Clean up after simulation.

    collect_plot_results: layers, output_shapes, ann, X_batch, idx


    """

    def __init__(self, ann):
        self.ann = ann
        #self.sim = initialize_simulator() #TODO i can get the megasim path from here!
        self.connections = []
        self.spikemonitors = []
        self.megadirname = ''
        self.megaschematic = 'megasim.sch'
        self.input_stimulus_file = "input_events.stim"
        self.layers = []
        self.add_input_layer()

        self.scaling_factor = settings['scaling_factor']

    def add_input_layer(self):
        input_shape = list(self.ann['input_shape'])
        self.layers.append(
            module_input_stimulus(label='InputLayer', pop_size = input_shape[1:])
        )


    def build(self):
        """
        Compile a spiking neural network to prepare for simulation with MegaSim.

        """

        echo('\n')
        echo("Compiling spiking network...\n")

        # Create megasim dir where it will store the SNN params and schematic
        self.megadirname = settings['path'] + settings['filename'] + '/'
        if not os.path.exists(self.megadirname):
            os.makedirs(self.megadirname)

        # clear the folder first from evs and log files
        #TODO: add a method to clean everything in that folder

        # Iterate over hidden layers to create spiking neurons and store
        # connections.
        for (layer_num, layer) in enumerate(self.ann['layers']):
            print (layer["layer_type"])
            if layer['layer_type'] == 'Dense':
                echo("Building layer: {}\n".format(layer['label']))

                # Fully connected layers
                try:
                    use_softmax = settings["output_type"]
                except(KeyError):
                    print("Activation type entry not found in the setting dictionary")
                    use_softmax = True

                self.layers.append(
                    Module_fully_connected(layer_params = layer, neuron_params = settings,
                                           scaling_factor = self.scaling_factor,
                                           enable_softmax = use_softmax)
                )

            elif layer['layer_type'] == 'Convolution2D':
                echo("Building layer: {}\n".format(layer['label']))
                self.layers.append(
                    Module_conv(layer_params=layer, neuron_params=settings,
                                scaling_factor = self.scaling_factor)
                )
            elif layer['layer_type'] == 'MaxPooling2D':
                echo("Building layer: {}\n".format(layer['label']))
                echo("Not Implemented!")
                sys.exit(88)
            elif layer['layer_type'] == 'AveragePooling2D':
                echo("Building layer: {}\n".format(layer['label']))
                self.layers.append(
                    Module_average_pooling(layer_params=layer, neuron_params=settings,
                                           scaling_factor = self.scaling_factor)
                )
            elif layer['layer_type'] == 'Flatten':
                echo("Building layer: {}\n".format(layer['label']))
                c_layer_len = len(self.layers)-1
                self.layers.append(
                    module_flatten(layer_params=layer, input_ports=self.layers[c_layer_len].num_of_FMs,
                                   fm_size=self.layers[c_layer_len].fm_size)
                )
            else:
                pass


        # build parameter files for all modules
        # ignore the input layer
        for mod_n,module in enumerate(self.layers[1:]):
            module.build_parameter_file(dirname=self.megadirname)
            module.build_state_file(dirname= self.megadirname)


        # build MegaSim Schematic file
        self.build_schematic_updated()

        echo("Compilation finished. Model is stored in the %s folder\n\n"%self.megadirname)


    def check_megasim_output(self, megalog):
        '''

        A method that checks the prints of MegaSim for errors

        Parameters
        ----------
        megalog: String
            String returned from executing megasim.

        Returns
        -------

        '''
        megalog = str(megalog)
        megalog = megalog.lower()
        if megalog.find("error")>0:
            print("MegaSim error: ")
            print(megalog)
            sys.exit(99)

    def poisson_spike_generator_megasim(self, mnist_digit):
        '''

        Parameters
        ----------
        mnist_digit: numpy array
            A 1d or 2d numpy array of an mnist digit (normalised 0-1)

        Returns
        -------
        It will store the generated spike trains to a stimulus file in the megasim sim folder
        '''
        spikes=[]
        rescale_fac = 1000/(settings['input_rate'] * settings['dt'])
        for t in np.arange(0, settings['duration'], settings['dt']):
            # Create poisson input.
            spike_snapshot = np.random.random_sample(mnist_digit.shape) * rescale_fac
            inp = (spike_snapshot <= mnist_digit).astype('float32')
            # find the indexes of the non-zero
            neuron_id = np.nonzero(inp)
            # check if input is flattened or 2d in order to extract the X,Y addresses correctly
            if len(inp.shape)==1:
                megasim_x = neuron_id[0]
                megasim_y = 0
            else:
                megasim_x = neuron_id[2]
                megasim_y = neuron_id[1]

            spike_for_t = np.zeros((len(megasim_x), 6), dtype="int")
            spike_for_t[:,0] = t            # time-stamps
            spike_for_t[:,1] = -1           # REQ
            spike_for_t[:,2] = -1           # ACK
            spike_for_t[:,3] = megasim_x    # X address
            spike_for_t[:,4] = megasim_y    # Y address
            spike_for_t[:,5] = 1            # polarity
            spikes.append(spike_for_t)
        spikes=np.vstack(spikes)
        np.savetxt(self.megadirname + self.layers[0].label+".stim", spikes, delimiter=" ", fmt=("%d"))


    def build_schematic_updated(self):
        '''

        This method generates the main MegaSim schematic file to test sample by sample
        -------

        '''
        #input_stimulus_file = self.input_stimulus_file
        #input_stimulus_node = "input_evs"

        fileo = open(self.megadirname+self.megaschematic, "w")

        fileo.write(".netlist\n")
        # stim file first - node is input_evs
        fileo.write(self.layers[0].module_string +" {" + self.layers[0].label + "} " + self.layers[0].label+".stim" + "\n")
        fileo.write("\n")
        self.layers[0].evs_files.append("node_"+self.layers[0].label+".evs")

        # Check if the output layer is softmax
        if self.layers[-1].module_string =="module_softmax":
            fileo.write("source " + " {" "softmax_input" + "} " + "softmax_input" + ".stim" + "\n")
            fileo.write("\n")

        for n in range(1,len(self.layers)):
            # CONVOLUTIONAL AND AVERAGE POOLING MODULES
            if self.layers[n].module_string == 'module_conv':
                for f in range(self.layers[n].num_of_FMs):
                    if self.layers[n].n_in_ports == 1:
                        # check if the presynaptic population is the input layer
                        if n==1:
                            pre_label_node = self.layers[n - 1].label
                        else:
                            pre_label_node = self.layers[n-1].label+"_"+str(f)

                        buildline = self.layers[n].module_string + " {" + pre_label_node + "}" + "{" + \
                                    self.layers[n].label+"_"+str(f) + "} " + self.layers[n].label+"_"+str(f) + ".prm" + " " + self.layers[
                                        n].label + ".stt"
                    else:
                        num_pre_nodes_in = self.layers[n].n_in_ports
                        pre_label = self.layers[n-1].label
                        build_in_nodes = ",".join([pre_label+"_"+str(x) for x in range(num_pre_nodes_in)])
                        buildline = self.layers[n].module_string + " {" + build_in_nodes+ "}" + "{" + \
                                self.layers[n].label + "_" + str(f) + "} " + self.layers[n].label + "_" + str(
                        f) + ".prm" + " " + self.layers[
                                    n].label + ".stt"

                    fileo.write(buildline + "\n")
                   # list to hold the filename of the events that will be generated
                    self.layers[n].evs_files.append("node_" + self.layers[n].label +"_"+str(f)+ ".evs")

                fileo.write("\n")

            # FLATTEN MODULE
            elif self.layers[n].module_string == 'module_flatten':
                num_pre_nodes_in = self.layers[n].n_in_ports
                pre_label_node = self.layers[n-1].label
                post_label_node = self.layers[n].label
                build_in_nodes = ",".join([pre_label_node + "_" + str(x) for x in range(num_pre_nodes_in)])
                buildline = self.layers[n].module_string + " {" + build_in_nodes + "}" + "{" + \
                            post_label_node + "} " +post_label_node + ".prm" + " " + post_label_node + ".stt"
                fileo.write(buildline + "\n")
                fileo.write("\n")
                # list to hold the filename of the events that will be generated
                self.layers[n].evs_files.append("node_" + post_label_node + ".evs")

            # FULLY CONNECTED MODULES
            elif self.layers[n].module_string == 'module_fully_connected' or self.layers[n].module_string =="module_softmax":
                #check if previous layer is flatten
                pre_label_node = self.layers[n-1].label
                if self.layers[n].module_string =="module_softmax":
                    buildline = self.layers[n].module_string + " {" +self.layers[n-1].label+",softmax_input" + "}" + "{" + \
                                    self.layers[n].label + "} " + self.layers[n].label + ".prm" + " " + self.layers[
                                        n].label + ".stt"
                else:
                    buildline = self.layers[n].module_string + " {" +self.layers[n-1].label + "}" + "{" + \
                                    self.layers[n].label + "} " + self.layers[n].label + ".prm" + " " + self.layers[
                                        n].label + ".stt"
                fileo.write(buildline + "\n")
                fileo.write("\n")
                # list to hold the filename of the events that will be generated
                self.layers[n].evs_files.append("node_" + self.layers[n].label + ".evs")

        fileo.write("\n")

        fileo.write(".options" + "\n")
        fileo.write("Tmax=" + str(int(settings['duration'])) + "\n")
        fileo.close()

    def clean_megasim_sim_data(self):
        '''
        A method that removes the previous stimulus file and generated event files before
        testing a new digit

        Returns
        -------

        '''
        files = os.listdir(self.megadirname)
        evs_data = [x for x in files if x[-3:]=='evs']
        stim_data = [x for x in files if x[-4:]=='stim']

        for evs in evs_data:
            os.remove(self.megadirname+evs)

        for stim in stim_data:
            os.remove(self.megadirname+stim)


    def store(self):
        '''
        Not needed since megasim always stores the simulation files, params and schematics in the
        self.megadirname
        '''
        pass

    def get_spikes(self, ):
        '''
        Method that fetches all the events from all layers after a simulation is over

        Returns: a list of all the events from all the layers
        -------

        '''
        events = []
        for l in self.layers:
            for fevs in l.evs_files:
                events.append(
                    np.genfromtxt(self.megadirname + fevs, delimiter=" ", dtype="int")
                )

        return events

    def spike_count_histogram(self, events, pop_size=10):
        '''
        This method first creates a histogram based on the size of the layer and then
        returns the argmax of the neuron that fired the most spikes for that particular stimulus.

        If there are no spikes it will return -1

        Parameters
        ----------
        events: list
            List of megasim events of a particular layer

        pop_size: int
            Size of the fully connected module

        Returns
        -------

        '''
        try:
            pop_spike_hist = np.histogram(events[:, 3], bins=pop_size,range=(0,pop_size))[0]
            pop_spike_hist = np.argmax(pop_spike_hist)
        except(IndexError):
            pop_spike_hist = -1# np.zeros((1,pop_size))
            import pdb;pdb.set_trace()
        return pop_spike_hist

    def run(self, snn_precomp, X_test, Y_test):
        """
        Simulate a spiking network with IF units and Poisson input in pyNN,
        using a simulator like Brian, NEST, NEURON, etc.

        This function will randomly select ``settings['num_to_test']`` test
        samples among ``X_test`` and simulate the network on those.

        Alternatively, a list of specific input samples can be given to the
        toolbox GUI, which will then be used for testing.

        If ``settings['verbose'] > 1``, the simulator records the
        spiketrains and membrane potential of each neuron in each layer, for
        the last sample.

        This is somewhat costly in terms of memory and time, but can be useful
        for debugging the network's general functioning.

        Parameters
        ----------

        snn_precomp: SNN
            The converted spiking network, before compilation (i.e. independent
            of simulator).
        X_test : float32 array
            The input samples to test. With data of the form
            (channels, num_rows, num_cols), X_test has dimension
            (num_samples, channels*num_rows*num_cols) for a multi-layer
            perceptron, and (num_samples, channels, num_rows, num_cols) for a
            convolutional net.
        Y_test : float32 array
            Ground truth of test data. Has dimension (num_samples, num_classes)

        Returns
        -------

        total_acc : float
            Number of correctly classified samples divided by total number of
            test samples.

        """

        from snntoolbox.io_utils.plotting import plot_confusion_matrix

        results = []
        guesses = []
        truth = []

        # Iterate over the number of samples to test
        for test_num in range(settings['num_to_test']):
            # If a list of specific input samples is given, iterate over that,
            # and otherwise pick a random test sample from among all possible
            # input samples in X_test.
            si = settings['sample_indices_to_test']
            #ind = randint(0, len(X_test) - 1) if si == [] else si[test_num]

            # Go through all the test set digits in the same order
            ind = test_num#range(settings['num_to_test'])

            # Clean any previous data. This is not necessary, only for debugging
            self.clean_megasim_sim_data()
            self.spikemonitors = []

            # Add Poisson input.
            if settings['verbose'] > 1:
                echo("Creating poisson input...\n")

            # Generate stimulus file
            echo("Using the same random seed for debugging\n")
            np.random.seed(1)
            if settings['poisson_input']:
                self.poisson_spike_generator_megasim(mnist_digit=X_test[ind, :])
            else:
                print("Only Poisson input supported")
                sys.exit(66)

            # Generate control events for the softmax module if it exists
            if self.layers[-1].module_string == "module_softmax":
                self.layers[-1].build_softmax_conrol_events(self.megadirname)

            # Run simulation for 'duration'.
            if settings['verbose'] > 1:
                echo("Starting new simulation...\n")

            #TODO this is ugly, in python3 i have to change folders to execute megasim
            current_dir = os.getcwd()
            os.chdir(self.megadirname)
            run_megasim = subprocess.check_output([MEGASIM_PATH + "megasim", self.megaschematic])
            os.chdir(current_dir)

            # Check megasim output for errors
            self.check_megasim_output(run_megasim)

            # use this if you want to access all the generated events
            spike_monitors = self.get_spikes()

            # list per input digit, a list per layer
            self.spikemonitors.append(spike_monitors)

            # use this to access spikes from a particular layer eg output
            #spike_monitor = self.get_spikes_from_layer(layer)
            #import pdb;pdb.set_trace()
            output_pop_activity = self.spike_count_histogram(spike_monitors[-1], self.layers[-1].population_size)
            # Get result by comparing the guessed class (i.e. the index of the
            # neuron in the last layer which spiked most) to the ground truth.
            #import pdb;pdb.set_trace()

            guesses.append(output_pop_activity)
            truth.append(np.argmax(Y_test[ind, :]))
            results.append(guesses[-1] == truth[-1])

            print("truth = %d guess = %d"%(np.argmax(Y_test[ind, :]),output_pop_activity ))

            if settings['verbose'] > 0:
                echo("Sample {} of {} completed.\n".format(test_num + 1,
                     settings['num_to_test']))
                echo("Moving average accuracy: {:.2%}.\n".format(
                    np.mean(results)))

            if settings['verbose'] > 1 and \
                    test_num == settings['num_to_test'] - 1:
                echo("Simulation finished. Collecting results...\n")
                output_shapes = [x.output_shapes for x in self.layers[1:]]
                self.collect_plot_results(
                    self.layers, output_shapes, snn_precomp,
                    X_test[ind:ind+settings['batch_size']])

            # Reset simulation time and recorded network variables for next
            # run.
            if settings['verbose'] > 1:
                echo("Resetting simulator...\n")
            # Skip during last run so the recorded variables are not discarded
            if test_num < settings['num_to_test'] - 1:
                pass
                #self.snn.restore()
            if settings['verbose'] > 1:
                echo("Done.\n")

        total_acc = np.mean(results)

        if settings['verbose'] > 1:
            plot_confusion_matrix(truth, guesses,
                                  settings['log_dir_of_current_run'])

        s = '' if settings['num_to_test'] == 1 else 's'
        echo("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
             total_acc, settings['num_to_test'], s))


        return total_acc

    def end_sim(self):
        """ Clean up after simulation. """
        pass

    def save(self, path=None, filename=None):
        """ Write model architecture and parameters to disk. """
        print("MegaSim model is already saved at %s"%self.megadirname)
        pass

    def collect_plot_results(self, layers, output_shapes, ann, X_batch, idx=0):
        """
        Collect spiketrains of all ``layers`` of a net from one simulation run,
        and plot results.

        Plots include: Spiketrains, activations, spikerates, membrane
        potential, correlations.

        To visualize the spikerates, neurons in hidden layers are spatially
        arranged on a 2d rectangular grid, and the firing rate of each neuron
        on the grid is encoded by color.

        Membrane potential vs time is plotted for all except the input layer.

        The activations are obtained by evaluating the original ANN ``ann`` on
        a sample ``X_batch``. The optional integer ``idx`` represents the index
        of a specific sample to plot.

        The ``output shapes`` of each layer are needed to reshape the output of
        layers back to original form when plotting results (During conversion,
        all layers are flattened).

        """

        from snntoolbox.io_utils.plotting import output_graphs, plot_potential

        # Collect spiketrains of all layers, for the last test sample.
        vmem = []
        showLegend = False

        # Allocate a list 'spiketrains_batch' with the following specification:
        # Each entry in ``spiketrains_batch`` contains a tuple
        # ``(spiketimes, label)`` for each layer of the network (for the first
        # batch only, and excluding ``Flatten`` layers).
        # ``spiketimes`` is an array where the last index contains the spike
        # times of the specific neuron, and the first indices run over the
        # number of neurons in the layer:
        # (num_to_test, n_chnls*n_rows*n_cols, duration)
        # ``label`` is a string specifying both the layer type and the index,
        # e.g. ``'03Dense'``.
        spiketrains_batch = []

        num_of_samples = 1 # how many samples?
        results_from_input_sample = 0 # which digit to pick?
        plot_c = 1 # counter is needed because in megasim we can have multiple conv modules for one layer

        for l in range(1,len(self.layers)):
            lbl = self.layers[l].label


            if self.layers[l].module_string == 'module_fully_connected' or self.layers[l].module_string=='module_softmax':
                print(self.layers[l].label)
                tmp = self.spikemonitors[results_from_input_sample][plot_c]
                spiketrain = np.zeros((num_of_samples, self.layers[l].population_size,  int(settings['duration'] / settings['dt'])) )

                spikes_megasim = tmp#[results_from_input_sample]

                # get spike counts per neuron per time-step
                spiketrain[ 0, spikes_megasim[:,3] , spikes_megasim[:,0]] = spikes_megasim[:,0]

                spiketrains_batch.append([spiketrain,lbl])
                plot_c += 1

            elif self.layers[l].module_string == 'module_conv':
                removeme = []

                spiketrain = np.zeros((num_of_samples, self.layers[l].num_of_FMs, self.layers[l].fm_size[0],
                                       self.layers[l].fm_size[1], int(settings['duration'] / settings['dt'])))

                num_fm = self.layers[l].num_of_FMs

                for fm in range(num_fm):
                    tmp = self.spikemonitors[results_from_input_sample][plot_c]
                    spikes_megasim = tmp


                    neuron_x_addr = np.copy(spikes_megasim[:, 3])
                    neuron_y_addr = np.copy(spikes_megasim[:, 4])
                    neuron_ts = np.copy(spikes_megasim[:, 0])
                    test = np.zeros((self.layers[l].fm_size[0],self.layers[l].fm_size[1],int(settings['duration'] / settings['dt'])))
                    test[neuron_y_addr,neuron_x_addr,neuron_ts] =neuron_ts
                    removeme.append(test)

                    plot_c+=1


                for f in range(len(removeme)):
                    spiketrain[0, f, :] = removeme[f]
                spiketrains_batch.append([spiketrain, lbl])

            elif self.layers[l].module_string == 'module_flatten':
                # ignore the spikes from the flatten layer
                plot_c += 1

        print(len(spiketrains_batch))
        for ll in range(len(spiketrains_batch)):
            print(ll)
            ll=0
            #import pdb;pdb.set_trace()
            output_graphs(spiketrains_batch, ann, X_batch,
                      settings['log_dir_of_current_run'], ll)
