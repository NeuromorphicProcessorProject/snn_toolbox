# -*- coding: utf-8 -*-
"""Building and simulating spiking neural networks using MegaSim.

@author: Evangelos Stromatias
"""

from __future__ import division, absolute_import
# For compatibility with python2
from __future__ import print_function, unicode_literals

import os
import subprocess
import sys
from abc import ABCMeta, abstractmethod
from builtins import int, range

import numpy as np
from future import standard_library

from snntoolbox.simulation.utils import AbstractSNN

standard_library.install_aliases()

INT32_MAX = 2147483646


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
    n_in_ports = -1
    n_out_ports = 1
    delay_to_process = 0
    delay_to_ack = 0
    fifo_depth = 0
    n_repeat = 1
    delay_to_repeat = 15

    # Parameters for the conv module and avg pooling
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
        """
        dirname = the full path of the
        """
        f = open(dirname + self.label + ".stt", "w")
        f.write(".integers\n")
        f.write("time_busy_initial %d\n" % self.time_busy_initial)
        f.write(".floats\n")
        f.close()

    @abstractmethod
    def build_parameter_file(self, dirname):
        pass


class module_input_stimulus:
    """
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
        List of strings of the event filenames that will generated when a
        megasim simulation is over.

    """

    def __init__(self, label, pop_size):
        self.label = label
        self.pop_size = pop_size
        self.input_stimulus_file = "input_events.stim"
        self.module_string = "source"
        self.evs_files = []


class module_flatten(Megasim_base):
    """
    A class for the flatten megasim module. The flatten module is used to
    connect a 3D population to a
    1D population. eg A convolutional layer to a fully connected one.

    Parameters
    ----------

    layer_params: Keras layer
        Layer from parsed input model.

    input_ports: int
        Number of input ports (eg feature maps from the previous layer)

    fm_size: tuple
        Tuple of integers that holds the size of the feature maps from the
        previous layer

    Attributes
    ----------

    module_string: string
        String that holds the module name for megasim

    output_shapes: tuple
        Tuple that holds the shape of the output of the module. Used for the
        plotting.

    evs_files: list
        List of strings of the event filenames that will generated when a
        megasim simulation is over.

    """

    def __init__(self, layer_params, input_ports, fm_size):
        self.module_string = "module_flatten"
        self.label = layer_params.name
        self.output_shapes = layer_params.output_shape
        self.evs_files = []

        self.n_in_ports = input_ports

        self.Nx_array = fm_size[0]
        self.Ny_array = fm_size[1]

    def build_parameter_file(self, dirname):
        """

        """
        param1 = (
            """.integers
n_in_ports %d
n_out_ports %d
delay_to_process %d
delay_to_ack %d
fifo_depth %d
n_repeat %d
delay_to_repeat %d
""" % (self.n_in_ports, self.n_out_ports, self.delay_to_process,
       self.delay_to_ack, self.fifo_depth, self.n_repeat,
       self.delay_to_repeat))

        param_k = (
            """Nx_array %d
Ny_array %d
""" % (self.Nx_array,
       self.Ny_array))

        q = open(dirname + self.label + '.prm', "w")
        q.write(param1)

        for k in range(self.n_in_ports):
            q.write(param_k)

        q.write(".floats\n")
        q.close()


class Module_average_pooling(Megasim_base):
    """

    duplicate code with the module_conv class - TODO: merge them
    layer_params
    Attributes: ['label', 'layer_num', 'padding', 'layer_type', 'strides',
    'input_shape', 'output_shape', 'get_activ', 'pool_size']
    """

    def __init__(self, layer_params, neuron_params, reset_input_event=False,
                 scaling_factor=10000000):
        self.uses_biases = False
        if reset_input_event:
            self.module_string = 'module_conv_NPP'
        else:
            self.module_string = 'module_conv'

        self.layer_type = layer_params.__class__.__name__
        self.output_shapes = layer_params.output_shape  # (none, 32, 26, 26)
        # last two
        self.label = layer_params.name
        self.evs_files = []
        self.reset_input_event = reset_input_event

        self.n_in_ports = 1  # one average pooling layer per conv layer
        # self.in_ports = 1 # one average pooling layer per conv layer
        self.num_of_FMs = layer_params.input_shape[1]

        self.fm_size = layer_params.output_shape[2:]
        self.Nx_array, self.Ny_array = self.fm_size[1] * 2, self.fm_size[0] * 2
        self.Dx, self.Dy = 0, 0
        self.crop_xmin, self.crop_xmax = 0, self.fm_size[0] * 2 - 1
        self.crop_ymin, self.crop_ymax = 0, self.fm_size[1] * 2 - 1
        self.xshift_pre, self.yshift_pre = 0, 0

        self.strides = layer_params.strides

        self.num_pre_modules = layer_params.input_shape[1]

        self.scaling_factor = int(scaling_factor)

        self.THplus = neuron_params["v_thresh"] * self.scaling_factor
        self.THminus = -2147483646
        self.refractory = neuron_params["tau_refrac"]
        self.MembReset = neuron_params["v_reset"] * self.scaling_factor
        self.TLplus = 0
        self.TLminus = 0

        self.kernel_size = (1, 1)  # layer_params.pool_size

        self.Reset_to_reminder = 0
        if neuron_params["reset"] == 'Reset to zero':
            self.Reset_to_reminder = 0
        else:
            self.Reset_to_reminder = 1

        self.pre_shapes = layer_params.input_shape  # (none, 1, 28 28) # last 2

        self.padding = layer_params.padding
        if self.padding != 'valid':
            print("Not implemented yet!")
            sys.exit(88)

        if self.reset_input_event:
            self.n_in_ports += 1

    def build_parameter_file(self, dirname):
        sc = self.scaling_factor
        fm_size = self.fm_size
        num_FMs = self.num_pre_modules

        print(
            "building %s with %d FM receiving input from %d pre pops. FM size is %d,%d" % (
                self.label, self.output_shapes[1], self.pre_shapes[1],
                self.output_shapes[2], self.output_shapes[3]))

        kernel = np.ones(self.kernel_size, dtype="float") * sc
        kernel *= (
        (1.0 / np.sum(self.kernel_size)))  # /(np.sum(self.kernel_size)))
        kernel = kernel.astype("int")

        for f in range(num_FMs):
            fm_filename = self.label + "_" + str(f)
            self.__build_single_fm(self.n_in_ports, self.n_out_ports, fm_size,
                                   kernel, dirname, fm_filename)
        pass

    def __build_single_fm(self, num_in_ports, num_out_ports, fm_size, kernel,
                          dirname, fprmname):
        """

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

        """
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
""" % (num_in_ports, num_out_ports, self.delay_to_process, self.delay_to_ack,
       self.fifo_depth,
       self.n_repeat, self.delay_to_repeat,
       self.Nx_array, self.Ny_array, self.Xmin, self.Ymin,
       self.THplus, self.THplusInfo, self.THminus, self.THminusInfo,
       self.Reset_to_reminder, self.MembReset, self.TLplus, self.TLminus,
       self.Tmin, self.T_Refract))

        param_k = (
            """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (self.kernel_size[0],
       self.kernel_size[1],
       self.Dx, self.Dy))

        kernels_list = []
        for k in range(1):
            # scale the weights
            w = kernel

            np.savetxt(dirname + "w.txt", w, delimiter=" ", fmt="%d")
            q = open(dirname + "w.txt")
            param2 = q.readlines()
            q.close()
            os.remove(dirname + "w.txt")
            kernels_list.append(param2)

        if self.reset_input_event:
            param_reset1 = (
                """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (1,
       1,
       0, 0
       ))
            param_reset2 = " ".join([str(x) for x in [0] * 1])

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
       2, 2,  # self.kernel_size[0], self.kernel_size[1],
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
        if self.reset_input_event:
            q.write(param_reset1)
            q.write(param_reset2)
            q.write("\n")
        q.write(param5)
        q.close()


class Module_conv(Megasim_base):
    """
    A class for the convolutional megasim module.

    Parameters
    ----------

    layer_params: Keras layer
        Layer from parsed input model.

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
        Tuple that holds the shape of the output of the module. Used for the
        plotting.

    evs_files: list
        List of strings of the event filenames that will generated when a
        megasim simulation is over.

    num_of_FMs: int
        Number of feature maps in this layer

    w: list
        list of weights

    padding: string
        String with the border mode used for the convolutional layer. So far
        only the valid mode is implemented


    layer_params
    Attributes: ['kernel_size', 'activation', 'layer_type', 'layer_num',
    'filters', 'output_shape', 'input_shape', 'label', 'parameters', 'padding']
    """

    def __init__(self, layer_params, neuron_params, flip_kernels=True,
                 reset_input_event=False, scaling_factor=10000000):
        if reset_input_event:
            self.module_string = 'module_conv_NPP'
        else:
            self.module_string = 'module_conv'

        self.layer_type = layer_params.__class__.__name__
        self.output_shapes = layer_params.output_shape  # (none, 32, 26, 26) last two
        self.label = layer_params.name
        self.evs_files = []
        self.reset_input_event = reset_input_event

        # self.size_of_FM = 0
        self.w = layer_params.get_weights()[0]
        self.num_of_FMs = self.w.shape[3]
        self.kernel_size = self.w.shape[:2]  # (kx, ky)
        try:
            self.b = layer_params.get_weights()[1]
            if np.nonzero(self.b)[0].size != 0:
                self.uses_biases = True
                print("%s uses biases" % self.module_string)
            else:
                self.uses_biases = False
                print("%s does not use biases" % self.module_string)
        except IndexError:
            self.uses_biases = False
            print("%s does not use biases" % self.module_string)

        self.n_in_ports = self.w.shape[2]
        self.pre_shapes = layer_params.input_shape  # (none, 1, 28 28) # last 2
        self.fm_size = self.output_shapes[2:]
        self.Nx_array = self.output_shapes[2:][1]
        self.Ny_array = self.output_shapes[2:][0]

        self.padding = layer_params.padding  # 'same', 'valid',

        self.Reset_to_reminder = 0
        if neuron_params["reset"] == 'Reset to zero':
            self.Reset_to_reminder = 0
        else:
            self.Reset_to_reminder = 1

        if self.padding == 'valid':
            # if its valid mode
            self.Nx_array = self.output_shapes[2:][1] + self.kernel_size[1] - 1
            self.Ny_array = self.output_shapes[2:][0] + self.kernel_size[0] - 1
            self.xshift_pre, self.yshift_pre = -int(
                self.kernel_size[1] / 2), -int(self.kernel_size[0] / 2)
            self.crop_xmin, self.crop_xmax = int(self.kernel_size[1] / 2), (
            self.Nx_array - self.kernel_size[1] + 1)
            self.crop_ymin, self.crop_ymax = int(self.kernel_size[0] / 2), (
            self.Ny_array - self.kernel_size[0] + 1)
        else:
            print("Not implemented yet!")
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

        if self.reset_input_event:
            self.n_in_ports += 1
        if self.uses_biases:
            self.n_in_ports += 1

    def build_parameter_file(self, dirname):
        fm_size = self.output_shapes[2:]
        pre_num_ports = self.pre_shapes[1]
        num_FMs = self.output_shapes[1]
        print("building %s with %d FM receiving input from %d pre pops. FM "
              "size is %d,%d" % (self.label, self.output_shapes[1],
                                 self.pre_shapes[1], self.output_shapes[2],
                                 self.output_shapes[3]))

        for f in range(num_FMs):
            fm_filename = self.label + "_" + str(f)
            kernel = self.w[:, :, :, f]
            if self.uses_biases:
                bias = self.b[f]
            else:
                bias = 0.0

            self.__build_single_fm(pre_num_ports, 1, fm_size, kernel, bias,
                                   dirname, fm_filename)
        pass

    def __build_single_fm(self, num_in_ports, num_out_ports, fm_size, kernel,
                          bias, dirname, fprmname):
        """
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

        """
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
""" % (self.n_in_ports, self.n_out_ports,
       self.delay_to_process, self.delay_to_ack, self.fifo_depth, self.n_repeat,
       self.delay_to_repeat,
       self.Nx_array, self.Ny_array,
       self.Xmin, self.Ymin,
       self.THplus, self.THplusInfo,
       self.THminus, self.THminusInfo,
       self.Reset_to_reminder, self.MembReset,
       self.TLplus, self.TLminus, self.Tmin, self.T_Refract))

        param_k = (
            """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (self.kernel_size[0],
       self.kernel_size[1],
       -int(self.kernel_size[0] / 2), -int(self.kernel_size[1] / 2)
       ))

        kernels_list = []
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

        if self.uses_biases:
            param_biases1 = (
                """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (self.Nx_array,
       self.Ny_array,
       0, 0
       ))
            b = np.ones((self.Nx_array, self.Ny_array)) * int(bias * sc)
            np.savetxt(dirname + "b.txt", b, delimiter=" ", fmt="%d")
            q = open(dirname + "b.txt")
            param_biases2 = q.readlines()
            q.close()
            os.remove(dirname + "b.txt")

        if self.reset_input_event:
            param_reset1 = (
                """Nx_kernel %d
Ny_kernel %d
Dx %d
Dy %d
""" % (1,
       1,
       0, 0
       ))
            param_reset2 = " ".join([str(x) for x in [0] * 1])

        # if self.label == "02Conv2D_32x24x24":
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
            for i in kernels_list[k]:  # param2:
                q.write(i)
        if self.uses_biases:
            q.write(param_biases1)
            for i in param_biases2:
                q.write(i)

        if self.reset_input_event:
            q.write(param_reset1)
            q.write(param_reset2)
            q.write("\n")
        q.write(param5)
        q.close()


class Module_fully_connected(Megasim_base):
    """
    A class for the fully connected megasim module.

    Parameters
    ----------

    layer_params: Keras layer
        Layer from parsed input model.

    neuron_params: dictionary
        This is the settings dictionary that is set in the config.py module

    scaling_factor: int
        An integer that will be used to scale all parameters.

    enable_softmax: Boolean
        A flag that if set will use (if the ann uses it) softmax for the
        output layer. If not set
        a population of LIF neurons will be used instead.

    Attributes
    ----------

    module_string: string
        String that holds the module name for megasim

    output_shapes: tuple
        Tuple that holds the shape of the output of the module. Used for the
        plotting.

    evs_files: list
        List of strings of the event filenames that will generated when a
        megasim simulation is over.

    num_of_FMs: int
        Number of feature maps in this layer

    w: list
        list of weights

    padding: string
        String with the border mode used for the convolutional layer. So far
        only the valid mode is implemented


    layer_params
    Attributes: ['kernel_size', 'activation', 'layer_type', 'layer_num',
    'filters', 'output_shape', 'input_shape', 'label', 'parameters', 'padding']
    """

    def __init__(self, layer_params, neuron_params, scaling_factor=10000000,
                 reset_input_event=False, enable_softmax=True):

        if reset_input_event:
            self.module_string = 'module_fully_connected_NPP'
        else:
            self.module_string = 'module_fully_connected'
        print(self.module_string)
        self.label = layer_params.name
        self.output_shapes = layer_params.output_shape
        self.evs_files = []

        self.population_size = layer_params.output_shape[1]
        self.scaling_factor = int(scaling_factor)
        self.w = layer_params.get_weights()[0]
        try:
            self.b = layer_params.get_weights()[1]
            if np.nonzero(self.b)[0].size != 0:
                self.uses_biases = True
                print("%s uses biases" % self.module_string)
            else:
                self.uses_biases = False
                print("%s does not use biases" % self.module_string)
        except IndexError:
            self.uses_biases = False
            print("%s does not use biases" % self.module_string)

        self.Nx_array_pre = len(self.w)

        self.enable_softmax = enable_softmax
        self.reset_input_event = reset_input_event

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
        if layer_params.activation == 'softmax' and self.enable_softmax:
            print("Using softmax for the output layer")
            self.module_string = 'module_softmax'
            self.n_in_ports = 2
        else:
            print("Using LIF")
            self.n_in_ports = 1

        if self.reset_input_event:
            self.n_in_ports += 1

        if self.uses_biases:
            self.n_in_ports += 1

    def build_parameter_file(self, dirname):
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
""" % (self.n_in_ports, self.n_out_ports,
       self.delay_to_process,
       self.delay_to_ack, self.fifo_depth, self.n_repeat, self.delay_to_repeat,
       self.population_size, 1, self.Xmin, self.Ymin,
       self.THplus, self.THplusInfo,
       self.THminus, self.THminusInfo,
       self.Reset_to_reminder, self.MembReset,
       self.TLplus, self.TLminus, self.Tmin, self.T_Refract, self.Nx_array_pre))

        w = self.w * sc

        # TODO: change these lines
        np.savetxt(dirname + "w.txt", w, delimiter=" ", fmt="%d")
        q = open(dirname + "w.txt")
        param2 = q.readlines()
        q.close()
        os.remove(dirname + "w.txt")

        if self.uses_biases:
            param_biases1 = (
                """Nx_array_pre 1
Ny_array_pre 1
"""
            )
            param_biases2 = " ".join([str(int(x * sc)) for x in self.b])

        # if the output activation is softmax add one more input for the control in events
        if self.module_string == 'module_softmax':
            param_softmax2 = " ".join(
                [str(x) for x in [0] * self.population_size])
            param_softmax1 = (
                """Nx_array_pre 1
Ny_array_pre 1
"""
            )
        if self.reset_input_event:
            param_reset2 = " ".join(
                [str(x) for x in [0] * self.population_size])
            param_reset1 = (
                """Nx_array_pre 1
Ny_array_pre 1
""")

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
    """ % (self.crop_xmin, self.crop_xmax, self.crop_ymin, self.crop_ymax,
           self.xshift_pre,
           self.yshift_pre, self.x_subsmp, self.y_subsmp, self.xshift_pos,
           self.yshift_pos, self.rectify))

        q = open(dirname + self.label + '.prm', "w")
        q.write(param1)
        for i in param2:
            q.write(i)

        # if we use a softmax use 0 weights for the control events
        if self.uses_biases:
            q.write(param_biases1)
            q.write(param_biases2)
            q.write("\n")
        if self.module_string == 'module_softmax':
            q.write(param_softmax1)
            q.write(param_softmax2)
            q.write("\n")
        if self.reset_input_event:
            q.write(param_reset1)
            q.write(param_reset2)
            q.write("\n")
        q.write(param5)
        q.close()

    def build_softmax_conrol_events(self, megadirname, duration, dt,
                                    input_rate, softmax_clockrate=300):
        softmax_in_events = []
        for t in range(0, int(duration / dt)):
            rnd = np.random.uniform(0, input_rate)
            if rnd < softmax_clockrate:
                softmax_in_events.append([t, -1, -1, 0, -1, -1])
        softmax_in_events = np.asarray(softmax_in_events)
        np.savetxt(megadirname + "softmax_input.stim", softmax_in_events,
                   delimiter=" ", fmt="%d")


# ---------------------------------------------------------------------------- #


class SNN(AbstractSNN):
    """
    Represents the compiled spiking neural network, ready for testing in a
    spiking simulator.

    Attributes
    ----------

    layers: list
        Each entry represents a layer.

    connections: list
        The connections between layers.

    megasim_path: str
        The path to megasim installation directory.

    megaschematic: str
        String that holds megasim main schmatic file that is needed to run a
        simulation

    megadirname: str
        String that holds the full path where the generated files for a megasim
        simulation are stored. These files include the stimulus, parameter,
        state and schematic files. The event files will be generated in the same
        folder.

    input_stimulus_file: str
        Filename of input stimulus.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons.

    use_biases: bool
        Whether or not to use biases.
    """

    def __init__(self, config, queue=None):

        AbstractSNN.__init__(self, config, queue)

        self.layers = []
        self.connections = []
        self.megasim_path = self.sim.megasim_path()
        self.megadirname = ''
        self.megaschematic = 'megasim.sch'
        self.input_stimulus_file = "input_events.stim"
        self.cellparams = {'reset': config['cell']['reset'],
                           'tau_refrac': config.getfloat('cell', 'tau_refrac'),
                           'v_reset': config.getfloat('cell', 'v_reset'),
                           'v_thresh': config.getfloat('cell', 'v_thresh')}
        self.use_biases = None

        if self.batch_size > 1:
            self.reset_signal_event = True
            print("Batch mode used, reset signal set")
        else:
            self.reset_signal_event = False
            print("Symbol by Symbol operation")
        self.scaling_factor = self.config.getint('cell', 'scaling_factor')

    @property
    def is_parallelizable(self):
        return True

    def add_input_layer(self, input_shape):

        self.layers.append(module_input_stimulus(label='InputLayer',
                                                 pop_size=input_shape[1:]))

        # Create megasim dir where it will store the SNN params and schematic
        self.megadirname = self.config['paths']['path_wd'] + "/MegaSim_" + \
            self.config['paths']['filename_ann'] + '/'
        # clear the folder first from evs and log files
        # TODO: add a method to clean everything in that folder
        if not os.path.exists(self.megadirname):
            os.makedirs(self.megadirname)

    def add_layer(self, layer):
        pass

    def build_dense(self, layer):

        enable_softmax = True if layer.activation == 'softmax' else False

        self.layers.append(Module_fully_connected(
            layer, self.cellparams, self.scaling_factor,
            self.reset_signal_event, enable_softmax))

    def build_convolution(self, layer):

        self.layers.append(Module_conv(
            layer, self.cellparams, reset_input_event=self.reset_signal_event,
            scaling_factor=self.scaling_factor))

    def build_pooling(self, layer):
        if layer.__class__.__name__ == 'MaxPooling2D':
            import warnings

            warnings.warn("Layer type 'MaxPooling' not supported yet. " +
                          "Falling back on 'AveragePooling'.", RuntimeWarning)

        self.layers.append(Module_average_pooling(
            layer, self.cellparams, self.reset_signal_event,
            self.scaling_factor))

    def build_flatten(self, layer):
        c_layer_len = len(self.layers) - 1
        self.layers.append(module_flatten(
            layer, self.layers[c_layer_len].num_of_FMs,
            self.layers[c_layer_len].fm_size))

    def compile(self):
        # Build parameter files for all modules, ignoring the input layer.
        for mod_n, mod in enumerate(self.layers[1:]):
            mod.build_parameter_file(self.megadirname)
            mod.build_state_file(self.megadirname)

        # build MegaSim Schematic file
        self.build_schematic_updated()

        print("Compilation finished. Model is stored at {}.\n".format(
            self.megadirname))

    def simulate(self, **kwargs):
        if self._poisson_input:
            np.random.seed(1)
            timestamp_batches = self.poisson_spike_generator_batchmode_megasim(
                kwargs['x_b_l'])
            # Generate control events for the softmax module if it exists
            if self.layers[-1].module_string == "module_softmax":
                self.layers[-1].build_softmax_conrol_events(
                    self.megadirname, self._duration, self._dt,
                    self.config('input', 'input_rate'))
        else:
            print("Only Poisson input supported")
            sys.exit(66)

        # check if biases are used and generate timestamps to apply them
        if self.use_biases:
            self.generate_bias_clk(timestamp_batches)

        current_dir = os.getcwd()
        os.chdir(self.megadirname)
        run_megasim = subprocess.check_output([self.megasim_path + "megasim",
                                               self.megaschematic])
        os.chdir(current_dir)

        # Check megasim output for errors
        self.check_megasim_output(str(run_megasim))

        output_b_l_t = self.get_recorded_vars(self.layers)

        return output_b_l_t

    def reset(self, sample_idx):

        # Clean any previous data. This is not necessary, only for debugging
        self.clean_megasim_sim_data()

    def end_sim(self):

        pass

    def save(self, path, filename):

        print("MegaSim model is already saved at %s" % self.megadirname)

    def load(self, path, filename):

        raise NotImplementedError

    def get_spiketrains(self, **kwargs):

        layer = kwargs['layer']

        # reset_ts marks the time when a new sample in the batch was simulated.
        reset_ts = [0] + list(np.array(np.genfromtxt(
            self.megadirname + "reset_event.stim", 'int', delimiter=" "),
            ndmin=2)[:, 0])

        try:
            spiketrains_b_l_t = np.zeros(list(layer.output_shapes) +
                                         [self._num_timesteps])
        except AttributeError:
            return

        module_string = layer.module_string

        if module_string in {'module_fully_connected', 'module_softmax',
                             'module_fully_connected_NPP'}:
            for i in range(self.batch_size):
                t_first = reset_ts[i]
                t_last = reset_ts[i + 1]
                events = np.genfromtxt(self.megadirname + layer.evs_files[0],
                                       delimiter=" ", dtype="int")
                # e == [timestamp, ?, ?, target address, polarity]
                for e in events:
                    t = e[0]
                    if t < t_first or t >= t_last:
                        continue
                    t -= t_first + i
                    spiketrains_b_l_t[i, e[3], t] = t
        elif module_string in {'module_conv', 'module_conv_NPP'}:
            # Convolutional and Average pooling layers

            for i in range(self.batch_size):
                t_first = reset_ts[i]
                t_last = reset_ts[i + 1]
                # There is one event file for each feature map.
                for f, event_file in enumerate(layer.evs_files):
                    events = np.genfromtxt(self.megadirname + event_file,
                                           delimiter=" ", dtype="int")
                    # e == [timestamp, ?, ?, x-addr, y-addr, polarity]
                    for e in events:
                        t = e[0]
                        if t < t_first or t >= t_last:
                            continue
                        t -= t_first + i
                        spiketrains_b_l_t[i, f, e[4], e[3], t] = t
        elif module_string == 'module_flatten':
            return
        else:
            return

        return spiketrains_b_l_t

    def get_spiketrains_input(self):
        # reset_ts marks the time when a new sample in the batch was simulated.
        reset_ts = [0] + list(np.array(np.genfromtxt(
            self.megadirname + "reset_event.stim", 'int', delimiter=" "),
            ndmin=2)[:, 0])

        layer = self.layers[0]

        spiketrains_b_l_t = np.zeros([self.batch_size, 1] + list(layer.pop_size) +
                                     [self._num_timesteps])

        # TODO: This part has not been tested. Input spikes are probably not
        # counted in self.operations_b_t.
        try:
            for i in range(self.batch_size):
                t_first = reset_ts[i]
                t_last = reset_ts[i + 1]
                # We assume here there is one event file for each input channel.
                for f, event_file in enumerate(layer.evs_files):
                    events = np.genfromtxt(self.megadirname + event_file,
                                           delimiter=" ", dtype="int")
                    # e == [timestamp, ?, ?, x-addr, y-addr, polarity]
                    for e in events:
                        t = e[0]
                        if t < t_first or t >= t_last:
                            continue
                        spiketrains_b_l_t[i, f, e[4], e[3], t] = t
        except IndexError:
            return spiketrains_b_l_t

    def get_vmem(self, **kwargs):
        return None

    def get_spikes(self):
        """
        Method that fetches all the events from all layers after a simulation
        is over.

        Returns
        -------

        A list of all the events from all the layers.

        """

        events = []
        for l in self.layers:
            for fevs in l.evs_files:
                events.append(np.genfromtxt(self.megadirname + fevs,
                                            delimiter=" ", dtype="int"))
        return events

    def get_output_spikes_batch(self):
        """Method that fetches the events from the output layer.

        Returns
        -------

        A numpy array of the output events.

        """

        outspikes_per_symbol = []
        output_events = np.genfromtxt(
            self.megadirname + self.layers[-1].evs_files[0], delimiter=" ",
            dtype="int")
        reset_events = np.genfromtxt(self.megadirname + "reset_event.stim",
                                     delimiter=" ", dtype="int")
        reset_ts = reset_events[:, 0]
        start = 0

        for i in range(len(reset_ts)):
            stop = reset_ts[i]
            indices = np.where(np.logical_and(output_events[:, 0] >= start,
                                              output_events[:, 0] < stop))

            outspikes_per_symbol.append(output_events[indices])
            start = stop
        return outspikes_per_symbol

    @staticmethod
    def spike_count_histogram(events, pop_size=10):
        """
        This method first creates a histogram based on the size of the layer
        and then returns the argmax of the neuron that fired the most spikes for
        that particular stimulus.

        If there are no spikes it will return -1.

        Parameters
        ----------

        events: list
            List of megasim events of a particular layer

        pop_size: int
            Size of the fully connected module

        Returns
        -------

        pop_spike_hist

        """

        try:
            pop_spike_hist = np.histogram(events[:, 3], pop_size,
                                          (0, pop_size))[0]
            pop_spike_hist = np.argmax(pop_spike_hist)
        except IndexError:
            pop_spike_hist = -1  # np.zeros((1,pop_size))
            import pdb
            pdb.set_trace()
        return pop_spike_hist

    @staticmethod
    def check_megasim_output(megalog):
        """

        A method that checks the prints of MegaSim for errors

        Parameters
        ----------

        megalog: str
            String returned from executing megasim.
        """

        megalog = str(megalog)
        megalog = megalog.lower()

        if megalog.find("error") > 0:
            print("MegaSim error: ")
            print(megalog)
            sys.exit(99)

    def poisson_spike_generator_megasim(self, mnist_digit):
        """

        Parameters
        ----------

        mnist_digit: ndarray
            A 1d or 2d numpy array of an mnist digit (normalised 0-1).

        Returns
        -------

        It will store the generated spike trains to a stimulus file in the
        megasim sim folder.
        """

        spikes = []
        for t in np.arange(0, self._duration, self._dt):
            # Create poisson input.
            spike_snapshot = np.random.random_sample(mnist_digit.shape) * \
                             self.rescale_fac
            inp = (spike_snapshot <= mnist_digit).astype('float32')
            # find the indexes of the non-zero
            neuron_id = list(np.nonzero(inp))
            # check if input is flattened or 2d in order to extract the X,Y
            # addresses correctly
            if len(inp.shape) == 1:
                megasim_x = neuron_id[0]
                megasim_y = 0
            else:
                megasim_x = neuron_id[2]
                megasim_y = neuron_id[1]

            spike_for_t = np.zeros((len(megasim_x), 6), dtype="int")
            spike_for_t[:, 0] = t  # time-stamps
            spike_for_t[:, 1] = -1  # REQ
            spike_for_t[:, 2] = -1  # ACK
            spike_for_t[:, 3] = megasim_x  # X address
            spike_for_t[:, 4] = megasim_y  # Y address
            spike_for_t[:, 5] = 1  # polarity
            spikes.append(spike_for_t)

        spikes = np.vstack(spikes)
        np.savetxt(self.megadirname + self.layers[0].label + ".stim", spikes,
                   delimiter=" ", fmt="%d")

    def poisson_spike_generator_batchmode_megasim(self, mnist_digits):
        """

        Parameters
        ----------

        mnist_digits: ndarray
            A 1d or 2d numpy array of an mnist digit (normalised 0-1).

        Returns
        -------

        It will store the generated spike trains to a stimulus file in the
        megasim sim folder.
        """

        spikes = []
        reset_events = []
        softmax_in_events = []

        timestamps = []

        last_ts = 0

        for i, digit in enumerate(mnist_digits):
            for t in np.arange(0, self._duration, self._dt):
                # Create poisson input.
                spike_snapshot = np.random.random_sample(digit.shape) * \
                                 self.rescale_fac
                inp = (spike_snapshot <= digit).astype('float32')
                # find the indexes of the non-zero
                neuron_id = list(np.nonzero(inp))
                # check if input is flattened or 2d in order to extract the X,Y
                # addresses correctly
                if len(inp.shape) == 1:
                    megasim_x = neuron_id[0]
                    megasim_y = 0
                else:
                    megasim_x = neuron_id[2]
                    megasim_y = neuron_id[1]

                spike_for_t = np.zeros((len(megasim_x), 6), dtype="int")
                spike_for_t[:, 0] = t + last_ts  # time-stamps
                spike_for_t[:, 1] = -1  # REQ
                spike_for_t[:, 2] = -1  # ACK
                spike_for_t[:, 3] = megasim_x  # X address
                spike_for_t[:, 4] = megasim_y  # Y address
                spike_for_t[:, 5] = 1  # polarity
                spikes.append(spike_for_t)

                # softmax control events
                rnd = np.random.uniform(
                    0, self.config.getint('input', 'input_rate'))
                if rnd < 300:
                    softmax_in_events.append([t + last_ts, -1, -1, 0, -1, -1])
            timestamps.append([last_ts, spike_for_t[-1][0]])
            last_ts = spike_for_t[-1][0] + 1  # (self._duration*(i+1)) + 1
            # reset control events
            reset_events.append([last_ts, -1, -1, 0, -2, -2])
            last_ts += 1
        if last_ts >= INT32_MAX:
            print("Timestamp larger than maximum 32bit integer, please use "
                  "smaller batch size")
            sys.exit(1)

        spikes = np.vstack(spikes)
        reset_events = np.asarray(reset_events, dtype="int")
        reset_events = np.vstack(reset_events)
        softmax_in_events = np.asarray(softmax_in_events)

        np.savetxt(self.megadirname + self.layers[0].label + ".stim", spikes,
                   delimiter=" ", fmt="%d")
        np.savetxt(self.megadirname + "reset_event.stim", reset_events,
                   delimiter=" ", fmt="%d")
        np.savetxt(self.megadirname + "softmax_input.stim", softmax_in_events,
                   delimiter=" ", fmt="%d")
        return timestamps

    def generate_bias_clk(self, timestamp_batches):
        """
        An external periodic (per timestep) event is used to trigger the
        biases, since megasim simulator is not a time-stepped simulator.

        Parameters
        ----------

        timestamp_batches: List[list]
            List that includes the first and last timestamps of the input
            source, e.g. [[start0, stop0], [start1, stop1]].

        Returns
        -------

        Generates a megasim stimulus file in the experiment folder.
        """

        bias_clk = []
        fname = "bias_clk.stim"
        for ts in timestamp_batches:
            for t in np.arange(ts[0], ts[1] + 1, self._dt):
                bias_clk.append([t, -1, -1, 0, 0, 1])

        np.savetxt(self.megadirname + fname, np.asarray(bias_clk),
                   delimiter=" ", fmt="%d")

    def build_schematic_updated(self):
        """

        This method generates the main MegaSim schematic file

        TODO: this method is quite ugly! need to refactor it
        99.20 non normalised first 100 samples = 100% reset to zero
        -------

        """

        use_biases = False
        bias_node = "bias_clk"
        for l in self.layers:
            try:
                if l.uses_biases:
                    use_biases = True
                    self.use_biases = True
            except AttributeError:
                pass

        reset_event_string = "reset_event"
        fileo = open(self.megadirname + self.megaschematic, "w")

        fileo.write(".netlist\n")
        # stim file first - node is input_evs
        fileo.write(
            self.layers[0].module_string + " {" + self.layers[0].label + "} " +
            self.layers[0].label + ".stim" + "\n")
        fileo.write("\n")
        self.layers[0].evs_files.append("node_" + self.layers[0].label + ".evs")

        if use_biases:
            fileo.write("source {bias_clk} bias_clk.stim")
            fileo.write("\n")

        # Check if the output layer is softmax
        if self.layers[-1].module_string == "module_softmax":
            fileo.write("source " + " {" "softmax_input" + "} " +
                        "softmax_input" + ".stim" + "\n")
            fileo.write("\n")

        # if we are in batch mode (with reset signal event)
        if self.reset_signal_event:
            fileo.write("source " + " {" + reset_event_string + "} " +
                        reset_event_string + ".stim" + "\n")
            fileo.write("\n")

        for n in range(1, len(self.layers)):
            # CONVOLUTIONAL AND AVERAGE POOLING MODULES
            # import  pdb;pdb.set_trace()
            if self.layers[n].module_string == 'module_conv' or \
                    self.layers[n].module_string == 'module_conv_NPP':
                for f in range(self.layers[n].num_of_FMs):
                    # if there is only one input or we use reset signals
                    if self.layers[n].n_in_ports == 1 or \
                            (self.layers[n].n_in_ports == 2 and use_biases) or \
                            (self.layers[n].n_in_ports == 3 and use_biases and
                             self.reset_signal_event):
                        # check if the presynaptic population is the input layer

                        if n == 1:
                            # import pdb;pdb.set_trace()
                            pre_label_node = self.layers[n - 1].label

                            if use_biases and self.layers[n].uses_biases:
                                pre_label_node = pre_label_node + "," + \
                                                 bias_node
                            if self.reset_signal_event:
                                pre_label_node = pre_label_node + "," + \
                                                 reset_event_string

                        else:
                            pre_label_node = self.layers[
                                                 n - 1].label + "_" + str(f)

                            if use_biases and self.layers[n].uses_biases:
                                pre_label_node = pre_label_node + "," + \
                                                 bias_node
                            if self.reset_signal_event:
                                pre_label_node = pre_label_node + "," + \
                                                 reset_event_string

                        buildline = self.layers[n].module_string + " {" + \
                            pre_label_node + "}" + "{" + \
                            self.layers[n].label + "_" + str(f) + "} " + \
                            self.layers[n].label + "_" + str(f) + ".prm" + \
                            " " + self.layers[n].label + ".stt"
                    else:
                        num_pre_nodes_in = self.layers[n].n_in_ports
                        pre_label = self.layers[n - 1].label
                        pre_nodes_str = [pre_label + "_" + str(x) for x in
                                         range(num_pre_nodes_in)]
                        # if we are in batch mode
                        if self.reset_signal_event:
                            if use_biases and self.layers[n].uses_biases:
                                pre_nodes_str[-2] = bias_node

                            pre_nodes_str[-1] = reset_event_string
                        else:
                            # if testing sample by sample we dont need to use
                            # reset signals
                            pre_nodes_str[-1] = bias_node

                        build_in_nodes = ",".join(pre_nodes_str)

                        buildline = self.layers[n].module_string + " {" + \
                            build_in_nodes + "}" + "{" + \
                            self.layers[n].label + "_" + str(f) + "} " + \
                            self.layers[n].label + "_" + str(f) + ".prm" + \
                            " " + self.layers[n].label + ".stt"

                    fileo.write(buildline + "\n")
                    # list to hold the filename of the events that will be
                    # generated
                    self.layers[n].evs_files.append(
                        "node_" + self.layers[n].label + "_" + str(f) + ".evs")

                fileo.write("\n")

            # FLATTEN MODULE
            elif self.layers[n].module_string == 'module_flatten':
                num_pre_nodes_in = self.layers[n].n_in_ports
                pre_label_node = self.layers[n - 1].label
                post_label_node = self.layers[n].label
                build_in_nodes = ",".join([pre_label_node + "_" + str(x) for x
                                           in range(num_pre_nodes_in)])
                buildline = self.layers[n].module_string + " {" + \
                    build_in_nodes + "}" + "{" + post_label_node + "} " + \
                    post_label_node + ".prm" + " " + post_label_node + ".stt"
                fileo.write(buildline + "\n")
                fileo.write("\n")
                # list to hold the filename of the events that will be generated
                self.layers[n].evs_files.append("node_" + post_label_node +
                                                ".evs")

            # FULLY CONNECTED MODULES
            elif self.layers[n].module_string == 'module_fully_connected' or \
                    self.layers[n].module_string == "module_softmax" or \
                    self.layers[n].module_string == \
                    "module_fully_connected_NPP":
                # check if previous layer is flatten
                if self.layers[n].module_string == "module_softmax":
                    pre_nodes = self.layers[n - 1].label  # + ",softmax_input"
                    if use_biases:
                        pre_nodes = pre_nodes + "," + bias_node

                    pre_nodes = pre_nodes + ",softmax_input"
                    if self.reset_signal_event:
                        pre_nodes = pre_nodes + "," + reset_event_string

                    buildline = self.layers[n].module_string + " {" + \
                        pre_nodes + "}" + "{" + self.layers[n].label + "} " + \
                        self.layers[n].label + ".prm" + " " + \
                        self.layers[n].label + ".stt"
                else:
                    pre_nodes = self.layers[n - 1].label

                    if use_biases:
                        pre_nodes = pre_nodes + "," + bias_node
                    if self.reset_signal_event:
                        pre_nodes = pre_nodes + "," + reset_event_string

                    buildline = self.layers[n].module_string + " {" + \
                        pre_nodes + "}" + "{" + self.layers[n].label + "} " + \
                        self.layers[n].label + ".prm" + " " + \
                        self.layers[n].label + ".stt"
                fileo.write(buildline + "\n")
                fileo.write("\n")
                # list to hold the filename of the events that will be generated
                self.layers[n].evs_files.append("node_" +
                                                self.layers[n].label + ".evs")

        fileo.write("\n")

        fileo.write(".options" + "\n")
        if self.reset_signal_event:
            fileo.write("Tmax=" + str(INT32_MAX) + "\n")
        else:
            fileo.write("Tmax=" + str(int(self._duration)) + "\n")
        fileo.close()

    def clean_megasim_sim_data(self):
        """
        A method that removes the previous stimulus file and generated event
        files before testing a new sample.
        """

        files = os.listdir(self.megadirname)
        evs_data = [x for x in files if x[-3:] == 'evs']
        stim_data = [x for x in files if x[-4:] == 'stim']

        for evs in evs_data:
            os.remove(self.megadirname + evs)

        for stim in stim_data:
            os.remove(self.megadirname + stim)

    def set_spiketrain_stats_input(self):
        # Added this here because PyCharm complains about not all abstract
        # methods being implemented (even though this is not abstract).
        AbstractSNN.set_spiketrain_stats_input(self)
