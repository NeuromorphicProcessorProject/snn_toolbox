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
from random import randint

from snntoolbox import echo
from snntoolbox.config import settings, initialize_simulator

standard_library.install_aliases()


MEGASIM_PATH = "/Users/Evangelos/Programming/NPP/megasim/megasim/bin/"

class module_fc():
    '''
    Helper class for the megasim fully connected module
    '''
    def __init__(self, pop_size, neuron_params, scale_weights_factor = 10000000):
        self.pop_size = pop_size
        self.module_string = 'module_fully_connected'
        self.scale_weights_factor = int(scale_weights_factor)
        self.w = []
        self.neuron_params = None
        self.label= None
        self.time_busy_initial = 0

        self.threshold = neuron_params["v_thresh"]
        self.threshold_negative = -2147483646
        self.refractory = neuron_params["tau_refrac"]
        self.vreset = neuron_params["v_reset"]
        self.leak_pos = 0
        self.leak_neg = 0

    def update_neuron_params(self):
        pass

    def build_state_file(self, dirname):
        f = open(dirname+self.label+".stt", "w")
        f.write(".integers\n")
        f.write("time_busy_initial %d\n" % self.time_busy_initial)
        f.write(".floats\n")
        f.close()

    def build_parameter_file(self, dirname):
        sc = self.scale_weights_factor
        #filename= "fc_con.prm"
        #Setting the values of the parameters
        neuroparams={
            "THplus": int(self.threshold * sc),
            "THplusInfo": 1,
            "THminus": self.threshold_negative,
            "THminusInfo": 0,
            "Reset_to_reminder": 0,
            "MembReset": int(self.vreset),
            "TLplus": int(self.leak_pos),
            "TLminus": int(self.leak_neg),
            "Tmin": 0,
            "T_Refract": int(self.refractory),
        }
        #Values of the genereal parameters
        generalparams={
            "n_out_ports": 1,
            "delay_to_process": 0,
            "delay_to_ack": 0,
            "fifo_depth": 1,
            "n_repeat": 1,
            "delay_to_repeat": 15,
        }
        #Values of the fc parameters
        fclayergeneralparams={
            "Pop_size": self.pop_size[0],
            "Ny_array": 1,
            "Xmin": 0,
            "Ymin": 0,
        }

        in_ports=1
        out_ports=1
        fan_in = len(self.w)

        if neuroparams["THplus"]>(2**31-1):
            print ("Threshold too high")
            sys.exit()
        print ("thres is %d"%neuroparams["THplus"])

        param1=(
    """.integers
n_in_ports %d
n_out_ports %d
delay_to_process 0
delay_to_ack 0
fifo_depth 0
n_repeat 1
delay_to_repeat 15
population_size %d
Ny_array %d
Xmin 0
Ymin 0
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
"""%(in_ports,out_ports,fclayergeneralparams["Pop_size"],fclayergeneralparams["Ny_array"],
            neuroparams["THplus"],neuroparams["THplusInfo"],neuroparams["THminus"],neuroparams["THminusInfo"],
            neuroparams["Reset_to_reminder"],neuroparams["MembReset"],neuroparams["TLplus"],neuroparams["TLminus"],
            neuroparams["Tmin"],neuroparams["T_Refract"],fan_in))

        w = self.w * sc

        np.savetxt(dirname+"w.txt",w,delimiter=" ",fmt="%d")
        q=open(dirname+"w.txt")
        param2=q.readlines()
        q.close()
        os.remove(dirname+"w.txt")
        param5=(
    """crop_xmin 0
crop_xmax 31
crop_ymin 0
crop_ymax 31
xshift_pre 0
yshift_pre 0
x_subsmp 1
y_subsmp 1
xshift_pos 0
yshift_pos 0
rectify 0
.floats
    """)
        q=open(dirname+self.label+'.prm',"w")
        q.write(param1)
        for i in param2:
            q.write(i)
        q.write(param5)
        q.close()

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
        Each entry represents a layer, i.e. a population of neurons, in form of
        Brian2 ``NeuronGroup`` objects.

    connections: list
        Brian2 ``Synapses`` objects representing the connections between
        individual layers.

    threshold: string
        Defines spiking threshold.

    reset: string
        Defines reset potential.

    eqs: string
        Differential equation for membrane potential.

    spikemonitors: list
        Brian2 ``SpikeMonitor`` s for each layer that records spikes.

    statemonitors: list
        Brian2 ``StateMonitor`` s for each layer that records membrane
        potential.

    labels: list
        The layer labels.

    output_shapes: list
        The output shapes of each layer. During conversion, all layers are
        flattened. Need output shapes to reshape the output of layers back to
        original form when plotting results later.

    cellparams: dict
        Neuron cell parameters determining properties of the spiking neurons in
        pyNN simulators.

    Methods
    -------

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

    """

    def __init__(self, ann):
        self.ann = ann
        #self.sim = initialize_simulator()
        #self.add_input_layer()
        self.connections = []
        self.threshold = 'v > v_thresh'
        self.reset = 'v = v_reset'
        self.eqs = 'dv/dt = -v/tau_m : volt'
        self.output_shapes = []
        #self.spikemonitors = [self.sim.SpikeMonitor(self.layers[0])]
        #self.statemonitors = []
        self.megadirname = ''
        self.megaschematic = 'megasim.sch'
        self.input_stimulus_file = "input_events.stim"
        self.labels = ['InputLayer']
        self.layers = []

    def add_input_layer(self):
        input_shape = list(self.ann['input_shape'])
        self.layers = [self.sim.PoissonGroup(
            np.prod(input_shape[1:]), rates=0*self.sim.Hz,
            dt=settings['dt']*self.sim.ms)]
        self.layers[0].add_attribute('label')
        self.layers[0].label = 'InputLayer'

    def build(self):
        """
        Compile a spiking neural network to prepare for simulation with Brian2.

        """

        echo('\n')
        echo("Compiling spiking network...\n")

        # Create megasim dir
        self.megadirname = settings['path'] + settings['filename'] + '/'
        if not os.path.exists(self.megadirname):
            os.makedirs(self.megadirname)

        # clear the folder first from evs and log files
        #TODO: add a method to clean everything in that folder

        # Iterate over hidden layers to create spiking neurons and store
        # connections.
        for (layer_num, layer) in enumerate(self.ann['layers']):
            if layer['layer_type'] in {'Dense', 'Convolution2D',
                                       'MaxPooling2D', 'AveragePooling2D'}:
                echo("Building layer: {}\n".format(layer['label']))
                self.add_layer(layer)
            else:
                echo("Skipped layer:  {}\n".format(layer['layer_type']))
                continue
            if layer['layer_type'] == 'Dense':
                print(layer_num)
                pass
                #self.build_dense(layer)
            elif layer['layer_type'] == 'Convolution2D':
                echo("Not implemented yet!\n")
                sys.exit(99)
                #self.build_convolution(layer)
            elif layer['layer_type'] in {'MaxPooling2D', 'AveragePooling2D'}:
                echo("Not implemented yet!\n")
                sys.exit(99)
                #self.build_pooling(layer)

        # build parameter files for all modules
        for mod_n,module in enumerate(self.layers):
            module.build_parameter_file(dirname=self.megadirname)
            module.build_state_file(dirname= self.megadirname)

        # build MegaSim Schematic file
        self.build_schematic()

        echo("Compilation finished.\n\n")


    def check_megasim_output(self, megalog):
        megalog = str(megalog)
        if megalog.find("error")>0:
            print("MegaSim error: ")
            print(megalog)
            sys.exit(99)

    def intensity_to_poisson_convertion(self, mnist_digit, shape=(28,28), dt=1, n_spikes=250, t_stop=1):
        '''
        This function receives an MNIST character, 28 x 28 array, as input and
        generates random spikes based on the intensity of the pixels, the number of
        spikes desired (n_spikes) and the duration of the stimulus (t_stop).

        Returns a NUMPY 2D array, where the first column holds the neuron IDs (addresses)
        and second column the spike-times (timestamps).

        Based on Daniel O'Neil's MATLAB scripts
        '''
        # y = randsample(n,k,true,w) or y = randsample(population,k,true,w)
        # returns a weighted sample taken with replacement, using a vector of
        # positive weights w, whose length is n. The probability that the integer
        # i is selected for an entry of y is w(i)/sum(w). Usually, w is a vector of
        # probabilities. randsample does not support weighted sampling without
        # replacement.

        # Following the Matlab implementation where the characters
        # are fliped horizontally and rotated by 90 dec counter clockwise

        MNIST = mnist_digit.flatten() * 0.2

        spikesAdr = np.random.choice(len(MNIST), n_spikes, True,
                                     (MNIST / sum(MNIST)))
        spikesTs = np.round(np.sort(np.random.random(len(spikesAdr))
                                    * t_stop) * dt)  # in ms
        Spikes = np.concatenate((spikesAdr, spikesTs), axis=-1)
        Spikes = Spikes.reshape((len(spikesAdr), 2), order='F')
        megastim = np.zeros((len(Spikes), 6), dtype="int")
        # copy time stamps
        megastim[:, 0] = Spikes[:, 1]
        megastim[:, 1] = np.ones((len(Spikes))) * -1
        megastim[:, 2] = np.ones((len(Spikes))) * -1
        megastim[:, 3] = Spikes[:, 0]
        megastim[:, 4] = 0#Spikes[:, 0] // shape[0]  # sensorSize-ys# in megasim Ys are inverted
        megastim[:, 5] = np.ones((len(Spikes)))

        np.savetxt(self.megadirname+self.input_stimulus_file, megastim, delimiter=" ", fmt=("%d"))

    def add_layer(self, layer):
        self.labels.append(layer['label'])
        self.layers.append(
            module_fc(pop_size = layer['output_shape'][1:], neuron_params = settings )
        )

        weights = layer['parameters'][0]  # [W, b][0]
        self.layers[-1].w = weights#.flatten()
        #self.connections.append(self.sim.Synapses(
        #    self.layers[-2], self.layers[-1], model='w:volt', on_pre='v+=w',
        #    dt=settings['dt']*self.sim.ms))
        self.output_shapes.append(layer['output_shape'])
        #self.layers[-1].add_attribute('label')
        self.layers[-1].label = layer['label']
        # if settings['verbose'] > 1:
        #     self.spikemonitors.append(self.sim.SpikeMonitor(self.layers[-1]))
        # if settings['verbose'] == 3:
        #     self.statemonitors.append(self.sim.StateMonitor(self.layers[-1],
        #                                                     'v', record=True))

    def build_schematic(self):
        '''

        This method generates the main MegaSim schematic file to test sample by sample
        -------

        '''
        input_stimulus_file = self.input_stimulus_file
        input_stimulus_node = "input_evs"

        fileo = open(self.megadirname+self.megaschematic, "w")

        fileo.write(".netlist\n")
        # stim file first - node is input_evs
        fileo.write("source {" + input_stimulus_node + "} " + input_stimulus_file + "\n")
        fileo.write("\n")

        for n in range(len(self.layers)):
            if n==0:
                buildline = self.layers[n].module_string +" {"+ input_stimulus_node+"}"+"{"+self.layers[n].label+"} "+self.layers[n].label+".prm" + " " + self.layers[n].label+".stt"
            else:
                buildline = self.layers[n].module_string +" {"+ self.layers[n-1].label+"}"+"{"+ self.layers[n].label+"} "+ self.layers[n].label+".prm" + " " +  self.layers[n].label+".stt"
            fileo.write(buildline + "\n")
        fileo.write("\n")

        fileo.write(".options" + "\n")
        fileo.write("Tmax=" + str(int(settings['duration']+1)) + "\n")
        fileo.close()


    def build_convolution(self, layer):
        weights = layer['parameters'][0]  # [W, b][0]
        nx = layer['input_shape'][3]  # Width of feature map
        ny = layer['input_shape'][2]  # Hight of feature map
        kx = layer['nb_col']  # Width of kernel
        ky = layer['nb_row']  # Hight of kernel
        px = int((kx - 1) / 2)  # Zero-padding columns
        py = int((ky - 1) / 2)  # Zero-padding rows
        if layer['border_mode'] == 'valid':
            # In border_mode 'valid', the original sidelength is
            # reduced by one less than the kernel size.
            mx = nx - kx + 1  # Number of columns in output filters
            my = ny - ky + 1  # Number of rows in output filters
            x0 = px
            y0 = py
        elif layer['border_mode'] == 'same':
            mx = nx
            my = ny
            x0 = 0
            y0 = 0
        else:
            raise Exception("Border_mode {} not supported".format(
                layer['border_mode']))
        # Loop over output filters 'fout'
        for fout in range(weights.shape[0]):
            for y in range(y0, ny - y0):
                for x in range(x0, nx - x0):
                    target = x - x0 + (y - y0) * mx + fout * mx * my
                    # Loop over input filters 'fin'
                    for fin in range(weights.shape[1]):
                        for k in range(-py, py + 1):
                            if not 0 <= y + k < ny:
                                continue
                            source = x + (y + k) * nx + fin * nx * ny
                            for l in range(-px, px + 1):
                                if not 0 <= x + l < nx:
                                    continue
                                self.connections[-1].connect(i=source+l,
                                                             j=target)
                                self.connections[-1].w[source + l, target] = (
                                    weights[fout, fin, py-k, px-l] *
                                    self.sim.volt)
                echo('.')
            echo(' {:.1%}\n'.format(((fout + 1) * weights.shape[1]) /
                 (weights.shape[0] * weights.shape[1])))

    def build_pooling(self, layer):
        if layer['layer_type'] == 'MaxPooling2D':
            echo("WARNING: Layer type 'MaxPooling' not supported yet. " +
                 "Falling back on 'AveragePooling'.\n")
        nx = layer['input_shape'][3]  # Width of feature map
        ny = layer['input_shape'][2]  # Hight of feature map
        dx = layer['pool_size'][1]  # Width of pool
        dy = layer['pool_size'][0]  # Hight of pool
        sx = layer['strides'][1]
        sy = layer['strides'][0]
        for fout in range(layer['input_shape'][1]):  # Feature maps
            for y in range(0, ny - dy + 1, sy):
                for x in range(0, nx - dx + 1, sx):
                    target = int(x / sx + y / sy * ((nx - dx) / sx + 1) +
                                 fout * nx * ny / (dx * dy))
                    for k in range(dy):
                        source = x + (y + k) * nx + fout * nx * ny
                        for l in range(dx):
                            self.connections[-1].connect(i=source+l, j=target)
                echo('.')
            echo(' {:.1%}\n'.format((1 + fout) / layer['input_shape'][1]))
        self.connections[-1].w = self.sim.volt / (dx * dy)

    def store(self):
        '''
        Not needed since megasim always stores the simulation files, params and schematics
        '''
        pass
        # self.snn = self.sim.Network(self.layers, self.connections,
        #                             self.spikemonitors, self.statemonitors)

    def get_spikes(self, ):
        '''

        Returns: a list of all the events from all the layers
        -------

        '''
        events = []
        for n in range(len(self.layers)):
            events.append(
                np.genfromtxt(self.megadirname+"node_"+ self.layers[n].label+".evs",delimiter=" ",dtype="int")
            )
        return events

    def spike_count(self, events, pop_size=10):
        pop_spike_hist = np.histogram(events[:, 3], bins=pop_size,)
        return pop_size

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

        # Load input layer
        # for obj in self.snn.objects:
        #     if 'poissongroup' in obj.name and 'thresholder' not in obj.name:
        #         input_layer = obj

        # Update parameters
        # namespace = {'v_thresh': settings['v_thresh'] * self.sim.volt,
        #              'v_reset': settings['v_reset'] * self.sim.volt,
        #              'tau_m': settings['tau_m'] * self.sim.ms}
        results = []
        guesses = []
        truth = []

        # Iterate over the number of samples to test
        for test_num in range(settings['num_to_test']):
            # If a list of specific input samples is given, iterate over that,
            # and otherwise pick a random test sample from among all possible
            # input samples in X_test.
            si = settings['sample_indices_to_test']
            ind = randint(0, len(X_test) - 1) if si == [] else si[test_num]

            # Add Poisson input.
            if settings['verbose'] > 1:
                echo("Creating poisson input...\n")

            # Generate stimulus file
            self.intensity_to_poisson_convertion(X_test[ind, :], dt= settings['dt'],n_spikes= settings['input_rate'],
                                                 t_stop=settings['duration'])
            #input_layer.rates = X_test[ind, :].flatten() * \
            #    settings['input_rate'] * self.sim.Hz

            # Run simulation for 'duration'.
            if settings['verbose'] > 1:
                echo("Starting new simulation...\n")


            current_dir = os.getcwd()
            os.chdir(self.megadirname)
            run_megasim = subprocess.check_output([MEGASIM_PATH + "megasim", self.megaschematic])
            os.chdir(current_dir)

            # Check megasim output for errors
            self.check_megasim_output(run_megasim)

            spike_monitors = self.get_spikes()
            output_pop_activity = self.spike_count(spike_monitors[-1], self.layers[-1].pop_size[0])
            print (output_pop_activity, self.layers[-1].pop_size[0])
            # Get result by comparing the guessed class (i.e. the index of the
            # neuron in the last layer which spiked most) to the ground truth.

            guesses.append(np.argmax(self.spikemonitors[-1].count))
            truth.append(np.argmax(Y_test[ind, :]))
            results.append(guesses[-1] == truth[-1])

            if settings['verbose'] > 0:
                echo("Sample {} of {} completed.\n".format(test_num + 1,
                     settings['num_to_test']))
                echo("Moving average accuracy: {:.2%}.\n".format(
                    np.mean(results)))

            if settings['verbose'] > 1 and \
                    test_num == settings['num_to_test'] - 1:
                echo("Simulation finished. Collecting results...\n")
                self.collect_plot_results(
                    self.layers, self.output_shapes, snn_precomp,
                    X_test[ind:ind+settings['batch_size']])

            # Reset simulation time and recorded network variables for next
            # run.
            if settings['verbose'] > 1:
                echo("Resetting simulator...\n")
            # Skip during last run so the recorded variables are not discarded
            if test_num < settings['num_to_test'] - 1:
                self.snn.restore()
            if settings['verbose'] > 1:
                echo("Done.\n")

        total_acc = np.mean(results)

        if settings['verbose'] > 1:
            plot_confusion_matrix(truth, guesses,
                                  settings['log_dir_of_current_run'])

        s = '' if settings['num_to_test'] == 1 else 's'
        echo("Total accuracy: {:.2%} on {} test sample{}.\n\n".format(
             total_acc, settings['num_to_test'], s))

        self.snn.restore()

        return total_acc

    def end_sim(self):
        """ Clean up after simulation. """
        pass

    def save(self, path=None, filename=None):
        """ Write model architecture and parameters to disk. """
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
        j = 0
        for (i, layer) in enumerate(layers):
            if i == 0 or 'Flatten' in layer.label:
                continue
            shape = list(output_shapes[j]) + \
                [int(settings['duration'] / settings['dt'])]
            shape[0] = 1  # simparams['num_to_test']
            spiketrains_batch.append((np.zeros(shape), layer.label))
            spiketrain_dict = self.spikemonitors[i].spike_trains()
            spiketrains = np.array(
                [spiketrain_dict[key] / self.sim.ms for key in
                 spiketrain_dict.keys()])
            spiketrains_full = np.empty((np.prod(shape[:-1]), shape[-1]))
            for k in range(len(spiketrains)):
                spiketrain = np.zeros(shape[-1])
                spiketrain[:len(spiketrains[k])] = np.array(
                    spiketrains[k][:shape[-1]])
                spiketrains_full[k] = spiketrain
            spiketrains_batch[j][0][:] = np.reshape(spiketrains_full, shape)
            # Maybe repeat for membrane potential, skipping input layer
            if settings['verbose'] == 3 and i > 0:
                vm = [np.array(v/1e6/self.sim.mV).transpose() for v in
                      self.statemonitors[i-1].v]
                vmem.append((vm, layer.label))
                times = self.statemonitors[0].t / self.sim.ms
                if i == len(layers) - 2:
                    showLegend = True
                plot_potential(times, vmem[-1], showLegend=showLegend)
            j += 1

        output_graphs(spiketrains_batch, ann, X_batch,
                      settings['log_dir_of_current_run'], idx)
