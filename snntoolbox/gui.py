# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:25:53 2016

Graphical user interface to set parameters, load models, display results, etc.

@author: rbodo
"""

import os
import sys
import textwrap
import webbrowser
import json

import snntoolbox
from snntoolbox.config import update_setup
from snntoolbox.config import datasets, architectures, model_libs
from snntoolbox.config import simulators, simulators_pyNN
from snntoolbox.config import globalparams, cellparams, simparams
from snntoolbox.config import cellparams_pyNN, simparams_pyNN
from snntoolbox.core.pipeline import test_full
from snntoolbox.tooltip import ToolTip

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

if sys.version_info[0] < 3:
    import Tkinter as tk
    from Tkinter import filedialog, messagebox, font
else:
    import tkinter as tk
    from tkinter import filedialog, messagebox, font


class snntoolboxGUI():
    def __init__(self, parent):
        self.initialized = False
        self.layer_rb_set = False
        self.layer_rbs = []
        self.parent = parent
        self.filetypes = (("json files", '*.json'),
                          ("hdf5 files", '*.h5'),
                          ("All files", '*.*'))
        self.default_path_to_pref = os.path.join(snntoolbox._dir,
                                                 'preferences')
        self.padx = 10
        self.pady = 5
        fontFamily = 'clearlyu devagari'
        self.header_font = (fontFamily, '11', 'bold')
        font.nametofont('TkDefaultFont').configure(family=fontFamily, size=11)
        font.nametofont('TkMenuFont').configure(family=fontFamily, size=11,
                                                weight=font.BOLD)
        font.nametofont('TkTextFont').configure(family=fontFamily, size=11)
        self.kwargs = {'fill': 'both', 'expand': True,
                       'padx': self.padx, 'pady': self.pady}
        self.is_plot_container_destroyed = True
        self.store_last_settings = False
        self.restore_last_pref = True
        self.declare_parameter_vars()
        self.load_settings()
        self.layer_to_plot = tk.StringVar()
        self.main_container = tk.Frame(parent, bg='white')
        self.main_container.pack(side='top', fill='both', expand=True)
        self.globalparams_widgets()
        self.cellparams_widgets()
        self.simparams_widgets()
        self.toggle_state_pyNN(self.simulator.get())
        self.action_widgets()
        self.graph_widgets()
        self.top_level_menu()
        self.initialized = True

    def globalparams_widgets(self):
        # Create a container for individual parameter widgets
        self.globalparams_frame = tk.LabelFrame(self.main_container,
                                                labelanchor='nw',
                                                text="Global parameters",
                                                relief='raised',
                                                borderwidth='3', bg='white')

        self.globalparams_frame.pack(side='left', fill=None, expand=False)
        tip = textwrap.dedent("""\
              Specify general properties of your model and the steps to
              include in your experiment.""")
        ToolTip(self.globalparams_frame, text=tip, wraplength=750, delay=1499)

        # Dataset
        dataset_frame = tk.Frame(self.globalparams_frame, bg='white')
        dataset_frame.pack(**self.kwargs)
        ToolTip(dataset_frame, text="Choose dataset to test.", wraplength=750)
        tk.Label(dataset_frame, text="Dataset", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        dataset_om = tk.OptionMenu(dataset_frame, self.dataset,
                                   *list(datasets))
        dataset_om.pack(fill='both', expand=True)

        # Architecture
        architecture_frame = tk.Frame(self.globalparams_frame, bg='white')
        architecture_frame.pack(**self.kwargs)
        tip = "The type of model architecture."
        ToolTip(architecture_frame, text=tip, wraplength=750)
        tk.Label(architecture_frame, text="Architecture", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        architecture_om = tk.OptionMenu(architecture_frame, self.architecture,
                                        *list(architectures))
        architecture_om.pack(fill='both', expand=True)

        # Model library
        model_lib_frame = tk.Frame(self.globalparams_frame, bg='white')
        model_lib_frame.pack(**self.kwargs)
        tip = "The neural network library used to create the input model."
        ToolTip(model_lib_frame, text=tip, wraplength=750)
        tk.Label(model_lib_frame, text="Model library", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        model_lib_om = tk.OptionMenu(model_lib_frame, self.model_lib,
                                     *list(model_libs))
        model_lib_om.pack(fill='both', expand=True)

        # Debug mode
        debug_cb = tk.Checkbutton(self.globalparams_frame, text=" Debug",
                                  variable=self.debug, height=2, width=20,
                                  bg='white')
        debug_cb.pack(**self.kwargs)
        tip = textwrap.dedent("""\
              If enabled, the dataset used for testing will be reduced to one
              'Batch size' (see parameter below).""")
        ToolTip(debug_cb, text=tip, wraplength=750)

        # Evaluate ANN
        if self.sim_only.get():
            state = tk.DISABLED
        else:
            state = tk.NORMAL
        self.evaluateANN_cb = tk.Checkbutton(self.globalparams_frame,
                                             text=" Evaluate ANN",
                                             variable=self.evaluateANN,
                                             height=2, width=20,
                                             state=state, bg='white')
        self.evaluateANN_cb.pack(**self.kwargs)
        tip = textwrap.dedent("""\
              Only relevant when converting a network, not during
              simulation. If enabled, test the ANN before conversion. If you
              also enabled 'Normalization' (see parameter below), then the
              network will be evaluated again after normalization.""")
        ToolTip(self.evaluateANN_cb, text=tip, wraplength=750)

        # Normalize
        self.normalize_cb = tk.Checkbutton(self.globalparams_frame,
                                           text=" Normalize", bg='white',
                                           variable=self.normalize,
                                           height=2, width=20, state=state)
        self.normalize_cb.pack(**self.kwargs)
        tip = textwrap.dedent("""\
              Only relevant when converting a network, not during simulation.
              If enabled, the weights of the spiking network will be
              normalized by the highest weight or activation.""")
        ToolTip(self.normalize_cb, text=tip, wraplength=750)

        # Overwrite
        overwrite_cb = tk.Checkbutton(self.globalparams_frame,
                                      text=" Overwrite",
                                      variable=self.overwrite,
                                      height=2, width=20, bg='white')
        overwrite_cb.pack(**self.kwargs)
        tip = textwrap.dedent("""\
              If disabled, the save methods will ask for permission to
              overwrite files before writing weights, activations, models etc.
              to disk.""")
        ToolTip(overwrite_cb, text=tip, wraplength=750)

        # Simulate only
        sim_only_cb = tk.Checkbutton(self.globalparams_frame,
                                     text=" Simulate only",
                                     variable=self.sim_only,
                                     height=2, width=20, bg='white')
        sim_only_cb.pack(**self.kwargs)
        sim_only_cb.bind('<Leave>', self.toggle_norm_and_eval_state)
        tip = textwrap.dedent("""\
              If true, skip conversion step and try to load SNN from the
              working directory (see below).""")
        ToolTip(sim_only_cb, text=tip, wraplength=750)

        # Batch size
        batch_size_frame = tk.Frame(self.globalparams_frame, bg='white')
        batch_size_frame.pack(**self.kwargs)
        tk.Label(batch_size_frame, text="Batch size", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        batch_size_sb = tk.Spinbox(batch_size_frame, bg='white',
                                   textvariable=self.batch_size,
                                   from_=1, to_=1e9, increment=1, width=10)
        batch_size_sb.pack(fill='y', expand=True, ipady=5)
        tip = textwrap.dedent("""\
              Number of samples to test ANN with, if 'debug' enabled (see
              parameter above). If the builtin simulator 'INI' is used, the
              batch size specifies the number of test samples that will be
              simulated in parallel. Important: When using 'INI' simulator,
              the batch size can only be run with the batch size it has been
              converted with. To run it with a different batch size, convert
              the ANN from scratch.""")
        ToolTip(batch_size_frame, text=tip, wraplength=700)

        # Verbosity
        verbose_frame = tk.Frame(self.globalparams_frame, bg='white')
        verbose_frame.pack(**self.kwargs)
        tk.Label(verbose_frame, text="Verbosity", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        [tk.Radiobutton(verbose_frame, variable=self.verbose, text=str(i),
                        value=i, bg='white').pack(fill='both', side='left',
                                                  expand=True)
         for i in range(4)]
        tip = textwrap.dedent("""\
              0: No intermediate results or status reports.
              1: Print progress of simulation and intermediate results.
              2: After each batch, plot guessed classes per sample and show an
                 input image. At the end of the simulation, plot the number of
                 spikes for each sample.
              3: Record, plot and return the membrane potential of all layers
                 for the last test sample. Very time consuming.""")
        ToolTip(verbose_frame, text=tip, wraplength=750)

        # Set and display working directory
        path_frame = tk.Frame(self.globalparams_frame, bg='white')
        path_frame.pack(**self.kwargs)
        tk.Button(path_frame, text="Set working dir",
                  command=self.set_cwd, font=self.header_font).pack(side='top')
        check_path_command = path_frame.register(self.check_path)
        self.path_entry = tk.Entry(path_frame, textvariable=self.path,
                                   width=20, validate='focusout',
                                   validatecommand=(check_path_command, '%P'),
                                   bg='white')
        self.path_entry.pack(fill='both', expand=True, side='left')
        scrollX = tk.Scrollbar(path_frame, orient=tk.HORIZONTAL,
                               command=self.__scrollHandler)
        scrollX.pack(fill='x', expand=True, side='bottom')
        self.path_entry['xscrollcommand'] = scrollX.set
        tip = textwrap.dedent("""\
              Specify the working directory. There, the toolbox will look for
              ANN models to convert or SNN models to test, load the weights it
              needs and store (normalized) weights.""")
        ToolTip(path_frame, text=tip, wraplength=750)

        # Specify filename base
        filename_frame = tk.Frame(self.globalparams_frame)
        filename_frame.pack(**self.kwargs)
        tk.Label(filename_frame, text="Filename base:", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        check_file_command = filename_frame.register(self.check_file)
        self.filename_entry = tk.Entry(filename_frame, bg='white',
                                       textvariable=self.filename,
                                       width=20, validate='focusout',
                                       validatecommand=(check_file_command,
                                                        '%P'))
        self.filename_entry.pack(fill='both', expand=True, side='bottom')
        tip = textwrap.dedent("""\
              Base name of all loaded and saved files during this run. The ANN
              model to be converted is expected to be named 'ann_<basename>'.
              The toolbox will save and load converted SNN models under the
              name 'snn_<basename>'. Normalized weights will follow the naming
              scheme 'ann_<basename>_normWeights'.""")
        ToolTip(filename_frame, text=tip, wraplength=750)

    def cellparams_widgets(self):
        # Create a container for individual parameter widgets
        self.cellparams_frame = tk.LabelFrame(self.main_container,
                                              labelanchor='nw',
                                              text="Cell\n parameters",
                                              relief='raised',
                                              borderwidth='3', bg='white')
        self.cellparams_frame.pack(side='left', fill=None, expand=False)
        tip = textwrap.dedent("""\
              Specify parameters of individual neuron cells in the
              converted spiking network. Some are simulator specific.""")
        ToolTip(self.cellparams_frame, text=tip, wraplength=750, delay=1499)

        # Threshold
        v_thresh_frame = tk.Frame(self.cellparams_frame, bg='white')
        v_thresh_frame.pack(**self.kwargs)
        tk.Label(v_thresh_frame, text="v_thresh", bg='white').pack(fill='both',
                                                                   expand=True)
        v_thresh_sb = tk.Spinbox(v_thresh_frame, textvariable=self.v_thresh,
                                 from_=-1e3, to_=1e3, increment=1e-3, width=10)
        v_thresh_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Threshold in mV defining the voltage at which a spike is fired."
        ToolTip(v_thresh_frame, text=tip, wraplength=750)

        # Refractory time constant
        tau_refrac_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_refrac_frame.pack(**self.kwargs)
        tk.Label(tau_refrac_frame, text="tau_refrac",
                 bg='white').pack(fill='both', expand=True)
        tau_refrac_sb = tk.Spinbox(tau_refrac_frame,
                                   textvariable=self.tau_refrac, width=10,
                                   from_=0, to_=1e3, increment=0.01)
        tau_refrac_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Duration of refractory period in milliseconds of the neurons
              after spiking.""")
        ToolTip(tau_refrac_frame, text=tip, wraplength=750)

        # Reset
        v_reset_frame = tk.Frame(self.cellparams_frame, bg='white')
        v_reset_frame.pack(**self.kwargs)
        self.v_reset_label = tk.Label(v_reset_frame, text="v_reset",
                                      state=self.state_pyNN.get(), bg='white')
        self.v_reset_label.pack(fill='both', expand=True)
        self.v_reset_sb = tk.Spinbox(v_reset_frame, textvariable=self.v_reset,
                                     from_=-1e3, to_=1e3, increment=0.1,
                                     width=10, state=self.state_pyNN.get(),
                                     disabledbackground='#eee')
        self.v_reset_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Reset potential in mV of the neurons after spiking."
        ToolTip(v_reset_frame, text=tip, wraplength=750)

        # Resting potential
        v_rest_frame = tk.Frame(self.cellparams_frame, bg='white')
        v_rest_frame.pack(**self.kwargs)
        self.v_rest_label = tk.Label(v_rest_frame, text="v_rest",
                                     state=self.state_pyNN.get(), bg='white')
        self.v_rest_label.pack(fill='both', expand=True)
        self.v_rest_sb = tk.Spinbox(v_rest_frame, textvariable=self.v_rest,
                                    from_=-1e3, to_=1e3, increment=0.1,
                                    width=10, state=self.state_pyNN.get(),
                                    disabledbackground='#eee')
        self.v_rest_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Resting membrane potential in mV.
              Only relevant in pyNN-simulators.""")
        ToolTip(v_rest_frame, text=tip, wraplength=750)

        # e_rev_E
        e_rev_E_frame = tk.Frame(self.cellparams_frame, bg='white')
        e_rev_E_frame.pack(**self.kwargs)
        self.e_rev_E_label = tk.Label(e_rev_E_frame, text="e_rev_E",
                                      state=self.state_pyNN.get(), bg='white')
        self.e_rev_E_label.pack(fill='both', expand=True)
        self.e_rev_E_sb = tk.Spinbox(e_rev_E_frame, textvariable=self.e_rev_E,
                                     from_=-1e-3, to_=1e3, increment=0.1,
                                     width=10, state=self.state_pyNN.get(),
                                     disabledbackground='#eee')
        self.e_rev_E_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Reversal potential for excitatory input in mV.
              Only relevant in pyNN-simulators.""")
        ToolTip(e_rev_E_frame, text=tip, wraplength=750)

        # e_rev_I
        e_rev_I_frame = tk.Frame(self.cellparams_frame, bg='white')
        e_rev_I_frame.pack(**self.kwargs)
        self.e_rev_I_label = tk.Label(e_rev_I_frame, text="e_rev_I",
                                      state=self.state_pyNN.get(), bg='white')
        self.e_rev_I_label.pack(fill='both', expand=True)
        self.e_rev_I_sb = tk.Spinbox(e_rev_I_frame, textvariable=self.e_rev_I,
                                     from_=-1e3, to_=1e3, increment=0.1,
                                     width=10, state=self.state_pyNN.get(),
                                     disabledbackground='#eee')
        self.e_rev_I_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Reversal potential for inhibitory input in mV.
              Only relevant in pyNN-simulators.""")
        ToolTip(e_rev_I_frame, text=tip, wraplength=750)

        # i_offset
        i_offset_frame = tk.Frame(self.cellparams_frame, bg='white')
        i_offset_frame.pack(**self.kwargs)
        self.i_offset_label = tk.Label(i_offset_frame, text="i_offset",
                                       state=self.state_pyNN.get(), bg='white')
        self.i_offset_label.pack(fill='both', expand=True)
        self.i_offset_sb = tk.Spinbox(i_offset_frame,
                                      textvariable=self.i_offset, from_=-1e3,
                                      to_=1e3, increment=1, width=10,
                                      state=self.state_pyNN.get(),
                                      disabledbackground='#eee')
        self.i_offset_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Offset current in nA.
              Only relevant in pyNN-simulators.""")
        ToolTip(i_offset_frame, text=tip, wraplength=750)

        # Membrane capacitance
        cm_frame = tk.Frame(self.cellparams_frame, bg='white')
        cm_frame.pack(**self.kwargs)
        self.cm_label = tk.Label(cm_frame, text="C_mem",
                                 state=self.state_pyNN.get(), bg='white')
        self.cm_label.pack(fill='both', expand=True)
        self.cm_sb = tk.Spinbox(cm_frame, textvariable=self.cm, from_=1e-3,
                                to_=1e3, increment=1e-3, width=10,
                                state=self.state_pyNN.get(),
                                disabledbackground='#eee')
        self.cm_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Membrane capacitance in nF.
              Only relevant in pyNN-simulators.""")
        ToolTip(cm_frame, text=tip, wraplength=750)

        # tau_m
        tau_m_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_m_frame.pack(**self.kwargs)
        self.tau_m_label = tk.Label(tau_m_frame, text="tau_m",
                                    state=self.state_pyNN.get(), bg='white')
        self.tau_m_label.pack(fill='both', expand=True)
        self.tau_m_sb = tk.Spinbox(tau_m_frame, textvariable=self.tau_m,
                                   from_=1, to_=1e6, increment=1, width=10,
                                   state=self.state_pyNN.get(),
                                   disabledbackground='#eee')
        self.tau_m_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Membrane time constant in milliseconds.
              Only relevant in pyNN-simulators.""")
        ToolTip(tau_m_frame, text=tip, wraplength=750)

        # tau_syn_E
        tau_syn_E_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_syn_E_frame.pack(**self.kwargs)
        self.tau_syn_E_label = tk.Label(tau_syn_E_frame, text="tau_syn_E",
                                        state=self.state_pyNN.get(),
                                        bg='white')
        self.tau_syn_E_label.pack(fill='both', expand=True)
        self.tau_syn_E_sb = tk.Spinbox(tau_syn_E_frame,
                                       textvariable=self.tau_syn_E, from_=1e-3,
                                       to_=1e3, increment=1e-3, width=10,
                                       state=self.state_pyNN.get(),
                                       disabledbackground='#eee')
        self.tau_syn_E_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Decay time of the excitatory synaptic conductance in
              milliseconds.
              Only relevant in pyNN-simulators.""")
        ToolTip(tau_syn_E_frame, text=tip, wraplength=750)

        # tau_syn_I
        tau_syn_I_frame = tk.Frame(self.cellparams_frame, bg='white')
        tau_syn_I_frame.pack(**self.kwargs)
        self.tau_syn_I_label = tk.Label(tau_syn_I_frame, text="tau_syn_I",
                                        state=self.state_pyNN.get(),
                                        bg='white')
        self.tau_syn_I_label.pack(fill='both', expand=True)
        self.tau_syn_I_sb = tk.Spinbox(tau_syn_I_frame,
                                       textvariable=self.tau_syn_I,
                                       from_=1e-3, to_=1e3, increment=1e-3,
                                       width=10, state=self.state_pyNN.get(),
                                       disabledbackground='#eee')
        self.tau_syn_I_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Decay time of the inhibitory synaptic conductance in
              milliseconds.
              Only relevant in pyNN-simulators.""")
        ToolTip(tau_syn_I_frame, text=tip, wraplength=750)

    def simparams_widgets(self):
        # Create a container for individual parameter widgets
        self.simparams_frame = tk.LabelFrame(self.main_container,
                                             labelanchor='nw',
                                             text="Simulation\n parameters",
                                             relief='raised',
                                             borderwidth='3', bg='white')
        self.simparams_frame.pack(side='left', fill=None, expand=False)
        tip = textwrap.dedent("""\
              Specify parameters concerning the simulation of the converted
              spiking network. Some are simulator specific.""")
        ToolTip(self.simparams_frame, text=tip, wraplength=750, delay=1499)

        # Simulator
        simulator_frame = tk.Frame(self.simparams_frame, bg='white')
        simulator_frame.pack(**self.kwargs)
        tip = textwrap.dedent("""\
            Choose a simulator to run the converted spiking network with.""")
        ToolTip(simulator_frame, text=tip, wraplength=750)
        tk.Label(simulator_frame, text="Simulator", bg='white',
                 font=self.header_font).pack(fill='both', expand=True)
        simulator_om = tk.OptionMenu(simulator_frame, self.simulator,
                                     *list(simulators),
                                     command=self.toggle_state_pyNN)
        simulator_om.pack(fill='both', expand=True)

        # Time resolution
        dt_frame = tk.Frame(self.simparams_frame, bg='white')
        dt_frame.pack(**self.kwargs)
        tk.Label(dt_frame, text="dt", bg='white').pack(fill='x', expand=True)
        dt_sb = tk.Spinbox(dt_frame, textvariable=self.dt,
                           from_=1e-3, to_=1e3, increment=1e-3, width=10)
        dt_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Time resolution of spikes in milliseconds."
        ToolTip(dt_frame, text=tip, wraplength=750)

        # Duration
        duration_frame = tk.Frame(self.simparams_frame, bg='white')
        duration_frame.pack(**self.kwargs)
        tk.Label(duration_frame, text="duration", bg='white').pack(fill='y',
                                                                   expand=True)
        duration_sb = tk.Spinbox(duration_frame, textvariable=self.duration,
                                 from_=self.dt.get(), to_=1e9, increment=1,
                                 width=10)
        duration_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Runtime of simulation of one input in milliseconds."
        ToolTip(duration_frame, text=tip, wraplength=750)

        # Maximum firing rate
        max_f_frame = tk.Frame(self.simparams_frame, bg='white')
        max_f_frame.pack(**self.kwargs)
        tk.Label(max_f_frame, text="max_f", bg='white').pack(fill='both',
                                                             expand=True)
        max_f_sb = tk.Spinbox(max_f_frame, textvariable=self.max_f,
                              from_=1, to_=10000, increment=1, width=10)
        max_f_sb.pack(fill='y', expand=True, ipady=3)
        tip = "Spike rate in Hz for a fully-on pixel."
        ToolTip(max_f_frame, text=tip, wraplength=750)

        # Delay
        delay_frame = tk.Frame(self.simparams_frame, bg='white')
        delay_frame.pack(**self.kwargs)
        self.delay_label = tk.Label(delay_frame, text="delay",
                                    state=self.state_pyNN.get(), bg='white')
        self.delay_label.pack(fill='both', expand=True)
        self.delay_sb = tk.Spinbox(delay_frame, textvariable=self.delay,
                                   from_=self.duration.get(), to_=1000,
                                   increment=1, width=10,
                                   state=self.state_pyNN.get(),
                                   disabledbackground='#eee')
        self.delay_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Delay in milliseconds. Must be equal to or greater than the
              resolution.
              Only relevant in pyNN-simulators.""")
        ToolTip(delay_frame, text=tip, wraplength=750)

        # Number of samples to test
        num_to_test_frame = tk.Frame(self.simparams_frame, bg='white')
        num_to_test_frame.pack(**self.kwargs)
        self.num_to_test_label = tk.Label(num_to_test_frame,
                                          text="num_to_test",
                                          state=self.state_pyNN.get(),
                                          bg='white')
        self.num_to_test_label.pack(fill='both', expand=True)
        self.num_to_test_sb = tk.Spinbox(num_to_test_frame,
                                         state=self.state_pyNN.get(),
                                         textvariable=self.num_to_test,
                                         from_=1, to_=1e9, increment=1,
                                         width=10, disabledbackground='#eee')
        self.num_to_test_sb.pack(fill='y', expand=True, ipady=3)
        tip = textwrap.dedent("""\
              Number of samples to test.
              Only relevant in pyNN-simulators.""")
        ToolTip(num_to_test_frame, text=tip, wraplength=750)

        # Name of directory where to save plots
        runlabel_frame = tk.Frame(self.simparams_frame, bg='white')
        runlabel_frame.pack(**self.kwargs)
        tk.Label(runlabel_frame, text='run label', bg='white').pack(
            fill='both', expand=True)
        check_runlabel_command = runlabel_frame.register(self.check_runlabel)
        runlabel_entry = tk.Entry(runlabel_frame, bg='white',
                                  textvariable=self.runlabel,
                                  validate='focusout',
                                  validatecommand=(check_runlabel_command,
                                                   '%P'))
        runlabel_entry.pack(fill='both', expand=True, side='bottom')
        tip = textwrap.dedent("""\
            Give your simulation run a name. If verbosity is high, the
            resulting plots will be saved in <cwd>/log/gui/<runlabel>.""")

    def action_widgets(self):
        self.action_frame = tk.Frame(self.globalparams_frame, bg='white')
        self.action_frame.pack(side='bottom', fill='x', expand=False)

        # Start experiment
        tk.Button(self.action_frame, text="Start",
                  font=self.header_font, foreground='red',
                  command=self.start_processing).pack(**self.kwargs)
        tip = textwrap.dedent("""\
              Start the conversion / simulation. Settings can not be changed
              during the run. Process can only aborted from the console.""")
        ToolTip(self.action_frame, text=tip, wraplength=750)

#        # Stop experiment
#        tk.Button(self.action_frame, text="Abort", fg='red',
#                  command=self.stop_processing).pack(**self.kwargs)

    def graph_widgets(self):
        # Create a container for buttons that display plots for individual
        # layers.
        if hasattr(self, 'graph_frame'):
            self.graph_frame.pack_forget()
            self.graph_frame.destroy()
        self.graph_frame = tk.Frame(self.main_container, background='white')
        self.graph_frame.pack(side='left', fill=None, expand=False)
        tip = textwrap.dedent("""\
              Select a layer to display plots like Spiketrains, Spikerates,
              Membrane Potential, Correlations, etc.""")
        ToolTip(self.graph_frame, text=tip, wraplength=750)
        self.select_plots_dir_rb()
        if hasattr(self, 'selected_plots_dir'):
            self.select_layer_rb()

    def select_plots_dir_rb(self):
        self.plot_dir_frame = tk.LabelFrame(self.graph_frame, labelanchor='nw',
                                            text="Select dir", relief='raised',
                                            borderwidth='3', bg='white')
        self.plot_dir_frame.pack(side='top', fill=None, expand=False)
        self.gui_log = os.path.join(self.path.get(), 'log', 'gui')
        if os.path.isdir(self.gui_log):
            plot_dirs = [d for d in sorted(os.listdir(self.gui_log))
                         if os.path.isdir(os.path.join(self.gui_log, d))]
            self.selected_plots_dir = tk.StringVar(value=plot_dirs[0])
            [tk.Radiobutton(self.plot_dir_frame, bg='white', text=name,
                            value=name, variable=self.selected_plots_dir,
                            command=self.select_layer_rb).pack(
                            fill='both', side='bottom', expand=True)
             for name in plot_dirs]
        open_new_cb = tk.Checkbutton(self.graph_frame, bg='white', height=2,
                                     width=20, text='open in new window',
                                     variable=self.open_new)
        open_new_cb.pack(**self.kwargs)
        tip = textwrap.dedent("""\
              If unchecked, the window showing graphs for a certain layer will
              close and be replaced each time you select a layer to plot.
              If checked, an additional window will pop up instead.""")
        ToolTip(open_new_cb, text=tip, wraplength=750)

    def select_layer_rb(self):
        if hasattr(self, 'layer_frame'):
            self.layer_frame.pack_forget()
            self.layer_frame.destroy()
        self.layer_frame = tk.LabelFrame(self.graph_frame, labelanchor='nw',
                                         text="Select layer", relief='raised',
                                         borderwidth='3', bg='white')
        self.layer_frame.pack(side='bottom', fill=None, expand=False)
        self.plots_dir = os.path.join(self.gui_log,
                                      self.selected_plots_dir.get())
        if os.path.isdir(self.plots_dir):
            layer_dirs = [d for d in sorted(os.listdir(self.plots_dir))
                          if d != 'normalization' and
                          os.path.isdir(os.path.join(self.plots_dir, d))]
            [tk.Radiobutton(self.layer_frame, bg='white', text=name,
                            value=name, variable=self.layer_to_plot,
                            command=self.display_graphs).pack(
                            fill='both', side='bottom', expand=True)
             for name in layer_dirs]

    def draw_canvas(self):
        # Create figure with subplots, a canvas to hold them, and add
        # matplotlib navigation toolbar.
        if self.layer_to_plot.get() is '':
            return
        if hasattr(self, 'plot_container') and not self.open_new.get() and \
                not self.is_plot_container_destroyed:
            self.plot_container.wm_withdraw()
        self.plot_container = tk.Toplevel(bg='white')
        self.plot_container.geometry('800x600')
        self.is_plot_container_destroyed = False
        self.plot_container.wm_title('Results from simulation run {}'.format(
            self.runlabel.get()))
        self.plot_container.protocol('WM_DELETE_WINDOW', self.close_window)
        tk.Button(self.plot_container, text='Close Window',
                  command=self.close_window).pack()
        f = plt.figure(figsize=(30, 15))
        f.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.99,
                          wspace=0.01, hspace=0.01)
        num_rows = 3
        num_cols = 5
        gs = gridspec.GridSpec(num_rows, num_cols)
        self.a = [plt.subplot(gs[i, 0:-2]) for i in range(3)]
        self.a += [plt.subplot(gs[i, -2]) for i in range(3)]
        self.a += [plt.subplot(gs[i, -1]) for i in range(3)]
        self.canvas = FigureCanvasTkAgg(f, self.plot_container)
        graph_widget = self.canvas.get_tk_widget()
        graph_widget.pack(side='top', fill='both', expand=True)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, graph_widget)

    def close_window(self):
        self.plot_container.destroy()
        self.is_plot_container_destroyed = True

    def display_graphs(self):
        self.draw_canvas()
        if self.layer_to_plot.get() is '':
            msg = ("Failed to load images. Please select a layer to plot, and "
                   "make sure your working directory contains appropriate "
                   "image files.")
            messagebox.showerror(title="Loading Error", message=msg)
            return
        path_to_plots = os.path.join(self.plots_dir, self.layer_to_plot.get())
        if not os.path.isdir(path_to_plots):
            msg = ("Failed to load images. Please set a working directory "
                   "that contains appropriate image files.")
            messagebox.showerror(title="Loading Error", message=msg)
            return
        saved_plots = sorted(os.listdir(path_to_plots))
        [a.clear() for a in self.a]
        for name in saved_plots:
            i = int(name[:1])
            self.a[i].imshow(mpimg.imread(os.path.join(path_to_plots, name)))

        layer_idx = int(self.layer_to_plot.get()[:2])
        plots_dir_norm = os.path.join(self.plots_dir, 'normalization')
        if os.path.exists(plots_dir_norm):
            normalization_plots = sorted(os.listdir(plots_dir_norm))
        else:
            normalization_plots = []
        activation_distr = None
        weight_distr = None
        for i in range(len(normalization_plots)):
            if int(normalization_plots[i][:2]) == layer_idx:
                activation_distr = normalization_plots[i]
                weight_distr = normalization_plots[i+1]
                break
        if activation_distr and weight_distr:
            self.a[3].imshow(mpimg.imread(os.path.join(self.plots_dir,
                                                       'normalization',
                                                       activation_distr)))
            self.a[6].imshow(mpimg.imread(os.path.join(self.plots_dir,
                                                       'normalization',
                                                       weight_distr)))
        self.a[-1].imshow(mpimg.imread(os.path.join(self.plots_dir,
                                                    'Pearson.png')))
        for a in self.a:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)

        self.canvas.draw()
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side='left', fill='both', expand=True)

    def top_level_menu(self):
        menubar = tk.Menu(self.parent)
        self.parent.config(menu=menubar)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save preferences",
                             command=self.save_settings)
        filemenu.add_command(label="Load preferences",
                             command=self.load_settings)
        filemenu.add_command(label="Restore default preferences",
                             command=self.restore_default_params)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.quit_toolbox)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        helpmenu.add_command(label="Documentation", command=self.documentation)
        menubar.add_cascade(label="Help", menu=helpmenu)

    def documentation(self):
        webbrowser.open('../Documentation.html')

    def about(self):
        msg = ("This is a collection of tools to convert analog neural "
               "networks to fast and high-performing spiking nets.\n\n"
               "Developed at the Institute of Neuroinformatics, \n"
               "University / ETH Zurich.\n\n"
               "Contact: Bodo Rueckauer \n"
               "bodo.rueckauer@gmail.com \n\n"
               "Version: {} \n\n".format('0.1dev') +
               "2016")
        messagebox.showinfo(title="About SNN Toolbox", message=msg)

    def quit_toolbox(self):
        self.store_last_settings = True
        self.save_settings()
        self.parent.destroy()
        self.parent.quit()

    def declare_parameter_vars(self):
        self.dataset = tk.StringVar()
        self.architecture = tk.StringVar()
        self.model_lib = tk.StringVar()
        self.debug = tk.BooleanVar()
        self.evaluateANN = tk.BooleanVar()
        self.normalize = tk.BooleanVar()
        self.overwrite = tk.BooleanVar()
        self.sim_only = tk.BooleanVar()
        self.batch_size = tk.IntVar()
        self.verbose = tk.IntVar()
        self.v_thresh = tk.DoubleVar()
        self.tau_refrac = tk.DoubleVar()
        self.v_reset = tk.DoubleVar()
        self.v_rest = tk.DoubleVar()
        self.e_rev_E = tk.DoubleVar()
        self.e_rev_I = tk.DoubleVar()
        self.i_offset = tk.IntVar()
        self.cm = tk.DoubleVar()
        self.tau_m = tk.IntVar()
        self.tau_syn_E = tk.DoubleVar()
        self.tau_syn_I = tk.DoubleVar()
        self.dt = tk.DoubleVar()
        self.simulator = tk.StringVar()
        self.duration = tk.IntVar()
        self.max_f = tk.IntVar()
        self.delay = tk.IntVar()
        self.num_to_test = tk.IntVar()
        self.path = tk.StringVar(value=snntoolbox._dir)
        self.filename = tk.StringVar()
        self.runlabel = tk.StringVar()
        self.open_new = tk.BooleanVar(value=True)
        self.log_dir_of_current_run = tk.StringVar()
        self.state_pyNN = tk.StringVar(value='normal')

    def restore_default_params(self):
        L = [globalparams, cellparams, simparams,
             cellparams_pyNN, simparams_pyNN]
        self.set_preferences(L)

    def set_preferences(self, L=None):
        if L is None:
            L = [globalparams, cellparams, simparams, cellparams_pyNN,
                 simparams_pyNN]
        p = {}
        for d in L:
            p.update(d)

        if p['path'] == '':
            p['path'] = os.getcwd()

        self.dataset.set(p['dataset'])
        self.architecture.set(p['architecture'])
        self.model_lib.set(p['model_lib'])
        self.path.set(p['path'])
        self.filename.set(p['filename'])
        self.batch_size.set(p['batch_size'])
        self.debug.set(p['debug'])
        self.evaluateANN.set(p['evaluateANN'])
        self.normalize.set(p['normalize'])
        self.overwrite.set(p['overwrite'])
        self.sim_only.set(p['sim_only'])
        self.verbose.set(p['verbose'])
        self.v_thresh.set(p['v_thresh'])
        self.tau_refrac.set(p['tau_refrac'])
        self.v_reset.set(p['v_reset'])
        self.v_rest.set(p['v_rest'])
        self.e_rev_E.set(p['e_rev_E'])
        self.e_rev_I.set(p['e_rev_I'])
        self.i_offset.set(p['i_offset'])
        self.cm.set(p['cm'])
        self.tau_m.set(p['tau_m'])
        self.tau_syn_E.set(p['tau_syn_E'])
        self.tau_syn_I.set(p['tau_syn_I'])
        self.simulator.set(p['simulator'])
        self.duration.set(p['duration'])
        self.dt.set(p['dt'])
        self.max_f.set(p['max_f'])
        self.delay.set(p['delay'])
        self.num_to_test.set(p['num_to_test'])
        self.runlabel.set(p['runlabel'])
        self.open_new.set(p['open_new'])

    def save_settings(self):
        self.globalparams = {'dataset': self.dataset.get(),
                             'architecture': self.architecture.get(),
                             'model_lib': self.model_lib.get(),
                             'path': self.path.get(),
                             'filename': self.filename.get(),
                             'batch_size': self.batch_size.get(),
                             'debug': self.debug.get(),
                             'evaluateANN': self.evaluateANN.get(),
                             'normalize': self.normalize.get(),
                             'overwrite': self.overwrite.get(),
                             'sim_only': self.sim_only.get(),
                             'verbose': self.verbose.get()}
        self.cellparams = {'v_thresh': self.v_thresh.get(),
                           'tau_refrac': self.tau_refrac.get(),
                           'v_reset': self.v_reset.get(),
                           'v_rest': self.v_rest.get(),
                           'e_rev_E': self.e_rev_E.get(),
                           'e_rev_I': self.e_rev_I.get(),
                           'i_offset': self.i_offset.get(),
                           'cm': self.cm.get(),
                           'tau_m': self.tau_m.get(),
                           'tau_syn_E': self.tau_syn_E.get(),
                           'tau_syn_I': self.tau_syn_I.get()}
        self.simparams = {'simulator': self.simulator.get(),
                          'duration': self.duration.get(),
                          'dt': self.dt.get(),
                          'max_f': self.max_f.get(),
                          'delay': self.delay.get(),
                          'num_to_test': self.num_to_test.get(),
                          'runlabel': self.runlabel.get()}
        self.guiparams = {'open_new': self.open_new.get()}

        s = {'globalparams': self.globalparams,
             'cellparams': self.cellparams,
             'simparams': self.simparams,
             'guiparams': self.guiparams}

        if self.store_last_settings:
            if not os.path.exists(self.default_path_to_pref):
                os.makedirs(self.default_path_to_pref)
            with open(os.path.join(self.default_path_to_pref,
                                   '_last_settings.json'), 'w') as f:
                f.write(json.dumps(s))
            self.store_last_settings = False
        else:
            path_to_pref = tk.filedialog.asksaveasfilename(
                defaultextension='.json', filetypes=[("json files", '*.json')],
                initialdir=self.default_path_to_pref,
                title="Choose filename")
            with open(path_to_pref, 'w') as f:
                f.write(json.dumps(s))

    def load_settings(self):
        if self.restore_last_pref:
            self.restore_last_pref = False
            if not os.path.isdir(self.default_path_to_pref):
                return
            path_to_pref = os.path.join(self.default_path_to_pref,
                                        '_last_settings.json')
            if not os.path.isfile(path_to_pref):
                return
        else:
            path_to_pref = tk.filedialog.askopenfilename(
                defaultextension='.json', filetypes=[("json files", '*.json')],
                initialdir=self.default_path_to_pref,
                title="Choose filename")
        s = json.load(open(path_to_pref))
        self.globalparams = s['globalparams']
        self.cellparams = s['cellparams']
        self.simparams = s['simparams']
        self.guiparams = s['guiparams']
        self.set_preferences([self.globalparams,
                              self.cellparams,
                              self.simparams,
                              self.guiparams])

    # Execute main script as a new thread to be able to stop it.
    # Problem: Output is delayed, and stop function does not really terminate
    # thread: Would have to pass "self.stopped()" into "test_full()" (or make
    # it a member of snntoolbox), and then call sys.exit() if self.stopped().
    # Put "self.script_thread = None" in App initializer.
#
#    import threading
#    from builtins import super
#
#    class MainScript(threading.Thread):
#        def __init__(self, viewer):
#            super().__init__()
#            self.__stop = threading.Event()
#            self.viewer = viewer
#
#        def stop(self):
#            self.__stop.set()
#
#        def stopped(self):
#            return self.__stop.isSet()
#
#        def run(self):
#            if self.viewer.filename.get() == '':
#                messagebox.showwarning(title="Warning",
#                                       message="Please specify a filename " +
#                                               "base.")
#                return
#
#            self.viewer.store_last_settings = True
#            self.viewer.save_settings()
#
#            snntoolbox.update_setup(self.viewer.globalparams,
#                                    self.viewer.cellparams,
#                                    self.viewer.simparams)
#
#            while not self.stopped():
#                snntoolbox.test_full()
#                self.stop()
#
#    def start_processing(self):
#        if not self.script_thread:
#            self.script_thread = self.MainScript(self)
#            self.script_thread.start()
#
#    def stop_processing(self):
#        if self.script_thread:
#            print("User stopped execution of current run.")
#            self.script_thread.stop()
#            self.script_thread = None

    def start_processing(self):
        if self.filename.get() == '':
            messagebox.showwarning(title="Warning",
                                   message="Please specify a filename base.")
            return

        self.store_last_settings = True
        self.save_settings()
        self.check_runlabel(self.runlabel.get())

        update_setup(self.globalparams, self.cellparams, self.simparams)
        test_full()

    def check_file(self, P):
        if not os.path.exists(self.path.get()) or \
                not any(P in fname for fname in os.listdir(self.path.get())):
            msg = ("Failed to set filename base:\n"
                   "Either working directory does not exist or contains no "
                   "files with base name \n '{}'".format(P))
            messagebox.showwarning(title="Warning", message=msg)
            return False
        else:
            return True

    def check_path(self, P):
        if not self.initialized:
            return True
        # Look for plots in working directory to display
        self.graph_widgets()

        if not os.path.exists(P):
            msg = "Failed to set working directory:\n" + \
                  "Specified directory does not exist."
            messagebox.showwarning(title="Warning", message=msg)
            return False
        if not any(fname.endswith('.json') for fname in os.listdir(P)):
            msg = "No model file '*.json' found in \n {}".format(P)
            messagebox.showwarning(title="Warning", message=msg)
            return False
        return True

    def check_runlabel(self, P):
        if self.initialized:
            # Set path to plots for the current simulation run
            self.log_dir_of_current_run.set(os.path.join(self.gui_log, P))
            self.globalparams.update({'log_dir_of_current_run':
                                      self.log_dir_of_current_run.get()})
            if not os.path.exists(self.log_dir_of_current_run.get()):
                os.makedirs(self.log_dir_of_current_run.get())

    def set_cwd(self):
        self.path.set(filedialog.askdirectory(title="Set working directory",
                                              initialdir=snntoolbox._dir))
        self.check_path(self.path.get())
        # Look for plots in working directory to display
        self.graph_widgets()

    def __scrollHandler(self, *L):
        op, howMany = L[0], L[1]
        if op == 'scroll':
            units = L[2]
            self.path_entry.xview_scroll(howMany, units)
        elif op == 'moveto':
            self.path_entry.xview_moveto(howMany)

    def toggle_norm_and_eval_state(self, event):
        if self.sim_only.get():
            self.evaluateANN_cb.configure(state=tk.DISABLED)
            self.normalize_cb.configure(state=tk.DISABLED)
        else:
            self.evaluateANN_cb.configure(state=tk.NORMAL)
            self.normalize_cb.configure(state=tk.NORMAL)

    def toggle_state_pyNN(self, val):
        if val not in simulators_pyNN:
            self.state_pyNN.set('disabled')
        else:
            self.state_pyNN.set('normal')
        self.v_reset_label.configure(state=self.state_pyNN.get())
        self.v_reset_sb.configure(state=self.state_pyNN.get())
        self.v_rest_label.configure(state=self.state_pyNN.get())
        self.v_rest_sb.configure(state=self.state_pyNN.get())
        self.e_rev_E_label.configure(state=self.state_pyNN.get())
        self.e_rev_E_sb.configure(state=self.state_pyNN.get())
        self.e_rev_I_label.configure(state=self.state_pyNN.get())
        self.e_rev_I_sb.configure(state=self.state_pyNN.get())
        self.i_offset_label.configure(state=self.state_pyNN.get())
        self.i_offset_sb.configure(state=self.state_pyNN.get())
        self.cm_label.configure(state=self.state_pyNN.get())
        self.cm_sb.configure(state=self.state_pyNN.get())
        self.tau_m_label.configure(state=self.state_pyNN.get())
        self.tau_m_sb.configure(state=self.state_pyNN.get())
        self.tau_syn_E_label.configure(state=self.state_pyNN.get())
        self.tau_syn_E_sb.configure(state=self.state_pyNN.get())
        self.tau_syn_I_label.configure(state=self.state_pyNN.get())
        self.tau_syn_I_sb.configure(state=self.state_pyNN.get())
        self.delay_label.configure(state=self.state_pyNN.get())
        self.delay_sb.configure(state=self.state_pyNN.get())
        self.num_to_test_label.configure(state=self.state_pyNN.get())
        self.num_to_test_sb.configure(state=self.state_pyNN.get())


def main():
    root = tk.Tk()
    root.title("SNN Toolbox")
    app = snntoolboxGUI(root)
    root.protocol('WM_DELETE_WINDOW', app.quit_toolbox)
    root.mainloop()


if __name__ == "__main__":
    main()
