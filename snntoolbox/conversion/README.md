# SNN toolbox core

Core modules of the SNN toolbox.


### File overview

* inisim.py - A collection of helper functions, including spiking layer classes derived from
  Keras layers, which were used to implement our own IF spiking simulator. Not needed when 
  converting and running the SNN in pyNN or other simulators.
* pipeline.py - Wrapper script that combines all tools of SNN Toolbox.
* util.py - Helper functions to handle parameters and variables of interest during
  conversion and simulation of an SNN.
