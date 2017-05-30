# Target simulators

The modules in ``target_simulators`` package allow building, testing and
exporting a spiking network.

Each of the files below offers functionality for a specific simulator. Adding
another simulator requires creating a new file ``<custom>_target_sim.py`` which 
implements the class ``SNN_compiled`` with its methods tailored to the specific
simulator.

### Integrated simulators:

* Brian - see pyNN
* Brian2 - Restriction: No saving / loading functions implemented. Model needs to be converted
  each time before simulation.
* IniSim - Export to simulate SNN in a self-implemented Integrate-and-Fire simulator using
  a timestepped approach.
* Nest - see pyNN
* pyNN - Models are exported in pyNN (http://neuralensemble.org/docs/PyNN/).
  pyNN is a simulator-independent language for building neural network
  models. It allows running the converted net in a Spiking Simulator like
  Brian, NEURON, or NEST.
* MegaSim - A simulator developed at the University of Seville with special focus on hardware features.
