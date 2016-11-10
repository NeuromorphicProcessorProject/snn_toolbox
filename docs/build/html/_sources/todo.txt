.. # coding=utf-8

Todo
====

Possible extensions
-------------------

* Visualize network: `draw_convnet <https://github.com/gwding/draw_convnet>`_
* Implement parser for Torch-input
* Implement export for Megasim
* Implement max-pooling layers (in progress)
* Test batch-normalization layers (implemented)
* Implement nonzero biases for pyNN and Brian2 simulators (working for INIsim).
  Currently, biases in our pyNN export are included as offset currents, but
  this does not seem to have any effect.
* In pyNN and Brian2, implement feedback mechanism to modify weights in runtime
  (works in INIsim)
* In pyNN and Brian2, implement analog input instead of Poisson input (works in
  INIsim)

Known bugs
----------

* When simulating directly (skipping the normalization and conversion step),
  spiking layers use the normalized weights loaded from disk, but the
  activation and correlation plots use the non-normalized weights. This only
  concerns pyNN simulators, not INIsim or Brian2. One would have to write a
  function that loads the normalized weights from the parsed model before
  simulation and writes them to the original network.
* The ipython console in Spyder needs to be restarted after killing the toolbox
  GUI. The problem seems to be thread-related.