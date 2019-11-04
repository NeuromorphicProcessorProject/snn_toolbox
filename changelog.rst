SNN Toolbox: Release Notes
==========================

Version 0.3.2
-------------

Simulation with Brian2 backend now supports:
    - Constant input currents (less noisy than Poisson input)
    - Reset-by-subtraction (more accurate than reset-to-zero).
    - Bias currents

Thanks to wilkieolin for this contribution.

Version 0.3.1
-------------

Bugfixes:
    - Setting biases in convolution layers for pyNN and Brian2 simulator
      backends.
    - Parsing axis parameter in BatchNorm layers.
    - Counting of SNN operations.
    - Minor issues due to updating to latest keras / tensorflow version.
    - Syntax error in equation for membrane potential due to updating Brian2.
    - Restoring a previously saved SNN to run with INIsim now works again.
    - Fixed issue #25 (permutation of weights after flatten layer in models
      trained with recent Keras version and simulated with Brian2 / pyNN).

Added support for:
    - Intel's neuromorphic platform "Loihi".
    - Tensorflow 2.0.
    - Parsing depthwise-separable convolutions.
    - Strides > 1 for pyNN and Brian2 simulator backends.
    - Parsing a model can be skipped now by loading a previously saved parsed
      model.
    - Using SNN toolbox more easily from within a python script instead of via
      terminal only.
    - Save and load functions for Brian2 networks.

Miscellaneous:
    - Added end-to-end examples for creating and training the model, saving
      the dataset, and setting up the config file to run SNN toolbox.
    - Moved large model files and datasets to separate repository
      (snntoolbox_applications) to shrink size of core package.
    - Minor refactoring, repo cleanup, and performance improvements.

Contributors:
    - rbodo
    - sflin
    - nandantumu
    - morth
    - wilkieolin
    - Al-pha
