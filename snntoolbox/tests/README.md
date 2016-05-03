# Testing Spiking Neural Networks

Code to test spiking conversion and simulation of neural nets.

### File overview

* all_in_one.py - A wrapper script to use all aspects of the toolbox, e.g. converting or simulating only, or performing a random or grid search to determine the optimal hyperparameters.
* augmentation_testsuit.py - Show effect of transforming input images (rotation, noise, centered mean, ...)
* cifar10_testsuit.py - Minimal example of loading a pretrained ANN, evaluating it, converting it to
    spiking, and simulating its performance on nest spiking simulator.
* mnist_testsuit.py - Small classification demonstration of an analog MLP on MNIST digits.
* test_cellparams.py - Simple visualization of how the membrane potential of one IF neuron varies with
    different cell parameters.
* util.py - Helper functions to run various tests on spiking networks.
* convert_only - Usecase:
	1. load and test a pretrained ANN
	2. normalize weights
	3. converting to SNN
	4. save SNN to disk
* simulate_only - Usecase:
	1. Load an already converted SNN
	2. Run it on a spiking simulator
	3. Plot spikerates etc.


* parameter_sweep - Usecase:
	1. Load an already converted SNN
	2. Run it on a spiking simulator while varying a hyperparameter like `v_thresh`
	3. Plot accuracy vs. hyperparameter

