# Input model parsing

Methods to parse an input model written in a certain model library and prepare 
it for further processing in the SNN toolbox.

The idea is to make all further steps in the conversion/simulation pipeline
independent of the original model format. Therefore, when a developer adds a
new input model library (e.g. Torch) to the toolbox, the following methods must
be implemented and satisfy the return requirements specified in their
respective docstrings:

* extract
* evaluate
* load_ann

### File overview

* common.py - Functions common to several input model parsers.
* caffe_input_lib.py - Methods to parse caffe models.
* keras_input_lib.py - Methods to parse keras models.
* lasagne_input_lib.py - Methods to parse lasagne models.
