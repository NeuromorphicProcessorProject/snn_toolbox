``snntoolbox.parsing``
======================

On the input side of the SNN conversion toolbox, models from the following
neural network libraries can be parsed:

.. autosummary::
    :nosignatures:

    snntoolbox.parsing.model_libs.keras_input_lib
    snntoolbox.parsing.model_libs.lasagne_input_lib
    snntoolbox.parsing.model_libs.caffe_input_lib

These parsers inherit from `AbstractModelParser` in

.. autosummary::
    :nosignatures:

    snntoolbox.parsing.utils

See :ref:`extending` on how to extend the toolbox by another input model
library.

``snntoolbox.parsing.model_libs``
---------------------------------

:mod:`~snntoolbox.parsing.model_libs.keras_input_lib`
.....................................................

.. automodule:: snntoolbox.parsing.model_libs.keras_input_lib

:mod:`~snntoolbox.parsing.model_libs.lasagne_input_lib`
.......................................................

.. automodule:: snntoolbox.parsing.model_libs.lasagne_input_lib

:mod:`~snntoolbox.parsing.model_libs.caffe_input_lib`
.....................................................

.. automodule:: snntoolbox.parsing.model_libs.caffe_input_lib

:mod:`snntoolbox.parsing.utils`
-------------------------------

.. automodule:: snntoolbox.parsing.utils
