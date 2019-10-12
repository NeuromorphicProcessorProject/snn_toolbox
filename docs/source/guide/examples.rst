.. # coding=utf-8

Examples
========

To set up an :doc:`experiment <running>` with the SNN toolbox, we need to
create a config file as described in :doc:`settings <configuration>`. The
``examples`` subdirectory of the repository root provides a number of
end-to-end tutorials covering how to

   - set up a small CNN using Keras and tensorflow,
   - train the model on MNIST,
   - store model and dataset in a temporary folder on disk,
   - create a configuration file for SNN toolbox,
   - call the main function of SNN toolbox to convert the trained ANN to an SNN
     and run it using some simulator backend.

More examples are included in a stand-alone `applications repository
<https://github.com/NeuromorphicProcessorProject/snntoolbox_applications.git>`_.
