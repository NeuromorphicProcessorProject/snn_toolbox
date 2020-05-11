# -*- coding: utf-8 -*-
"""PyTorch model parser.

@author: rbodo
"""

import os
import numpy as np

import torch
import onnx
import onnxruntime
from tensorflow.keras import backend, models, metrics

from snntoolbox.parsing.model_libs import keras_input_lib
from snntoolbox.utils.utils import import_script


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.cpu().numpy()


class ModelParser(keras_input_lib.ModelParser):

    def try_insert_flatten(self, layer, idx, name_map):
        return False
        

def load(path, filename):
    """Load network from file.

    Parameters
    ----------

    path: str
        Path to directory where to load model from.

    filename: str
        Name of file to load model from.

    Returns
    -------

    : dict[str, Union[keras.models.Sequential, function]]
        A dictionary of objects that constitute the input model. It must
        contain the following two keys:

        - 'model': keras.models.Sequential
            Keras model instance of the network.
        - 'val_fn': function
            Function that allows evaluating the original model.
    """

    filepath = str(os.path.join(path, filename))

    # Load the Pytorch model.
    mod = import_script(path, filename)
    kwargs = mod.kwargs if hasattr(mod, 'kwargs') else {}
    model_pytorch = mod.Model(**kwargs)
    map_location = 'cpu' if not torch.cuda.is_available() else None
    for ext in ['.pth', '.pkl']:
        model_path = filepath + ext
        if os.path.exists(model_path):
            break
    assert model_path, "Pytorch state_dict not found at {}".format(model_path)
    model_pytorch.load_state_dict(torch.load(model_path,
                                             map_location=map_location))

    # state_dict = torch.load(model_path, map_location=map_location)['state_dict']
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     k = str(k).replace('module.', '')
    #     new_state_dict[k] = v
    # model_pytorch.load_state_dict(new_state_dict, strict=False)

    # Switch from train to eval mode to ensure Dropout / BatchNorm is handled
    # correctly.
    model_pytorch.eval()

    # Run on dummy input with correct shape to trace the Pytorch model.
    input_shape = [1] + list(model_pytorch.input_shape)
    input_numpy = np.random.random_sample(input_shape).astype(np.float32)
    input_torch = torch.from_numpy(input_numpy).float()
    output_torch = model_pytorch(input_torch)
    output_numpy = to_numpy(output_torch)

    # Export as onnx model, and then reload.
    input_names = ['input_0']
    output_names = ['output_{}'.format(i) for i in range(len(output_torch))]
    dynamic_axes = {'input_0': {0: 'batch_size'}}
    dynamic_axes.update({name: {0: 'batch_size'} for name in output_names})
    torch.onnx.export(model_pytorch, input_torch, filepath + '.onnx',
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)
    model_onnx = onnx.load(filepath + '.onnx')
    # onnx.checker.check_model(model_onnx)  # Crashes with segmentation fault.

    # Compute ONNX Runtime output prediction.
    ort_session = onnxruntime.InferenceSession(filepath + '.onnx')
    input_onnx = {ort_session.get_inputs()[0].name: input_numpy}
    output_onnx = ort_session.run(None, input_onnx)

    # Compare ONNX Runtime and PyTorch results.
    err_msg = "Pytorch model could not be ported to ONNX. Output difference: "
    np.testing.assert_allclose(output_numpy, output_onnx[0],
                               rtol=1e-03, atol=1e-05, err_msg=err_msg)
    print("Pytorch model was successfully ported to ONNX.")

    change_ordering = backend.image_data_format() == 'channels_last'
    if change_ordering:
        input_numpy = np.moveaxis(input_numpy, 1, -1)
        output_numpy = np.moveaxis(output_numpy, 1, -1)

    # Import this here; import changes image_data_format to channels_first.
    from onnx2keras import onnx_to_keras
    # Port ONNX model to Keras.
    model_keras = onnx_to_keras(model_onnx, input_names, [input_shape[1:]],
                                change_ordering=change_ordering, verbose=False)
    if change_ordering:
        backend.set_image_data_format('channels_last')

    # Save the keras model.
    model_keras.compile('sgd', 'categorical_crossentropy',
                        ['accuracy', metrics.top_k_categorical_accuracy])
    models.save_model(model_keras, filepath + '.h5')

    # Compute Keras output and compare against ONNX.
    output_keras = model_keras.predict(input_numpy)
    err_msg = "ONNX model could not be ported to Keras. Output difference: "
    np.testing.assert_allclose(output_numpy, output_keras,
                               rtol=1e-03, atol=1e-05, err_msg=err_msg)
    print("ONNX model was successfully ported to Keras.")

    return {'model': model_keras, 'val_fn': model_keras.evaluate}


def evaluate(*args, **kwargs):
    return keras_input_lib.evaluate(*args, **kwargs)
