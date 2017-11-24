import numpy as np
import keras.backend as k
from keras.layers import MaxPooling2D
from snntoolbox.simulation.backends.inisim.temporal_mean_rate_tensorflow \
    import SpikeLayer, spike_call


class SpikeMaxPooling2D(MaxPooling2D, SpikeLayer):
    """Spike Max Pooling."""

    def __init__(self, **kwargs):
        MaxPooling2D.__init__(self, **kwargs)
        self.spikerate_pre = self.previous_x = None
        self.activation_str = None

    def build(self, input_shape):
        """Creates the layer neurons and connections..

        Parameters
        ----------

        input_shape: Union[list, tuple, Any]
            Keras tensor (future input to layer) or list/tuple of Keras tensors
            to reference for weight shape computations.
        """

        MaxPooling2D.build(self, input_shape)
        self.init_neurons(input_shape)
        self.spikerate_pre = k.zeros(input_shape)
        self.previous_x = k.zeros(input_shape)

    @spike_call
    def call(self, x, mask=None):
        """Layer functionality."""

        print("WARNING: Rate-based spiking MaxPooling layer is not implemented "
              "in TensorFlow backend. Falling back on AveragePooling. Switch "
              "to Theano backend to use MaxPooling.")
        return k.pool2d(x, self.pool_size, self.strides, self.padding,
                        pool_mode='avg')

    def _pooling_function(self, inputs, pool_size, strides, padding,
                          data_format):
        return spike_pool2d(inputs, pool_size, strides, padding, data_format)

    def reset(self, sample_idx):
        """Reset layer variables."""

        self.reset_spikevars(sample_idx)
        mod = self.config.getint('simulation', 'reset_between_nth_sample')
        mod = mod if mod else sample_idx + 1
        k.set_value(self.spikerate_pre, k.tf.cond(
            sample_idx % mod == 0,
            lambda: np.zeros(self.input_shape, k.floatx()),
            lambda: k.get_value(self.spikerate_pre)))


# noinspection PyProtectedMember
def spike_pool2d(inputs, pool_size, strides=(1, 1), padding='valid',
                 data_format=None):
    """2D max pooling with spikes.

    # Arguments
        inputs: Tensor or variable.
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        padding: string, `"same"` or `"valid"`.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        A tensor, result of 2D pooling.

    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or
        `"channels_first"`.
        ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
    """

    if data_format is None:
        data_format = k.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    padding = k.tensorflow_backend._preprocess_padding(padding)
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)

    x = inputs[0]  # Presynaptic spike-rates
    y = inputs[1]  # Presynaptic spikes

    x = k.tensorflow_backend._preprocess_conv2d_input(x, data_format)
    y = k.tensorflow_backend._preprocess_conv2d_input(y, data_format)

    x = spike_max_pool(x, y, pool_size, strides, padding)

    x = k.tensorflow_backend._postprocess_conv2d_output(x, data_format)

    return x


def spike_max_pool(value_x, value_y, ksize, strides, padding, name=None):
    """Performs the max pooling on the input.

    Args:
      value_x: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
        type `tf.float32`.
      value_y: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
        type `tf.float32`.
      ksize: A list of ints that has length >= 4.  The size of the window for
        each dimension of the input tensor.
      strides: A list of ints that has length >= 4.  The stride of the sliding
        window for each dimension of the input tensor.
      padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
        See the @{tf.nn.convolution$comment here}
      name: Optional name for the operation.

    Returns:
      A `Tensor` with type `tf.float32`. The max pooled output tensor.
    """

    with k.tf.name_scope(name, "SpikeMaxPool", [value_x, value_y]) as name:
        value_x = k.tf.convert_to_tensor(value_x, name="value_x")
        value_y = k.tf.convert_to_tensor(value_y, name="value_y")
        return k.tf.py_func(_spike_max_pool, [value_x, value_y, ksize, strides,
                            padding], k.tf.float32, False, name)


def _spike_max_pool(xr, xs, ksize, strides, padding):
    """Performs max pooling on the input.

    Args:
    xr: A `Tensor`. Must be one of the following types: `float32`,
      `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D input to pool over.
    xs: A `Tensor`. Must be one of the following types: `float32`,
      `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
      4-D input to pool over.
    ksize: A list of `ints` that has length `>= 4`.
      The size of the window for each dimension of the input tensor.
    strides: A list of `ints` that has length `>= 4`.
      The stride of the sliding window for each dimension of the
      input tensor.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.

    Returns:
    A `Tensor`. Has the same type as `input`. The max pooled output tensor.
    """

    stride = strides[1:3]
    ws = ksize[1:3]

    padding = padding.decode('utf-8')

    if padding == 'SAME':
        w_pad = ws[0] - 2 if ws[0] > 2 and ws[0] % 2 == 1 else ws[0] - 1
        h_pad = ws[1] - 2 if ws[1] > 2 and ws[1] % 2 == 1 else ws[1] - 1
        pad = (w_pad, h_pad)
    elif padding == 'VALID':
        pad = (0, 0)
    else:
        raise ValueError('Invalid border mode: ', padding)

    # xr contains the presynaptic spike-rates, and xs the presynaptic spikes
    nd = 2
    if len(xr.shape) < nd:
        raise NotImplementedError(
            'Pool requires input with {} or more dimensions'.format(nd))

    z_shape = out_shape(xr.shape, ws, stride, pad, nd)

    z = np.zeros(z_shape, dtype=xr.dtype)
    # size of pooling output
    pool_out_shp = z.shape[-nd:]
    img_shp = tuple(xr.shape[-nd + i] + 2 * pad[i] for i in range(nd))

    # pad the image
    if max(pad) != 0:
        yr = np.zeros(xr.shape[:-nd] + img_shp, dtype=xr.dtype)
        # noinspection PyTypeChecker
        yr[(slice(None),)*(len(xr.shape)-nd) + tuple(
            slice(pad[i], img_shp[i]-pad[i]) for i in range(nd))] = xr
        ys = np.zeros(xs.shape[:-nd] + img_shp, dtype=xs.dtype)
        # noinspection PyTypeChecker
        ys[(slice(None),)*(len(xs.shape)-nd) + tuple(slice(
            pad[i], img_shp[i]-pad[i]) for i in range(nd))] = xs
    else:
        yr = xr
        ys = xs

    # precompute the region boundaries for each dimension
    region_slices = [[] for _ in range(nd)]
    for i in range(nd):
        for j in range(pool_out_shp[i]):
            start = j * stride[i]
            end = min(start + ws[i], img_shp[i])
            start = max(start, pad[i])
            end = min(end, img_shp[i] - pad[i])
            region_slices[i].append(slice(start, end))

    # TODO: Spike size should equal threshold, which may vary during simulation.
    spike = 1  # kwargs[str('v_thresh')]

    def f(ysn_, r_, rate_patch_):
        spike_patch = ysn_[[region_slices[i][r_[i]] for i in range(nd)]]
        # The second condition is not completely equivalent to the first
        # because the former has a higher chance of admitting spikes.
        # return k.tf.cond(
        #     (spike_patch*(rate_patch_ == np.argmax(rate_patch_))).any(),
        #     lambda: spike, lambda: 0)
        return k.tf.cond(
            spike_patch.flatten()[np.argmax(rate_patch_)],
            lambda: spike, lambda: 0)

    def g():
        # Need to prevent the layer to output a spike at
        # index 0 if all rates are equally zero.
        return 0

    # iterate over non-pooling dimensions
    for n in np.ndindex(*xr.shape[:-nd]):
        yrn = yr[n]
        ysn = ys[n]
        zn = z[n]
        # iterate over pooling regions
        for r in np.ndindex(*pool_out_shp):
            rate_patch = yrn[[region_slices[i][r[i]] for i in range(nd)]]
            zn[r] = k.tf.cond(rate_patch.any(), f(ysn, r, rate_patch), g)

    if padding == 'SAME':
        expected_width = (xr.shape[1] + stride[0] - 1) // stride[0]
        expected_height = (xr.shape[2] + stride[1] - 1) // stride[1]
        z = z[:, : expected_width, : expected_height, :]

    return z


def out_shape(imgshape, ws, stride, pad, ndim=2):
    patch_shape = tuple(imgshape[-ndim + i] + pad[i] * 2
                        for i in range(ndim))
    outshape = [compute_out(patch_shape[i], ws[i], stride[i])
                for i in range(ndim)]
    return list(imgshape[:-ndim]) + outshape


def compute_out(v, downsample, stride):
    if downsample == stride:
        return v // stride
    else:
        out = (v - downsample) // stride + 1
        try:
            if k.is_keras_tensor(out):
                return k.maximum(out, 0)
        except ValueError:
            return np.maximum(out, 0)
