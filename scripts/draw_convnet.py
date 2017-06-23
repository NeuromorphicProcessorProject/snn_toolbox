# coding=utf-8

"""
Copyright (c) 2016, Gavin Weiguang Ding
All rights reserved.

Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
plt.rcdefaults()


NumConvMax = 8
NumFcMax = 20
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Black = 0.


def add_layer(patch_list, color_list, size=24, num=5, top_left=None,
              loc_diff=None):
    """Add layer.

    Parameters
    ----------
    patch_list :
    color_list :
    size :
    num :
    top_left :
    loc_diff :
    """

    if top_left is None:
        top_left = [0, 0]
    if loc_diff is None:
        loc_diff = [3, -3]
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size])
    for i in range(num):
        patch_list.append(Rectangle(loc_start + i * loc_diff, size, size))
        if i % 2:
            color_list.append(Medium)
        else:
            color_list.append(Light)


def add_mapping(patch_list, color_list, start_ratio, patch_size, ind_bgn,
                topleftlist, locdifflist, numshowlist, sizelist):
    """

    Parameters
    ----------
    patch_list :
    color_list :
    start_ratio :
    patch_size :
    ind_bgn :
    topleftlist :
    locdifflist :
    numshowlist :
    sizelist :
    """

    start_loc = topleftlist[ind_bgn] \
        + (numshowlist[ind_bgn] - 1) * np.array(locdifflist[ind_bgn]) \
        + np.array([start_ratio[0] * sizelist[ind_bgn],
                    -start_ratio[1] * sizelist[ind_bgn]])

    end_loc = topleftlist[ind_bgn + 1] + (numshowlist[ind_bgn + 1] - 1) \
        * np.array(locdifflist[ind_bgn + 1]) \
        + np.array([(start_ratio[0] + .5 * patch_size / sizelist[ind_bgn]) *
                    sizelist[ind_bgn + 1],
                    -(start_ratio[1] - .5 * patch_size / sizelist[ind_bgn]) *
                    sizelist[ind_bgn + 1]])

    patch_list.append(Rectangle(start_loc, patch_size, patch_size))
    color_list.append(Dark)
    patch_list.append(Line2D([start_loc[0], end_loc[0]],
                             [start_loc[1], end_loc[1]]))
    color_list.append(Black)
    patch_list.append(Line2D([start_loc[0] + patch_size, end_loc[0]],
                             [start_loc[1], end_loc[1]]))
    color_list.append(Black)
    patch_list.append(Line2D([start_loc[0], end_loc[0]],
                             [start_loc[1] + patch_size, end_loc[1]]))
    color_list.append(Black)
    patch_list.append(Line2D([start_loc[0] + patch_size, end_loc[0]],
                             [start_loc[1] + patch_size, end_loc[1]]))
    color_list.append(Black)


def label(xy, text, xy_off=None):
    """Create layer label.

    Parameters
    ----------
    xy: list[int]
    text: str
    xy_off: Optional[list[int]]
    """

    if xy_off is None:
        xy_off = [0, 4]
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)


if __name__ == '__main__':

    fc_unit_size = 2
    layer_width = 60

    patches = []
    colors = []

    fig, ax = plt.subplots()

    ############################
    # conv layers
    size_list = [32, 32, 32, 16, 14, 12, 6]
    num_list = [3, 32, 32, 32, 64, 64, 64]
    x_diff_list = [0, layer_width, layer_width, layer_width, layer_width,
                   layer_width, layer_width]
    text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
    loc_diff_list = [[3, -3]] * len(size_list)

    num_show_list = map(min, num_list, [NumConvMax] * len(num_list))
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

    for ind in range(len(size_list)):
        add_layer(patches, colors, size=size_list[ind],
                  num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
            num_list[ind], size_list[ind], size_list[ind]))

    ############################
    # in between layers
    start_ratio_list = [[0.4, 0.5], [0.4, 0.5], [0.4, 0.8],
                        [0.4, 0.5], [0.4, 0.5], [0.4, 0.8]]
    patch_size_list = [3, 3, 2, 3, 3, 2]
    ind_bgn_list = range(len(patch_size_list))
    text_list = ['Convolution', 'Convolution', 'Max-pooling',
                 'Convolution', 'Convolution', 'Max-pooling']

    for ind in range(len(patch_size_list)):
        add_mapping(patches, colors, start_ratio_list[ind],
                    patch_size_list[ind], ind,
                    top_left_list, loc_diff_list, num_show_list, size_list)
        label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
            patch_size_list[ind], patch_size_list[ind]), xy_off=[26, -75])

    ############################
    # fully connected layers
    size_list = [fc_unit_size, fc_unit_size]
    num_list = [512, 10]
    num_show_list = map(min, num_list, [NumFcMax] * len(num_list))
    x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
    top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
    loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
    text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

    for ind in range(len(size_list)):
        add_layer(patches, colors, size=size_list[ind], num=num_show_list[ind],
                  top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label(top_left_list[ind], text_list[ind] + '\n{}'.format(
            num_list[ind]))

    text_list = ['Flatten\n', 'Fully\nconnected']

    for ind in range(len(size_list)):
        label(top_left_list[ind], text_list[ind], xy_off=[-10, -75])

    ############################
    colors += [0, 1]
    collection = PatchCollection(patches, cmap=plt.cm.gray)
    collection.set_array(np.array(colors))
    ax.add_collection(collection)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.pdf'
    fig.savefig(os.path.join(fig_dir, 'convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)
