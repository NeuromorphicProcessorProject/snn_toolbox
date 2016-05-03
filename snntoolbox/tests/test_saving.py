# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:54:20 2016

@author: rbodo
"""

# For compatibility with python2
from __future__ import print_function, unicode_literals
from __future__ import division, absolute_import
from future import standard_library
standard_library.install_aliases()


if __name__ == '__main__':
    import pyNN.nest as sim
    from pyNN.common.populations import Assembly
    from snntoolbox.io.save import save_assembly
    from snntoolbox.io.load import load_assembly
    import snntoolbox

    cell = sim.Population(1, sim.IF_cond_exp())
    cell2 = sim.Population(2, sim.IF_cond_exp())

    cell.set(**snntoolbox.cellparams)
    cell2.set(**snntoolbox.cellparams)

    p = sim.Projection(cell, cell2, sim.FromListConnector([(0, 0, 0.1),
                                                           (0, 1, 0.2)],
                                                          ['weight']))
    A = Assembly(cell, cell2)

    filename = 'assembly'
    save_assembly(A, filename)
    load_assembly(filename, sim)
    p.save('all', filename)
