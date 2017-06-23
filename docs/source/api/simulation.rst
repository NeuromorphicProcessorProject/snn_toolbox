``snntoolbox.simulation``
=========================

On the output side of the toolbox, the following simulators are currently
implemented:

.. autosummary::
    :nosignatures:

    snntoolbox.simulation.target_simulators.INI_target_sim
    snntoolbox.simulation.target_simulators.pyNN_target_sim
    snntoolbox.simulation.target_simulators.brian2_target_sim
    snntoolbox.simulation.target_simulators.MegaSim_target_sim

The abstract base class :py:class:`~snntoolbox.simulation.utils.AbstractSNN` for
the simulation tools above is contained here:

.. autosummary::
    :nosignatures:

    snntoolbox.simulation.utils

See :ref:`extending` on how to extend the toolbox by another simulator.

The backends for our built-in simulator ``INIsim`` and the custom simulator
``MegaSim`` are included here:

.. autosummary::
    :nosignatures:

    snntoolbox.simulation.backends.inisim.inisim
    snntoolbox.simulation.backends.megasim.megasim

Finally, utility functions for plotting are contained in

.. autosummary::
    :nosignatures:

    snntoolbox.simulation.plotting

:mod:`snntoolbox.simulation.utils`
----------------------------------

.. automodule:: snntoolbox.simulation.utils

:mod:`snntoolbox.simulation.plotting`
-------------------------------------

.. automodule:: snntoolbox.simulation.plotting

``snntoolbox.simulation.backends``
----------------------------------

:mod:`~snntoolbox.simulation.backends.inisim`
.............................................

.. automodule:: snntoolbox.simulation.backends.inisim.inisim

:mod:`~snntoolbox.simulation.backends.megasim`
..............................................

.. automodule:: snntoolbox.simulation.backends.megasim.megasim

``snntoolbox.simulation.target_simulators``
-------------------------------------------

:mod:`~snntoolbox.simulation.target_simulators.INI_target_sim`
..............................................................

.. automodule:: snntoolbox.simulation.target_simulators.INI_target_sim

:mod:`~snntoolbox.simulation.target_simulators.pyNN_target_sim`
...............................................................

.. automodule:: snntoolbox.simulation.target_simulators.pyNN_target_sim

:mod:`~snntoolbox.simulation.target_simulators.brian2_target_sim`
.................................................................

.. automodule:: snntoolbox.simulation.target_simulators.brian2_target_sim

:mod:`~snntoolbox.simulation.target_simulators.MegaSim_target_sim`
..................................................................

.. automodule:: snntoolbox.simulation.target_simulators.MegaSim_target_sim

