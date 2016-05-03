SNN toolbox Documentation
=========================

A completed but most likely outdated build of the documentation is linked to ``Documentation.html`` in the repository root directory.
The source for this documentation is located in the directory ``docs/source/``.

We plan in near future to host the documentation on a NPP website in the INI domain, which will contain a webhook to the repository so it is updated at every commit.

If you want to build an up-to-date version of the documentation, here are the steps.

- Checkout the repository.
- Install the documentation tool Sphinx: ``pip install sphinx recommonmark``.
- Configure sphinx: ``sphinx-quickstart``.
  When answering the questions, be sure to enable the ``autodoc`` extension.
  This can also be done later by editing ``docs/source/conf.py`` to contain the line 
  ``extensions = ['sphinx.ext.autodoc']``.
- ``cd`` to the ``docs/`` folder and run ``make html``.
- Open ´´/docs/build/html/index.html´´ in the webbrowser to view the documentation.

Sidenote: Sphinx ``autodoc`` `extension <http://www.sphinx-doc.org/en/stable/ext/autodoc.html>`_ scans the project source files to generate an API reference documentation based on the modules' docstrings. For this, ``autodoc`` needs to import the modules of the project. Consequently, if not all `dependencies <requirements.txt>`_ of the toolbox (``matplotlib``, ``python-future``, ...) were installed properly, the documentation should still build, but some of the API reference documentation will be missing. Similarly, if you already installed the toolbox in a virtual environment and later want to build the documentation, be sure to activate the environment before running ``make html``, otherwise ``autodoc`` won't find the source code when trying to generate the API reference documentation.

