"""
Setup SNN toolbox

"""

import os
import sys
from codecs import open
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
# from sphinx.setup_command import BuildDoc


# try:
#     here = os.environ['PYTHONPATH'].split(os.pathsep)[-1]
# except KeyError:
#     here = os.path.abspath(os.path.dirname(__file__))


# Get the long description from the README file
# try:
#     with open(os.path.join(here, 'docs', 'source', 'index.rst'),
#               encoding='utf-8') as f:
#         long_description = f.read()
# except FileNotFoundError:
#     long_description = ''  # Tox can't find the file...


# Tell setuptools to run 'tox' when calling 'python setup.py test'.
class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import tox
        import shlex
        args = self.tox_args
        if args:
            args = shlex.split(self.tox_args)
        errno = tox.cmdline(args=args)
        sys.exit(errno)

setup(
    name='snntoolbox',
    version='0.1.0',  # see https://www.python.org/dev/peps/pep-0440/
    description='Spiking Neural Net Conversion',
    # long_description=long_description,
    author='Bodo Rueckauer',
    author_email='bodo.rueckauer@gmail.com',
    url='https://github.com/NeuromorphicProcessorProject/snn_toolbox',
    download_url='git@github.com:NeuromorphicProcessorProject/snn_toolbox.git',
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who this project is intended for
        'Intended Audience :: Researchers',
        # 'Topic :: Software Development :: Build Tools',

        # License
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    keywords='neural networks, deep learning, spiking',

    install_requires=[
        'future',
        'h5py',
        'keras',
        'matplotlib',
    ],

    setup_requires=['pytest-runner'],

    tests_require=['tox', 'pytest'],

    cmdclass={'test': Tox},  # , 'build_doc': BuildDoc},

    # Additional groups of dependencies (e.g. development dependencies).
    # Install them with $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['foo'],
    #     'test': ['bar']
    # },

    packages=find_packages(exclude=['ann_architectures', 'deprecated',
                                    'snntoolbox.scotopic', 'scripts']),

    package_data={
        'snntoolbox': ['config_defaults']
    },

    # Include documentation
    data_files=[
        ('', ['README.rst']),
    ],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
        'console_scripts': ['snntoolbox=bin.run:main'],
        'gui_scripts': ['snntoolbox_gui=bin.gui.gui:main']
    },
)
