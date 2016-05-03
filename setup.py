"""
Setup SNN toolbox

"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'docs', 'source', 'index.rst'),
	      encoding='utf-8') as f:
    long_description = f.read()

setup(
	name='snntoolbox',
    version='0.1.0.dev1', # see https://www.python.org/dev/peps/pep-0440/
	description='Spiking Neural Net Conversion',
	long_description=long_description,
	author='Bodo Rueckauer',
	author_email='bodo.rueckauer@gmail.com',
	url='https://code.ini.uzh.ch/NPP_theory/SNN_toolbox',
	download_url='git@code.ini.uzh.ch:NPP_theory/SNN_toolbox.git',
	license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who this project is intended for
        'Intended Audience :: Researchers',
        #'Topic :: Software Development :: Build Tools',

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
        'numpy',
        'scipy',
        'matplotlib',
        'future',
        'neo',
        'lazyarray',
        'sympy',
        'pyNN'],


    # Additional groups of dependencies (e.g. development dependencies).
	# Install them with
    # $ pip install -e .[dev,test]
#	extras_require={
#		'dev': ['foo'],
#		'test': ['bar']	
#	},

	packages=find_packages(),

	# Include everything in source control
	include_package_data=True,

	# Include documentation
	data_files=[
		('',['README.rst', 'requirements.txt']),
		('docs/',['docs/Makefile']),
#		('docs/source',['docs/source/*.rst', 'workflow.png'])
	],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
		'console_scripts': [
	        'snntoolbox=gui:main',
        ],
    },
)
