"""
Setup SNN toolbox

"""

from setuptools import setup, find_packages


with open('README.rst') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='snntoolbox',
    version='0.5.0',  # see https://www.python.org/dev/peps/pep-0440/
    description='Spiking neural network conversion toolbox',
    long_description=long_description,
    author='Bodo Rueckauer',
    author_email='bodo.rueckauer@gmail.com',
    url='https://github.com/NeuromorphicProcessorProject/snn_toolbox',
    download_url='https://github.com/NeuromorphicProcessorProject/snn_toolbox'
                 '.git',
    license='MIT',

    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        # 'Development Status :: 3 - Alpha',

        # Indicate who this project is intended for
        # 'Intended Audience :: Researchers',
        # 'Topic :: Software Development :: Build Tools',

        # License
        'License :: OSI Approved :: MIT License',

        # Supported Python versions
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    keywords='neural networks, deep learning, spiking',

    install_requires=requirements,

    setup_requires=['pytest-runner'],

    tests_require=['pytest'],

    # cmdclass={'test': Tox, 'build_doc': BuildDoc},

    # Additional groups of dependencies (e.g. development dependencies).
    # Install them with $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['foo'],
    #     'test': ['bar']
    # },

    packages=find_packages(exclude=[]),

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
        'console_scripts': ['snntoolbox=snntoolbox.bin.run:main']
        # 'gui_scripts': ['snntoolbox_gui=snntoolbox.bin.gui.gui:main']
    },
)
