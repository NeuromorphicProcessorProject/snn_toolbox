# coding=utf-8

"""Test configuration of toolbox."""

import os

import pytest

from snntoolbox.bin.utils import update_setup
from snntoolbox.utils.utils import import_configparser

with open(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       '..', '..', 'requirements.txt'))) as f:
    requirements = []
    for s in f.readlines():
        requirements.append(s.rstrip('\n').split('==')[0])


@pytest.mark.parametrize('required_module', requirements)
def test_imports_from_requirements(required_module):
    import importlib
    assert importlib.import_module(required_module)


# Todo: Add configuration that is expected to pass.
_in_and_out = [
    ({}, False),
    ({'paths': {'path_wd': os.path.dirname(__file__),
                'dataset_path': os.path.dirname(__file__),
                'filename_ann': '98.96'}}, False)
    ]


@pytest.mark.parametrize('params, expect_pass', _in_and_out)
def test_updating_settings(params, expect_pass, _path_wd):
    configparser = import_configparser()
    config = configparser.ConfigParser()
    config.read_dict(params)
    configpath = os.path.join(str(_path_wd), 'config')
    with open(configpath, 'w') as file:
        config.write(file)
    if expect_pass:
        assert update_setup(configpath)
    else:
        pytest.raises(AssertionError, update_setup, configpath)
