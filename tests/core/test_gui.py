# coding=utf-8

"""Test snntoolbox GUI."""
import shutil

import pytest


@pytest.mark.skip("Passes when all dependencies are installed properly.")
def test_gui(_config):
    import time
    from snntoolbox.bin.gui.gui import tk, SNNToolboxGUI

    root = tk.Tk()
    app = SNNToolboxGUI(root, _config)
    root.update_idletasks()
    root.update()
    time.sleep(0.1)
    app.quit_toolbox()
    shutil.rmtree(app.default_path_to_pref)

    assert True
