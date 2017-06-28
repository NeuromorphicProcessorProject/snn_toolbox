# coding=utf-8

"""Test snntoolbox GUI."""


def test_gui(_config):
    import time
    from snntoolbox.bin.gui.gui import tk, SNNToolboxGUI

    root = tk.Tk()
    app = SNNToolboxGUI(root, _config)
    root.update_idletasks()
    root.update()
    time.sleep(0.1)
    app.quit_toolbox()

    assert True
