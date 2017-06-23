# coding=utf-8

"""Test snntoolbox GUI."""


def test_gui():
    import time
    from bin.gui.gui import tk, SNNToolboxGUI

    root = tk.Tk()
    app = SNNToolboxGUI(root)
    root.update_idletasks()
    root.update()
    time.sleep(0.1)
    app.quit_toolbox()

    assert True
