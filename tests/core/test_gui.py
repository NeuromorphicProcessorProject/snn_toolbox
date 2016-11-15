# coding=utf-8

"""Test snntoolbox GUI."""


def test_gui():
    from snntoolbox.gui.gui import tk, SNNToolboxGUI
    import time

    root = tk.Tk()
    app = SNNToolboxGUI(root)
    root.update_idletasks()
    root.update()
    time.sleep(1)
    app.quit_toolbox()
