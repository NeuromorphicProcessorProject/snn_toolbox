# coding=utf-8

"""Test snntoolbox GUI."""


def test_gui():
    try:
        from snntoolbox.gui.gui import tk, SNNToolboxGUI
    except ImportError:
        return
    import time

    root = tk.Tk()
    app = SNNToolboxGUI(root)
    root.update_idletasks()
    root.update()
    time.sleep(0.1)
    app.quit_toolbox()
