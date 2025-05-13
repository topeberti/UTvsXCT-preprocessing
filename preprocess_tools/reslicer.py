from base_plugin import BasePlugin
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
import numpy as np

def ask_reslice(plugin):
        window = QDialog(plugin.main_window)
        layout = QVBoxLayout()

        button_name = None

        def on_button_clicked(name):
            nonlocal button_name
            button_name = name
            window.close()

        for name in ["Top", "Bot", "Left", "Right"]:
            button = QPushButton(name)
            button.clicked.connect(lambda checked, name=name: on_button_clicked(name))
            layout.addWidget(button)

        window.setLayout(layout)
        window.show()
        window.exec()

        return button_name
    
def reslice(volume, name):

    if name == 'Top':
        resliced = np.transpose(volume, (1, 0, 2))[::-1, :, :]
        resliced = np.flip(resliced, axis=1)
    elif name == 'Left':
        resliced = np.transpose(volume, (2, 1, 0))
        resliced = np.flip(resliced, axis=1)
        resliced = np.flip(resliced, axis=2)
    elif name == 'Right':
        resliced = np.transpose(volume, (2, 1, 0))[::-1, :, :]
        resliced = np.flip(resliced, axis=1)
    elif name == 'Bottom':
        resliced = np.transpose(volume, (1, 0, 2))
    
    return resliced

def ask_rotate(plugin, volume):
    window = QDialog(plugin.main_window)
    layout = QVBoxLayout()

    button_name = None

    def on_button_clicked(name):
        nonlocal button_name
        button_name = name
        window.close()

    for name in ["Rotate 90 Left", "Rotate 90 Right", "No Rotation"]:
        button = QPushButton(name)
        button.clicked.connect(lambda checked, name=name: on_button_clicked(name))
        layout.addWidget(button)

    window.setLayout(layout)
    window.show()
    window.exec()

    if button_name == 'Rotate 90 Left':
        rotated = np.rot90(volume, 1, (1, 2))
    elif button_name == 'Rotate 90 Right':
        rotated = np.rot90(volume, -1, (1, 2))
    else:
        rotated = volume

    return rotated

def rotate_auto(volume):
    return np.rot90(volume, -1, (1, 2))