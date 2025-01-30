import tkinter as tk
from tkinter import filedialog
import easygui


file = easygui.fileopenbox(title="Selecciona un archivo")
print(f"Archivo seleccionado: {file}")
