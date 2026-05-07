"""Curve-Plotter 程序入口"""

import sys
import tkinter as tk
from core.app import CurvePlotterApp


def main():
    root = tk.Tk()
    root.title("Curve-Plotter - 深度学习训练曲线生成工具")
    root.geometry("1400x900")
    root.minsize(1000, 700)

    app = CurvePlotterApp(root)
    app.pack(fill=tk.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()