"""可交互的 matplotlib 画布，支持拖拽编辑曲线点"""

import numpy as np
import tkinter as tk
from typing import Dict, List, Optional, Tuple, Callable
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from .style import PlotStyle


class DraggablePoint:
    """可拖拽的数据点"""

    def __init__(self, line: Line2D, index: int, canvas: "InteractiveCanvas"):
        self.line = line
        self.index = index
        self.canvas = canvas


class InteractiveCanvas(tk.Frame):
    """
    可交互的绘图画布。
    支持：
    - 显示多条曲线
    - 拖拽编辑曲线上的点
    - 实时更新
    """

    def __init__(self, parent, style: Optional[PlotStyle] = None, **kwargs):
        super().__init__(parent, **kwargs)

        self.style = style or PlotStyle()
        self.curves: Dict[str, dict] = {}  # name -> {x, y, line, scatter}
        self.dragging: Optional[dict] = None
        self.drag_enabled = True
        self.on_curve_modified: Optional[Callable] = None

        self._setup_figure()
        self._setup_events()

    def _setup_figure(self):
        """初始化 matplotlib Figure 和 Canvas"""
        self.figure = Figure(
            figsize=self.style.figure_size,
            facecolor=self.style.background_color,
            dpi=100,
        )
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(self.style.face_color)
        self.ax2: Optional[plt.Axes] = None

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # 工具栏
        self.toolbar_frame = tk.Frame(self)
        self.toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def _setup_events(self):
        """绑定鼠标事件用于拖拽"""
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_press(self, event):
        """鼠标按下：检测是否点击了数据点"""
        if not self.drag_enabled or event.inaxes is None:
            return
        if event.button != 1:  # 只响应左键
            return

        min_dist = float("inf")
        closest_curve = None
        closest_index = None

        for name, curve_data in self.curves.items():
            if not curve_data.get("editable", True):
                continue

            x_data = curve_data["x"]
            y_data = curve_data["y"]

            # 转换为显示坐标计算距离
            ax = curve_data.get("ax", self.ax)
            transform = ax.transData
            for i in range(len(x_data)):
                display_coords = transform.transform((x_data[i], y_data[i]))
                dx = display_coords[0] - event.x
                dy = display_coords[1] - event.y
                # 注意：event.x/y 是 canvas 像素坐标，需要用不同方式
                # 使用数据坐标的近似距离
                x_range = ax.get_xlim()
                y_range = ax.get_ylim()
                norm_dx = (x_data[i] - event.xdata) / (x_range[1] - x_range[0])
                norm_dy = (y_data[i] - event.ydata) / (y_range[1] - y_range[0])
                dist = np.sqrt(norm_dx**2 + norm_dy**2)

                if dist < min_dist:
                    min_dist = dist
                    closest_curve = name
                    closest_index = i

        # 阈值：归一化距离小于 0.03 才算点击到了
        if min_dist < 0.03 and closest_curve is not None:
            self.dragging = {
                "curve": closest_curve,
                "index": closest_index,
            }

    def _on_motion(self, event):
        """鼠标移动：拖拽数据点"""
        if self.dragging is None or event.inaxes is None:
            return

        curve_name = self.dragging["curve"]
        index = self.dragging["index"]
        curve_data = self.curves[curve_name]

        # 更新 Y 值