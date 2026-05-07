"""可交互的 matplotlib 画布，支持拖拽编辑曲线点"""

import numpy as np
import tkinter as tk
from typing import Dict, List, Optional, Tuple, Callable
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from .style import PlotStyle


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

            # 改进的点击检测，使用数据坐标系下归一化的距离
            x_range = ax.get_xlim()
            y_range = ax.get_ylim()

            for i in range(len(x_data)):
                norm_dx = (x_data[i] - event.xdata) / (x_range[1] - x_range[0])
                norm_dy = (y_data[i] - event.ydata) / (y_range[1] - y_range[0])
                dist = np.sqrt(norm_dx ** 2 + norm_dy ** 2)

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

        ax = curve_data.get("ax", self.ax)

        # 确保鼠标在正确的坐标轴内
        if event.inaxes != ax:
            return

        # 更新 Y 值
        new_y = event.ydata
        curve_data["y"][index] = new_y

        # 更新线条的数据
        curve_data["line"].set_ydata(curve_data["y"])

        # 更新散点的位置
        if "scatter" in curve_data and curve_data["scatter"]:
            offsets = np.column_stack((curve_data["x"], curve_data["y"]))
            curve_data["scatter"].set_offsets(offsets)

        # 重绘图表
        self.canvas.draw_idle()

    def _on_release(self, event):
        """鼠标释放：结束拖拽"""
        if self.dragging is not None:
            if self.on_curve_modified:
                curve_name = self.dragging["curve"]
                index = self.dragging["index"]
                new_y = self.curves[curve_name]["y"][index]
                self.on_curve_modified(curve_name, index, new_y)

            self.dragging = None

    def add_curve(self, name: str, x: np.ndarray, y: np.ndarray,
                  is_secondary_axis: bool = False, **kwargs):
        """
        向画布中添加一条新曲线
        """
        ax = self.ax
        # 处理双Y轴的情况
        if is_secondary_axis:
            if self.ax2 is None:
                self.ax2 = self.ax.twinx()
            ax = self.ax2

        # 从 style 或 kwargs 中获取样式
        color = kwargs.get("color", self.style.get_color(len(self.curves)))
        linestyle = kwargs.get("linestyle", self.style.get_linestyle(len(self.curves)))
        label = kwargs.get("label", name)

        # 绘制折线
        line, = ax.plot(x, y, color=color, linestyle=linestyle,
                        linewidth=self.style.line_width, label=label)

        # 绘制透明的散点，以增加鼠标点击拖拽的判定区域
        scatter = ax.scatter(x, y, color=color, s=self.style.marker_size ** 2 * 4,
                             alpha=0.5, zorder=line.get_zorder() + 1)

        # 保存曲线状态
        self.curves[name] = {
            "x": np.array(x),
            "y": np.array(y),
            "line": line,
            "scatter": scatter,
            "ax": ax,
            "editable": kwargs.get("editable", True)
        }

        if self.style.show_legend:
            self.update_legend()

        self.canvas.draw_idle()

    def update_legend(self):
        """更新图例（处理可能存在的双Y轴图例合并）"""
        lines1, labels1 = self.ax.get_legend_handles_labels()
        lines2, labels2 = [], []
        if self.ax2:
            lines2, labels2 = self.ax2.get_legend_handles_labels()

        if lines1 or lines2:
            self.ax.legend(lines1 + lines2, labels1 + labels2,
                           loc=self.style.legend_loc,
                           fontsize=self.style.legend_fontsize)

    def configure_axes(self, title: str = None, xlabel: str = None, ylabel: str = None, ylabel2: str = None):
        """配置坐标轴标签和标题"""
        if title:
            self.ax.set_title(title, fontsize=self.style.title_fontsize, fontweight=self.style.title_fontweight)
        if xlabel:
            self.ax.set_xlabel(xlabel, fontsize=self.style.axis_fontsize)
        if ylabel:
            self.ax.set_ylabel(ylabel, fontsize=self.style.axis_fontsize)
        if ylabel2 and self.ax2:
            self.ax2.set_ylabel(ylabel2, fontsize=self.style.axis_fontsize)
        self.canvas.draw_idle()

    def clear(self):
        """清空画布上的所有内容"""
        self.ax.clear()
        if self.ax2:
            self.ax2.clear()
            self.ax2 = None

        self.curves.clear()

        # 恢复默认背景和网格
        self.ax.set_facecolor(self.style.face_color)
        if self.style.show_grid:
            self.ax.grid(True, alpha=self.style.grid_alpha, linestyle=self.style.grid_style)

        self.canvas.draw_idle()