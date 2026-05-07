"""图表样式管理"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib


@dataclass
class PlotStyle:
    """图表样式配置"""

    # 标题
    title: str = "Training Curves"
    title_fontsize: int = 14
    title_fontweight: str = "bold"

    # 坐标轴
    xlabel: str = "Epoch"
    ylabel_loss: str = "Loss"
    ylabel_accuracy: str = "Accuracy"
    axis_fontsize: int = 12
    tick_fontsize: int = 10

    # 线条
    line_width: float = 1.8
    line_colors: List[str] = field(default_factory=lambda: [
        "#2196F3",  # 蓝色 - train loss
        "#FF5722",  # 红色 - val loss
        "#4CAF50",  # 绿色 - train acc
        "#FF9800",  # 橙色 - val acc
        "#9C27B0",  # 紫色
        "#00BCD4",  # 青色
    ])
    line_styles: List[str] = field(default_factory=lambda: [
        "-", "--", "-", "--", "-.", ":"
    ])
    markers: List[str] = field(default_factory=lambda: [
        "", "", "", "", "", ""
    ])
    marker_size: float = 4.0

    # 图例
    legend_loc: str = "best"
    legend_fontsize: int = 10
    show_legend: bool = True

    # 网格
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_style: str = "--"

    # 图表尺寸
    figure_size: Tuple[float, float] = (10, 6)
    subplot_layout: str = "single"  # "single", "dual_axis", "subplots"

    # 背景
    background_color: str = "#FFFFFF"
    face_color: str = "#FAFAFA"

    def apply_global_style(self):
        """应用全局 matplotlib 样式"""
        matplotlib.rcParams.update({
            "font.size": self.tick_fontsize,
            "axes.titlesize": self.title_fontsize,
            "axes.labelsize": self.axis_fontsize,
            "legend.fontsize": self.legend_fontsize,
            "figure.figsize": self.figure_size,
            "axes.grid": self.show_grid,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": self.grid_style,
        })

    def get_color(self, index: int) -> str:
        """获取指定索引的颜色"""
        return self.line_colors[index % len(self.line_colors)]

    def get_linestyle(self, index: int) -> str:
        """获取指定索引的线条样式"""
        return self.line_styles[index % len(self.line_styles)]


# 预设样式
STYLE_PRESETS = {
    "default": PlotStyle(),
    "dark": PlotStyle(
        background_color="#1E1E1E",
        face_color="#2D2D2D",
        line_colors=["#64B5F6", "#EF5350", "#81C784", "#FFB74D", "#CE93D8", "#4DD0E1"],
    ),
    "paper": PlotStyle(
        line_colors=["#000000", "#555555", "#000000", "#555555", "#888888", "#AAAAAA"],
        line_styles=["-", "--", "-.", ":", "-", "--"],
        show_grid=True,
        grid_alpha=0.2,
        line_width=1.5,
    ),
    "colorful": PlotStyle(
        line_colors=["#E91E63", "#3F51B5", "#009688", "#FFC107", "#795548", "#607D8B"],
        line_width=2.2,
    ),
}