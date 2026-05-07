"""高清图表导出"""

import os
from typing import Optional
from matplotlib.figure import Figure


class Exporter:
    """将 matplotlib Figure 导出为多种高清格式"""

    SUPPORTED_FORMATS = {
        "png": {"dpi": 300, "description": "PNG 图片 (*.png)"},
        "jpg": {"dpi": 300, "description": "JPEG 图片 (*.jpg)"},
        "jpeg": {"dpi": 300, "description": "JPEG 图片 (*.jpeg)"},
        "pdf": {"dpi": 300, "description": "PDF 文档 (*.pdf)"},
        "svg": {"dpi": 300, "description": "SVG 矢量图 (*.svg)"},
    }

    @staticmethod
    def export(
        figure: Figure,
        filepath: str,
        dpi: int = 300,
        transparent: bool = False,
        bbox_inches: str = "tight",
        pad_inches: float = 0.1,
    ) -> str:
        """
        导出图表到文件。

        Args:
            figure: matplotlib Figure 对象
            filepath: 保存路径
            dpi: 分辨率
            transparent: 是否透明背景
            bbox_inches: 边界裁剪方式
            pad_inches: 边距

        Returns:
            实际保存的文件路径
        """
        ext = os.path.splitext(filepath)[1].lower().lstrip(".")

        if ext not in Exporter.SUPPORTED_FORMATS:
            filepath += ".png"
            ext = "png"

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        figure.savefig(
            filepath,
            format=ext if ext != "jpg" else "jpeg",
            dpi=dpi,
            transparent=transparent,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor=figure.get_facecolor() if not transparent else "none",
            edgecolor="none",
        )

        return filepath

    @staticmethod
    def get_file_filter() -> str:
        """获取文件对话框的过滤器字符串"""
        filters = []
        for ext, info in Exporter.SUPPORTED_FORMATS.items():
            if ext == "jpeg":
                continue
            filters.append(f"{info['description']}|*.{ext}")
        return "||".join(filters)

    @staticmethod
    def get_tk_filetypes() -> list:
        """获取 tkinter 文件对话框的 filetypes 参数"""
        return [
            ("PNG 图片", "*.png"),
            ("JPEG 图片", "*.jpg *.jpeg"),
            ("PDF 文档", "*.pdf"),
            ("SVG 矢量图", "*.svg"),
            ("所有文件", "*.*"),
        ]