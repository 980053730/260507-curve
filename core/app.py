import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

from plotting.canvas import InteractiveCanvas
from plotting.style import PlotStyle
from core.curve_generator import CurveGenerator
from core.data_parser import DataParser

# 兼容您的拼写错误，或者如果已经修正则尝试正常的 exporter
try:
    from core.exporter import Exporter
except ImportError:
    try:
        from core.expoter import Exporter
    except ImportError:
        Exporter = None


class CurvePlotterApp(tk.Frame):
    """Curve Plotter 主应用程序界面"""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        # 初始化样式
        self.style = PlotStyle()
        self.style.apply_global_style()

        self._setup_ui()

    def _setup_ui(self):
        """设置左右分栏 UI 布局"""
        self.paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧控制面板
        self.control_panel = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(self.control_panel, weight=0)

        # 右侧绘图区域
        self.canvas_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.canvas_panel, weight=1)

        self._setup_control_panel()
        self._setup_canvas()

    def _setup_control_panel(self):
        """设置左侧控制面板内容"""
        # ==================== 数据导入区 ====================
        load_frame = ttk.LabelFrame(self.control_panel, text="数据导入", padding=10)
        load_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(load_frame, text="打开日志文件 (CSV/JSON/TXT)",
                   command=self.load_data_file).pack(fill=tk.X, pady=5)

        # ==================== 曲线生成区 ====================
        gen_frame = ttk.LabelFrame(self.control_panel, text="模拟曲线生成", padding=10)
        gen_frame.pack(fill=tk.X, padx=5, pady=5)

        # Epochs
        ttk.Label(gen_frame, text="Epoch 数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.var_epochs = tk.IntVar(value=100)
        ttk.Entry(gen_frame, textvariable=self.var_epochs, width=10).grid(row=0, column=1, sticky=tk.EW, pady=2)

        # Loss 目标
        ttk.Label(gen_frame, text="目标 Loss:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.var_target_loss = tk.DoubleVar(value=0.2)
        ttk.Entry(gen_frame, textvariable=self.var_target_loss, width=10).grid(row=1, column=1, sticky=tk.EW, pady=2)

        # Acc 目标
        ttk.Label(gen_frame, text="目标 Acc:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.var_target_acc = tk.DoubleVar(value=95.5)
        ttk.Entry(gen_frame, textvariable=self.var_target_acc, width=10).grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Button(gen_frame, text="生成 Loss 与 Acc 曲线",
                   command=self.generate_mock_curves).grid(row=3, column=0, columnspan=2,sticky=tk.EW, pady=10)

        # ==================== 画布操作区 ====================
        action_frame = ttk.LabelFrame(self.control_panel, text="操作", padding=10)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(action_frame, text="清空画布", command=self.clear_canvas).pack(fill=tk.X, pady=5)
        ttk.Button(action_frame, text="导出高清图表", command=self.export_chart).pack(fill=tk.X, pady=5)

    def _setup_canvas(self):
        """初始化右侧画板"""
        self.plot_canvas = InteractiveCanvas(self.canvas_panel, style=self.style)
        self.plot_canvas.pack(fill=tk.BOTH, expand=True)

    def load_data_file(self):
        """通过 DataParser 导入真实训练数据"""
        filepath = filedialog.askopenfilename(
            title="选择训练日志文件",
            filetypes=(
                ("所有支持的文件", "*.csv *.json *.txt *.tfevents*"),
                ("CSV 文件", "*.csv"),
                ("JSON 文件", "*.json"),
                ("文本日志", "*.txt"),
                ("所有文件", "*.*")
            )
        )
        if not filepath:
            return

        try:
            data = DataParser.parse_file(filepath)
            if not data:
                messagebox.showwarning("警告", "未能从文件中解析出有效的指标数据。")
                return

            self.clear_canvas()

            # 基础 X 轴
            epochs = data.get("epochs")
            if epochs is None:
                messagebox.showerror("错误", "数据中缺失 epoch 序列。")
                return

            # 绘制解析出的每一条曲线
            has_loss = False
            has_acc = False

            for key, values in data.items():
                if key.lower() == "epochs":
                    continue

                # 简单判断是否需要放到右侧（副）Y轴
                is_acc = "acc" in key.lower()
                if is_acc:
                    has_acc = True
                else:
                    has_loss = True

                self.plot_canvas.add_curve(
                    name=key,
                    x=epochs,
                    y=values,
                    label=key,
                    is_secondary_axis=is_acc
                )

            # 配置轴标签
            self.plot_canvas.configure_axes(
                title="Training Logs",
                xlabel="Epochs",
                ylabel="Loss" if has_loss else None,
                ylabel2="Accuracy (%)" if has_acc else None
            )

        except Exception as e:
            messagebox.showerror("解析错误", f"读取文件时发生错误:\n{str(e)}")

    def generate_mock_curves(self):
        """调用 CurveGenerator 生成平滑的假数据，用于论文配图等场景"""
        try:
            epochs = self.var_epochs.get()
            target_loss = self.var_target_loss.get()
            target_acc = self.var_target_acc.get()
        except tk.TclError:
            messagebox.showerror("输入错误", "请输入有效的数字！")
            return

        self.clear_canvas()

        # 生成 Loss 数据对
        loss_data = CurveGenerator.generate_train_val_pair(
            train_target=target_loss,
            val_target=target_loss + 0.15,  # 模拟验证集 loss 稍微高一点
            epochs=epochs,
            curve_type="loss"
        )

        # 生成 Accuracy 数据对
        acc_data = CurveGenerator.generate_train_val_pair(
            train_target=target_acc,
            val_target=target_acc - 2.5,  # 模拟验证集 acc 稍微低一点
            epochs=epochs,
            curve_type="accuracy"
        )

        x = loss_data["epochs"]

        # 添加 Loss 曲线 (主轴)
        self.plot_canvas.add_curve("Train Loss", x, loss_data["train"], color=self.style.line_colors[0], linestyle="-")
        self.plot_canvas.add_curve("Val Loss", x, loss_data["val"], color=self.style.line_colors[1], linestyle="--")

        # 添加 Accuracy 曲线 (副轴)
        self.plot_canvas.add_curve("Train Acc", x, acc_data["train"], color=self.style.line_colors[2], linestyle="-",
                                   is_secondary_axis=True)
        self.plot_canvas.add_curve("Val Acc", x, acc_data["val"], color=self.style.line_colors[3], linestyle="--",
                                   is_secondary_axis=True)

        self.plot_canvas.configure_axes(
            title="Simulated Training Curves",
            xlabel="Epochs",
            ylabel="Loss",
            ylabel2="Accuracy"
        )

    def clear_canvas(self):
        """清空画布"""
        self.plot_canvas.clear()
        self.plot_canvas.configure_axes(xlabel="Epochs", ylabel="Value")

    def export_chart(self):
        """导出高分辨率图表"""
        if Exporter is None:
            messagebox.showerror("组件缺失", "Exporter 模块加载失败，请检查文件 core/exporter.py 是否存在。")
            return

        filepath = filedialog.asksaveasfilename(
            title="导出图表",
            defaultextension=".png",
            filetypes=Exporter.get_tk_filetypes()
        )
        if not filepath:
            return

        try:
            Exporter.export(
                figure=self.plot_canvas.figure,
                filepath=filepath,
                dpi=300,
                bbox_inches="tight"
            )
            messagebox.showinfo("导出成功", f"图表已成功保存至:\n{filepath}")
        except Exception as e:
            messagebox.showerror("导出失败", f"保存图表时发生错误:\n{str(e)}")