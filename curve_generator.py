"""自然收敛曲线生成算法"""

import numpy as np
from typing import Tuple, Optional


class CurveGenerator:
    """生成模拟真实训练过程的收敛曲线"""

    @staticmethod
    def generate_loss_curve(
            target_value: float,
            epochs: int = 100,
            initial_value: Optional[float] = None,
            noise_scale: float = 0.02,
            decay_rate: float = 3.0,
            seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成一条收敛到目标值的 Loss 曲线。

        Args:
            target_value: 最终收敛的目标 Loss 值
            epochs: 总 Epoch 数
            initial_value: 初始 Loss 值，默认自动推断
            noise_scale: 噪声幅度比例，模拟训练中的自然波动
            decay_rate: 衰减速率，控制收敛快慢
            seed: 随机种子，用于复现

        Returns:
            (epochs_array, loss_values) 元组
        """
        if seed is not None:
            np.random.seed(seed)

        if initial_value is None:
            initial_value = target_value + np.random.uniform(1.5, 3.0)

        x = np.arange(1, epochs + 1, dtype=float)
        t = x / epochs

        # 指数衰减基础曲线
        base_curve = target_value + (initial_value - target_value) * np.exp(-decay_rate * t)

        # 添加与 epoch 相关的自然噪声（前期波动大，后期波动小）
        noise_envelope = (1 - t) ** 1.5
        noise = np.random.normal(0, noise_scale * (initial_value - target_value), epochs)
        noise = noise * noise_envelope

        # 添加偶尔的小幅跳跃，模拟学习率调整等事件
        spike_mask = np.random.random(epochs) < 0.03
        spikes = spike_mask * np.random.uniform(0.05, 0.15, epochs) * (initial_value - target_value) * noise_envelope

        loss_values = base_curve + noise + spikes
        loss_values = np.maximum(loss_values, target_value * 0.8)

        return x, loss_values

    @staticmethod
    def generate_accuracy_curve(
            target_value: float,
            epochs: int = 100,
            initial_value: Optional[float] = None,
            noise_scale: float = 0.01,
            growth_rate: float = 3.0,
            seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成一条收敛到目标值的 Accuracy 曲线。

        Args:
            target_value: 最终收敛的目标 Accuracy 值 (0-1 或 0-100)
            epochs: 总 Epoch 数
            initial_value: 初始 Accuracy 值
            noise_scale: 噪声幅度比例
            growth_rate: 增长速率
            seed: 随机种子

        Returns:
            (epochs_array, accuracy_values) 元组
        """
        if seed is not None:
            np.random.seed(seed)

        max_val = 100.0 if target_value > 1.0 else 1.0

        if initial_value is None:
            if max_val == 100.0:
                initial_value = np.random.uniform(10, 30)
            else:
                initial_value = np.random.uniform(0.1, 0.3)

        x = np.arange(1, epochs + 1, dtype=float)
        t = x / epochs

        # 对数增长基础曲线
        base_curve = target_value - (target_value - initial_value) * np.exp(-growth_rate * t)

        # 自然噪声
        noise_envelope = (1 - t) ** 1.5
        noise = np.random.normal(0, noise_scale * (target_value - initial_value), epochs)
        noise = noise * noise_envelope

        # 偶尔的小幅下降，模拟训练中的波动
        dip_mask = np.random.random(epochs) < 0.03
        dips = dip_mask * np.random.uniform(-0.03, -0.01, epochs) * (target_value - initial_value) * noise_envelope

        accuracy_values = base_curve + noise + dips
        accuracy_values = np.clip(accuracy_values, 0, max_val)

        return x, accuracy_values

    @staticmethod
    def generate_train_val_pair(
            train_target: float,
            val_target: float,
            epochs: int = 100,
            curve_type: str = "loss",
            gap_epoch: Optional[int] = None,
            seed: Optional[int] = None,
    ) -> dict:
        """
        生成训练集和验证集的配对曲线，自动模拟过拟合间隙。

        Args:
            train_target: 训练集最终值
            val_target: 验证集最终值
            epochs: 总 Epoch 数
            curve_type: "loss" 或 "accuracy"
            gap_epoch: 训练/验证曲线开始分离的 epoch
            seed: 随机种子

        Returns:
            包含 train 和 val 数据的字典
        """
        if seed is not None:
            np.random.seed(seed)

        if gap_epoch is None:
            gap_epoch = int(epochs * np.random.uniform(0.3, 0.5))

        if curve_type == "loss":
            x, train_curve = CurveGenerator.generate_loss_curve(
                train_target, epochs, seed=seed
            )
            _, val_curve = CurveGenerator.generate_loss_curve(
                val_target, epochs, noise_scale=0.03, seed=seed + 1 if seed else None
            )
        else:
            x, train_curve = CurveGenerator.generate_accuracy_curve(
                train_target, epochs, seed=seed
            )
            _, val_curve = CurveGenerator.generate_accuracy_curve(
                val_target, epochs, noise_scale=0.015, seed=seed + 1 if seed else None
            )

        # 前 gap_epoch 个点让两条曲线接近
        blend = np.minimum(np.arange(epochs) / gap_epoch, 1.0)
        val_curve_blended = train_curve * (1 - blend) + val_curve * blend

        return {
            "epochs": x,
            "train": train_curve,
            "val": val_curve_blended,
        }