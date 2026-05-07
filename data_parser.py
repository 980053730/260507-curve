"""多格式训练数据解析器"""

import json
import csv
import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from tbparse import SummaryReader
    HAS_TBPARSE = True
except ImportError:
    HAS_TBPARSE = False


class DataParser:
    """支持 CSV / JSON / TXT / TensorBoard 日志的训练数据解析"""

    @staticmethod
    def parse_file(filepath: str) -> Dict[str, np.ndarray]:
        """
        根据文件扩展名自动选择解析方式。

        Returns:
            字典，键为指标名称（如 'train_loss', 'val_accuracy'），
            值为 numpy 数组
        """
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".csv":
            return DataParser.parse_csv(filepath)
        elif ext == ".json":
            return DataParser.parse_json(filepath)
        elif ext == ".txt":
            return DataParser.parse_txt(filepath)
        elif ext in (".tfevents", "") and "events.out" in os.path.basename(filepath):
            return DataParser.parse_tensorboard(filepath)
        else:
            # 尝试作为目录解析 TensorBoard 日志
            if os.path.isdir(filepath):
                return DataParser.parse_tensorboard(filepath)
            raise ValueError(f"不支持的文件格式: {ext}")

    @staticmethod
    def parse_csv(filepath: str) -> Dict[str, np.ndarray]:
        """
        解析 CSV 文件。

        期望格式：
            epoch,train_loss,val_loss,train_acc,val_acc
            1,2.35,2.41,0.21,0.19
            ...
        """
        data = {}

        if HAS_PANDAS:
            df = pd.read_csv(filepath)
            for col in df.columns:
                col_lower = col.strip().lower()
                if col_lower in ("epoch", "step", "iteration"):
                    data["epochs"] = df[col].values.astype(float)
                else:
                    data[col.strip()] = df[col].values.astype(float)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if not rows:
                return data

            for key in rows[0].keys():
                key_lower = key.strip().lower()
                values = []
                for row in rows:
                    try:
                        values.append(float(row[key]))
                    except (ValueError, TypeError):
                        values.append(np.nan)

                arr = np.array(values)
                if key_lower in ("epoch", "step", "iteration"):
                    data["epochs"] = arr
                else:
                    data[key.strip()] = arr

        if "epochs" not in data and len(data) > 0:
            first_key = next(iter(data))
            data["epochs"] = np.arange(1, len(data[first_key]) + 1, dtype=float)

        return data

    @staticmethod
    def parse_json(filepath: str) -> Dict[str, np.ndarray]:
        """
        解析 JSON 文件。

        支持格式：
        1. {"train_loss": [...], "val_loss": [...], ...}
        2. [{"epoch": 1, "train_loss": 2.3, ...}, ...]
        """
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data = {}

        if isinstance(raw, dict):
            for key, values in raw.items():
                if isinstance(values, list):
                    key_lower = key.strip().lower()
                    arr = np.array(values, dtype=float)
                    if key_lower in ("epoch", "epochs", "step"):
                        data["epochs"] = arr
                    else:
                        data[key.strip()] = arr

        elif isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], dict):
            keys = raw[0].keys()
            for key in keys:
                values = []
                for entry in raw:
                    try:
                        values.append(float(entry.get(key, np.nan)))
                    except (ValueError, TypeError):
                        values.append(np.nan)
                arr = np.array(values)
                key_lower = key.strip().lower()
                if key_lower in ("epoch", "epochs", "step"):
                    data["epochs"] = arr
                else:
                    data[key.strip()] = arr

        if "epochs" not in data and len(data) > 0:
            first_key = next(iter(data))
            data["epochs"] = np.arange(1, len(data[first_key]) + 1, dtype=float)

        return data

    @staticmethod
    def parse_txt(filepath: str) -> Dict[str, np.ndarray]:
        """
        解析 TXT 格式的训练日志。

        支持常见格式如：
            Epoch 1/100 - loss: 2.3456 - acc: 0.2134 - val_loss: 2.4521 - val_acc: 0.1987
            Epoch [1/100] train_loss=2.3456 train_acc=0.2134 val_loss=2.4521 val_acc=0.1987
        """
        patterns = [
            # Keras 风格: loss: 0.1234
            re.compile(r"(\w+):\s*([\d.]+(?:e[+-]?\d+)?)"),
            # 等号风格: train_loss=0.1234
            re.compile(r"(\w+)\s*=\s*([\d.]+(?:e[+-]?\d+)?)"),
        ]

        epoch_pattern = re.compile(r"[Ee]poch\s*\[?(\d+)")

        collected: Dict[str, List[float]] = {}
        epochs_list: List[float] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 提取 epoch 编号
                epoch_match = epoch_pattern.search(line)
                if epoch_match:
                    epochs_list.append(float(epoch_match.group(1)))

                # 提取指标
                for pattern in patterns:
                    matches = pattern.findall(line)
                    for key, value in matches:
                        key_lower = key.lower()
                        if key_lower in ("epoch", "epochs", "step", "batch"):
                            continue
                        if key not in collected:
                            collected[key] = []
                        try:
                            collected[key].append(float(value))
                        except ValueError:
                            pass

        data = {}
        for key, values in collected.items():
            if len(values) > 0:
                data[key] = np.array(values)

        if epochs_list:
            data["epochs"] = np.array(epochs_list)
        elif len(data) > 0:
            first_key = next(iter(data))
            data["epochs"] = np.arange(1, len(data[first_key]) + 1, dtype=float)

        return data

    @staticmethod
    def parse_tensorboard(log_path: str) -> Dict[str, np.ndarray]:
        """
        解析 TensorBoard 日志文件或目录。

        需要安装 tbparse 库。
        """
        if not HAS_TBPARSE:
            raise ImportError(
                "解析 TensorBoard 日志需要安装 tbparse: pip install tbparse"
            )

        reader = SummaryReader(log_path)
        df = reader.scalars

        data = {}
        if df is not None and not df.empty:
            tags = df["tag"].unique()
            for tag in tags:
                tag_data = df[df["tag"] == tag].sort_values("step")
                clean_name = tag.replace("/", "_").strip()
                data[clean_name] = tag_data["value"].values
                if "epochs" not in data:
                    data["epochs"] = tag_data["step"].values.astype(float)

        return data