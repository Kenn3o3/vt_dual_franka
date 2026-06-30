from typing import Optional, Dict
import json
import os
import tempfile


class TopKCheckpointManager:
    def __init__(
        self,
        save_dir,
        monitor_key: str,
        mode="min",
        k=1,
        format_str="epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt",
    ):
        assert mode in ["max", "min"]
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def _write_best_info(self, ckpt_path: str, data: Dict[str, float], value: float) -> None:
        info_path = os.path.join(self.save_dir, "best.info.json")
        payload = {
            "checkpoint": os.path.basename(ckpt_path),
            "monitor_key": self.monitor_key,
            "monitor_value": float(value),
            "mode": self.mode,
        }
        for key in ("epoch", "global_step", "train_loss", "val_loss", "lr"):
            if key in data:
                try:
                    payload[key] = float(data[key])
                except (TypeError, ValueError):
                    payload[key] = data[key]
        os.makedirs(self.save_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self.save_dir, prefix=".tmp.best.info.json.") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            tmp_info_path = f.name
        os.replace(tmp_info_path, info_path)

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None
        if self.monitor_key not in data:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            self._write_best_info(ckpt_path, data, value)
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == "max":
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            self._write_best_info(ckpt_path, data, value)
            return ckpt_path
