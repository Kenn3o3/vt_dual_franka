import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Spatial softmax that returns keypoint coordinates.

    Input shape: (B, C, H, W)
    Output shape: (B, K * 2), where K is num_kp (or C if num_kp is None)
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        c, h, w = input_shape
        self._input_c = c
        self._h = h
        self._w = w
        self._num_kp = c if num_kp is None else int(num_kp)

        if self._num_kp != c:
            self.proj = nn.Conv2d(c, self._num_kp, kernel_size=1)
        else:
            self.proj = None

        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, w),
            torch.linspace(-1.0, 1.0, h),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, h * w))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, h * w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape {tuple(x.shape)}")
        b, c, h, w = x.shape
        if h != self._h or w != self._w:
            raise ValueError(
                f"Expected spatial shape ({self._h}, {self._w}), got ({h}, {w})"
            )

        if self.proj is not None:
            x = self.proj(x)

        x = x.reshape(b, self._num_kp, h * w)
        attn = F.softmax(x, dim=-1)

        expected_x = torch.sum(self.pos_x * attn, dim=-1)
        expected_y = torch.sum(self.pos_y * attn, dim=-1)
        keypoints = torch.cat([expected_x, expected_y], dim=-1)
        return keypoints
