"""Let a PyTorch potential run on kUPS' fixed-size, partly padded particle arrays."""

from typing import override

import torch


class DropPadding(torch.nn.Module):
    """Hide padding atoms from ``inner`` and give them zero gradients.

    kUPS sorts atoms by system, so padding (outside every system) is a tail
    that can simply be sliced off; its neighbor-list slots already point past
    the real atoms and are dropped by the model.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    @override
    def forward(self, graph: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        n = int((graph["batch"] < len(graph["cell"])).sum())
        real = {
            k: v[:n] if k in ("pos", "atomic_numbers", "batch") else v
            for k, v in graph.items()
        }
        out = self.inner(real)
        gradients = graph["pos"].new_zeros(graph["pos"].shape)
        gradients[:n] = out["position_gradients"]
        return out | {"position_gradients": gradients}
