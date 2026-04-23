# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ase",
#     "h5py",
#     "imageio",
#     "imageio-ffmpeg",
#     "matplotlib",
#     "numpy",
#     "pillow",
# ]
# ///
"""Render a batched kUPS MLFF relaxation trajectory as a side-by-side video.

Reads an ``.h5`` trajectory written by ``kups_relax_mlff`` and produces an
``.mp4`` showing each batched system as its own tile, aligned on a consistent
lattice axis with ``|F|max`` and energy per step. Useful for visualizing how
many independent structures relax in one shared JAX computation.

Example:
    uv run --script relax_batch_video.py relax_uma_mofs.h5 relax_batch.mp4
"""

from __future__ import annotations

import argparse
import io
import warnings

import h5py
import imageio.v2 as imageio
import numpy as np
from ase import Atoms
from ase.io import eps as ase_eps, write
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Disable ASE's default black outline around atoms
# ---------------------------------------------------------------------------
_orig_make_patch_list = ase_eps.make_patch_list


def _no_outline_make_patch_list(writer):
    patches = _orig_make_patch_list(writer)
    for p in patches:
        if isinstance(p, Circle):
            p.set_edgecolor(p.get_facecolor())
    return patches


ase_eps.make_patch_list = _no_outline_make_patch_list
warnings.filterwarnings("ignore")


def tril_to_cell(tril: np.ndarray) -> np.ndarray:
    L = np.zeros((3, 3))
    L[0, 0] = tril[0]
    L[1, 0] = tril[1]
    L[1, 1] = tril[2]
    L[2, 0] = tril[3]
    L[2, 1] = tril[4]
    L[2, 2] = tril[5]
    return L


def render_atoms(atoms: Atoms, w: int, h: int) -> Image.Image:
    # Pre-align cell `c` with screen +y so one lattice axis is vertical in
    # every tile regardless of the input lattice orientation.
    atoms = atoms.copy()
    atoms.rotate(atoms.cell[2], [0, 1, 0], rotate_cell=True)
    buf = io.BytesIO()
    write(
        buf,
        atoms,
        format="png",
        rotation="-20x,0y,0z",
        show_unit_cell=2,
        radii=0.5,
        maxwidth=w,
    )
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    arr = np.asarray(img).copy()
    # ASE draws the unit cell in black; recolor opaque-near-black pixels to
    # white so the cell stays visible on the black canvas.
    rgb = arr[..., :3]
    a = arr[..., 3]
    blackish = (rgb.max(axis=-1) < 40) & (a > 0)
    arr[blackish] = [255, 255, 255, 255]
    img = Image.fromarray(arr, mode="RGBA")
    img.thumbnail((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2), img)
    return canvas


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
        big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
        return small, big
    except OSError:
        fallback = ImageFont.load_default()
        return fallback, fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", help="kUPS relax trajectory HDF5 file.")
    parser.add_argument("out", help="Output mp4 path.")
    parser.add_argument("--tile-size", type=int, default=420)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--title",
        default="kUPS batched relaxation",
        help="Title text shown on top of the video.",
    )
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as f:
        step = f["group.step"]
        init = f["group.init"]
        sys_idx = init["array.atoms.data.system.indices"][:]
        Z = init["array.atoms.data.atomic_numbers"][:]
        all_pos = step["array.atoms.data.positions"][:]  # (T, N, 3)
        max_force = step["array.max_force"][:]  # (T, S)
        energy = step["array.potential_energy"][:]  # (T, S)
        init_cell_tril = init["array.systems.data.unitcell.tril"][:]

    n_sys = int(sys_idx.max()) + 1
    T = all_pos.shape[0]
    # kUPS writes zeros to max_force after batched convergence — trim the
    # frozen tail so it is not shown in the video.
    nonzero = np.where(max_force.max(axis=1) > 0)[0]
    T_eff = int(nonzero[-1]) + 1 if len(nonzero) else T

    frames_idx = list(range(0, T_eff, args.stride))
    if frames_idx[-1] != T_eff - 1:
        frames_idx.append(T_eff - 1)

    masks = [sys_idx == s for s in range(n_sys)]
    font_small, font_big = load_fonts()

    tile_w = args.tile_size
    tile_h = args.tile_size
    title_h = 36
    total_w = tile_w * n_sys
    total_h = tile_h + title_h

    def build_frame(t: int) -> Image.Image:
        pos_t = all_pos[t]
        tiles = []
        for s in range(n_sys):
            m = masks[s]
            atoms = Atoms(
                numbers=Z[m],
                positions=pos_t[m],
                cell=tril_to_cell(init_cell_tril[s]),
                pbc=True,
            )
            atoms.wrap()
            tile = render_atoms(atoms, tile_w, tile_h - 20)
            full = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 255))
            full.paste(tile, (0, 0), tile)
            d = ImageDraw.Draw(full)
            d.text(
                (8, tile_h - 18),
                f"n={int(m.sum())}",
                font=font_small,
                fill=(220, 220, 220),
            )
            d.text(
                (80, tile_h - 18),
                f"E={energy[t, s]:.1f} eV",
                font=font_small,
                fill=(220, 220, 220),
            )
            d.text(
                (200, tile_h - 18),
                f"|F|max={max_force[t, s]:.2f}",
                font=font_small,
                fill=(255, 120, 120) if max_force[t, s] > 1 else (120, 255, 120),
            )
            tiles.append(full)

        canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 255))
        d = ImageDraw.Draw(canvas)
        d.text(
            (12, 6),
            f"{args.title} — step {t + 1}/{T_eff}",
            font=font_big,
            fill=(255, 255, 255),
        )
        for s, tile in enumerate(tiles):
            canvas.paste(tile, (s * tile_w, title_h), tile)
        return canvas.convert("RGB")

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7)
    for i, t in enumerate(frames_idx):
        writer.append_data(np.asarray(build_frame(t)))
        if i % 10 == 0:
            print(f"  frame {i + 1}/{len(frames_idx)}  (step {t})")
    writer.close()
    print(f"Wrote {args.out}  ({len(frames_idx)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
