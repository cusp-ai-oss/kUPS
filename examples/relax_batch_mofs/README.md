# Batched MOF relaxation with UMA-S-1.2 (ODAC)

End-to-end example: five MOFs (90–148 atoms, various metals) relaxed
**simultaneously** in a single batched kUPS computation using UMA-S-1.2
with the ODAC task head, then rendered side-by-side.

## Layout

```
relax_batch_mofs/
├── relax_uma_mofs.yaml      # kups_relax_mlff config
├── relax_batch_video.py     # uv run --script trajectory → mp4
├── convert_uma.sh           # one-shot UMA .pt → tojaxed .zip
├── host/*.cif               # 5 MOFs
└── mlffs/uma-s-1p2_odac.zip # tojaxed UMA (not committed — 35 MB)
```

## Run

```sh
# 1. (once) Convert UMA checkpoint to a JAX-export .zip
./convert_uma.sh /path/to/uma-s-1p2.pt

# 2. Relax all five MOFs in one batched JAX graph (~5 min on an L4)
kups_relax_mlff relax_uma_mofs.yaml

# 3. Render the trajectory as a side-by-side video
uv run --script relax_batch_video.py relax_uma_mofs.h5 relax_batch.mp4
```

Convergence tolerance is `0.05 eV/Å`; all five systems reach it around step
~1000 on the default config.
