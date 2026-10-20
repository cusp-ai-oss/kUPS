# kUPS playground

A booth demo: UiO-66 with CO₂, simulated live in the browser. Molecular dynamics
and grand-canonical Monte Carlo share one structure: switching hands it over,
so GCMC starts from MD's framework and CO₂, and MD then evolves every molecule
GCMC adsorbed. Building blocks swap while the simulation runs.

- **MD**: potential (MACE-MPA-0 exported to JAX and Meta's UMA via the PyTorch
  bridge, or a classical force field) · thermostat · barostat · temperature.
- **GCMC**: rigid snapshot of MD's framework · Monte Carlo moves as toggleable
  blocks · temperature and pressure (fugacity) as data · isotherm from 8
  pressures batched as 8 systems in one state.
- **Hybrid**: flexible-framework GCMC: 20 GCMC cycles on a frozen snapshot,
  then 100 MD steps that relax framework and CO₂, repeated on the same MD and
  GCMC objects · isotherm compares rigid and flexible.

MD runs on fixed-size arrays with padding slots for up to 15 CO₂, so every
loading reuses the same compiled program; GCMC never inserts beyond 15. GCMC
uses MD's cutoff (6 Å for ML, 5.5 Å classical) so both methods fit the
123-atom primitive cell, which keeps MD fast; its uptake is qualitative.
The pipeline strip shows the running propagator, and the code panel highlights
the line that changed on each swap. "Play tour" walks through all of it.

## ML potentials (GPU machine)

Once, online:

```bash
uv sync --extra uma --extra hf --extra demo
hf auth login                  # UMA is gated: request access at huggingface.co/facebook/UMA
cd demo
uv run python -m playground.prepare
```

`prepare` downloads MACE and UMA and compiles every block combination into a
persistent cache (`~/.cache/kups-playground`, override with
`KUPS_PLAYGROUND_CACHE`), saving the compiled programs so later starts skip
tracing them (about 7 min). Rerun it after updating kUPS, JAX, torch or the
playground.

At the booth:

```bash
cd demo && uv run python -m playground.server          # on the GPU machine
ssh -L 8000:localhost:8000 <gpu-machine>              # on the laptop
```

Open http://localhost:8000 in full screen. The server runs offline from the
cache and is live after about 10 s on MACE. Background threads warm the rest,
the tour's blocks first (MACE after about 20 s, UMA after about 1 min, GCMC
after about 1.5 min, everything after about 2–3 min on an L4); until then a block
shows "warming…" and can't be selected. After that, every swap and every
switch between methods takes well under a second.

`--no-prewarm` compiles blocks on first click instead. Use `--port` if 8000 is
taken.

## Classical force field (any CPU, e.g. a MacBook)

```bash
cd demo
uv run --no-default-groups --extra demo python -m playground.prepare --forcefield classical
uv run --no-default-groups --extra demo python -m playground.server --forcefield classical
```

No GPU, torch or downloads. MD uses GCMC's nonbonded terms (UFF Lennard-Jones
and DDEC charges for UiO-66, TraPPE CO₂, Ewald) plus UFF4MOF bonded terms that
keep the framework near its crystal structure (`playground/classical.py`); CO₂
is flexible. MD and GCMC cut off at 5.5 Å so NPT cells stay wider than twice the
cutoff. The swap alternative is the same force field without charges.

`prepare` (optional, 5-7 min) fills the compile cache, which is specific to
the machine; later starts are live after about 10 s and fully warm in under a
minute. Without it, the first start is live after about 1 min and fully warm
after about 5 min. Open http://localhost:8000.

## Recorded tour (no server)

When the live server is out of reach, the page plays a recorded tour through
the same UI, labelled "Replay · recorded on …"; it goes live again as soon as
the server answers. `?replay` in the URL forces it. The `static/` folder alone
is a working page, e.g. on any static host:

```bash
python3 -m http.server -d demo/playground/static 8005
```

A page that must never look for a server, e.g. a bundle on a host without
gzip, adds `<meta name="kups-replay" content="recordings/tour.json">` to the head.

To record a new tour (on the GPU machine), run the server with `--record`,
wait until the log says "Everything warm", play one full tour loop, stop the
server and cut it:

```bash
cd demo
uv run python -m playground.server --record raw.jsonl.gz
uv run python -m playground.recording raw.jsonl.gz    # writes static/recordings/tour.json(.gz)
```
