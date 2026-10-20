"""Download every model and fill the compilation caches, ahead of the booth.

Run once (online) on the demo machine: ``uv run python -m playground.prepare``.
The server then starts offline from the caches. With ``--forcefield classical``
nothing is downloaded.
"""

import argparse
import time

from . import CACHE, configure_jax
from .gcmc import GrandCanonicalMC
from .md import (
    FORCEFIELDS,
    MACE_PATH,
    UMA_PATH,
    UNAVAILABLE,
    MolecularDynamics,
    use_forcefield,
)

MODELS = {"ml": (MACE_PATH, UMA_PATH), "classical": ()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forcefield", choices=FORCEFIELDS, default="ml")
    args = parser.parse_args()
    configure_jax()
    use_forcefield(args.forcefield)
    start = time.perf_counter()

    def status(text: str) -> None:
        print(f"[{time.perf_counter() - start:6.0f}s] {text}", flush=True)

    for uri in MODELS[args.forcefield]:
        from huggingface_hub import hf_hub_download

        owner, repo, filename = uri.removeprefix("hf://").split("/", 2)
        status(f"Downloading {filename}")
        hf_hub_download(f"{owner}/{repo}", filename)
    GrandCanonicalMC(args.forcefield).warm_up(status)
    MolecularDynamics().warm_up(status)
    if UNAVAILABLE:
        status(f"WARNING unavailable potentials: {UNAVAILABLE}")
    status(f"Done. Caches in {CACHE}")


if __name__ == "__main__":
    main()
