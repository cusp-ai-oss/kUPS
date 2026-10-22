# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Small CO2-in-vacuum configurations for the rigid-body MCMC entry point."""

import tempfile

import jax.numpy as jnp

from kups.application.mcmc.analysis import analyze_mcmc_file
from kups.application.mcmc.data import AdsorbateConfig, HostConfig, RunConfig
from kups.application.simulations.mcmc_rigid import Config, EwaldConfig, LJConfig

BOX = 14.0  # box side (Å); cutoff 5.0 < box/2 for minimum image

# Hand-written empty-box CIF with one pseudo "X" host atom (no LJ params),
# i.e. CO2 adsorbing into vacuum; small enough to compile fast.
EMPTY_CIF = f"""data_cell
_cell_length_a    {BOX}
_cell_length_b    {BOX}
_cell_length_c    {BOX}
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
X1 X 0.0 0.0 0.0
"""


def tmp_file(suffix: str, content: str | None = None) -> str:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    if content is not None:
        f.write(content)
    f.close()
    return f.name


def co2() -> AdsorbateConfig:
    # Three-site charged CO2 (TraPPE): exercises Ewald + rigid rotation.
    return AdsorbateConfig(
        critical_temperature=303.75,
        critical_pressure=7.84e6,
        acentric_factor=0.22394,
        positions=((0.0, 0.0, 0.0), (-1.16, 0.0, 0.0), (1.16, 0.0, 0.0)),
        symbols=("C_co2", "O_co2", "O_co2"),
        charges=(0.7, -0.35, -0.35),
    )


def co2_config(
    *, exchange_prob: float, init_adsorbates: tuple[int, ...], charged: bool = True
) -> Config:
    return Config(
        adsorbates=(
            co2() if charged else co2().model_copy(update={"charges": (0.0,) * 3}),
        ),
        hosts=(
            HostConfig(
                cif_file=tmp_file(".cif", EMPTY_CIF),
                pressure=1e4,
                temperature=298.15,
                init_adsorbates=init_adsorbates,
                cell_replication=1,
            ),
        ),
        run=RunConfig(
            out_file=tmp_file(".h5"),
            num_cycles=2,
            num_warmup_cycles=0,
            min_cycle_length=1,
            exchange_prob=exchange_prob,
            seed=42,
        ),
        lj=LJConfig(
            cutoff=5.0,
            tail_correction=True,
            mixing_rule="lorentz_berthelot",
            # (sigma [Å], epsilon [eV]); X1 is the host pseudo-type (no LJ).
            parameters={
                "O_co2": (3.05, 0.0068077),
                "C_co2": (2.8, 0.0023267),
                "X1": (None, None),
            },
        ),
        ewald=EwaldConfig(real_cutoff=5.0, precision=1e-3),
        max_num_adsorbates=4,
    )


def batched_co2_config(n_hosts: int, init_adsorbates: tuple[int, ...]) -> Config:
    """A multi-host (batched) CO2 config; each host is an independent system."""
    base = co2_config(exchange_prob=0.5, init_adsorbates=init_adsorbates)
    host = base.hosts[0]
    return base.model_copy(update={"hosts": tuple(host for _ in range(n_hosts))})


def assert_readable(out_file: str) -> None:
    results = analyze_mcmc_file(out_file, n_blocks=2)
    assert len(results) == 1
    result = next(iter(results.values()))
    assert jnp.isfinite(result.energy.mean).all().item()
    assert jnp.isfinite(result.loading.mean).all().item()
    assert (result.loading.mean >= 0.0).all().item()
