# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Machine learning interatomic potentials (MLIAPs).

This module provides interfaces to machine learning models for computing atomic
energies and forces. MLIAPs offer quantum-mechanical accuracy at classical force
field computational cost, enabling accurate simulations of complex systems.

## Available Models

- **[tojax][kups.potential.mliap.tojax]**: Generic jaxified MLFF models (exported JAX)
- **[local][kups.potential.mliap.local]**: Local MLIAP with single message passing and incremental updates
- **[torch.mace][kups.potential.mliap.torch.mace]**: PyTorch MACE models via TorchModuleWrapper
- **[interface][kups.potential.mliap.interface]**: Generic MLIAP potential factory

MLIAPs are trained on ab initio data and can capture complex many-body interactions,
bond breaking/forming, and reactive chemistry that classical force fields cannot.
"""
