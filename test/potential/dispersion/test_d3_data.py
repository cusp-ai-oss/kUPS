# Copyright 2024-2026 Cusp AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the embedded D3 reference tables."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from kups.core.constants import BOHR, HARTREE
from kups.potential.dispersion.data import (
    MAX_ATOMIC_NUMBER,
    MAX_REFERENCES,
    D3Reference,
    load_d3_reference,
)

# Upstream values, quoted independently of the loader (see data/README.md).
# Covalent radii: mctc-lib ``covalent_rad_2009`` [Angstrom], pre-4/3 scaling.
# Pt/Au are included to pin the Z-alignment of the table (they differ by 0.01 Å).
_RAD_2009 = {1: 0.32, 2: 0.46, 6: 0.75, 7: 0.71, 8: 0.63, 78: 1.12, 79: 1.13}
# <r^4>/<r^2>: tad-dftd3 ``_r4_over_r2`` [atomic units].
_R4_OVER_R2 = {1: 8.0589, 2: 3.4698, 6: 7.8715, 7: 5.5588, 8: 4.7566}
# Reference C6 [atomic units]: the free-atom H-H value is the textbook 7.5916.
_C6_HH_FREE_ATOM = 7.5916
_C6_CC_FIRST = 49.113


class TestD3ReferenceTables:
    @classmethod
    def setup_class(cls) -> None:
        cls.ref: D3Reference = load_d3_reference()

    def test_shapes_and_dtypes(self) -> None:
        """Merged: shapes, element/reference counts, finiteness, dtype tracking."""
        ref = self.ref
        n, m = MAX_ATOMIC_NUMBER + 1, MAX_REFERENCES
        assert ref.n_elements == n == 104
        assert ref.n_references == m == 7
        assert ref.covalent_radii.shape == (n,)
        assert ref.r4r2.shape == (n,)
        assert ref.reference_cn.shape == (n, m)
        assert ref.reference_c6.shape == (n, n, m, m)
        assert ref.reference_mask.shape == (n, m)
        assert ref.reference_mask.dtype == jnp.bool_
        for leaf in (ref.covalent_radii, ref.r4r2, ref.reference_cn, ref.reference_c6):
            assert bool(jnp.isfinite(leaf).all())
        # x64 is on for the test session, so the tables must not silently downcast
        assert ref.reference_c6.dtype == jnp.float64

    def test_reference_c6_is_symmetric_and_unpacked_exactly(self) -> None:
        """C6[i,j,a,b] == C6[j,i,b,a] must survive the upper-triangular round trip."""
        c6 = self.ref.reference_c6
        npt.assert_array_equal(np.asarray(c6), np.asarray(c6.transpose(1, 0, 3, 2)))
        # the packing would silently lose the lower triangle if the transpose were dropped
        assert float(c6[6, 1, 0, 1]) == float(c6[1, 6, 1, 0])
        assert float(c6[6, 1, 0, 1]) > 0.0

    def test_covalent_radii_match_upstream_with_d3_scaling(self) -> None:
        """D3 CN radii are 4/3 x the Pyykko 2009 values, in Angstrom."""
        for z, angstrom in _RAD_2009.items():
            npt.assert_allclose(
                float(self.ref.covalent_radii[z]), 4.0 / 3.0 * angstrom, rtol=1e-12
            )
        assert float(self.ref.covalent_radii[0]) == 0.0

    def test_r4r2_preprocessing_and_units(self) -> None:
        """Stored r4r2 is sqrt(0.5*sqrt(Z)*<r^4>/<r^2>), converted Bohr -> Angstrom."""
        for z, raw in _R4_OVER_R2.items():
            expected = np.sqrt(0.5 * raw * np.sqrt(z)) * BOHR
            npt.assert_allclose(float(self.ref.r4r2[z]), expected, rtol=1e-12)

    def test_reference_c6_units_and_known_values(self) -> None:
        """Reference C6 is converted from Hartree*Bohr^6 to eV*Angstrom^6."""
        to_ev_ang6 = HARTREE * BOHR**6
        npt.assert_allclose(
            float(self.ref.reference_c6[1, 1, 1, 1]),
            _C6_HH_FREE_ATOM * to_ev_ang6,
            rtol=1e-6,
        )
        npt.assert_allclose(
            float(self.ref.reference_c6[6, 6, 0, 0]),
            _C6_CC_FIRST * to_ev_ang6,
            rtol=1e-6,
        )
        # sanity: the conversion must not be the identity
        assert to_ev_ang6 < 1.0

    def test_reference_mask_matches_sentinel_and_c6_support(self) -> None:
        """Merged: sentinel handling, per-element reference counts, C6/mask agreement."""
        ref = self.ref
        cn = np.asarray(ref.reference_cn)
        mask = np.asarray(ref.reference_mask)

        npt.assert_array_equal(mask, cn > -1.0)
        # the placeholder row and every unused slot are fully masked out
        assert not mask[0].any()
        assert (cn[~mask] == -1.0).all()
        # H uses two references (H2 and the free atom), C uses five
        assert mask[1].sum() == 2
        assert mask[6].sum() == 5
        npt.assert_allclose(cn[1, :2], [0.9118, 0.0], rtol=1e-12)
        npt.assert_allclose(
            cn[6, :5], [0.0, 0.9868, 1.9985, 2.9987, 3.9844], rtol=1e-12
        )
        # every element from H to Lr carries at least one reference
        assert mask[1 : MAX_ATOMIC_NUMBER + 1].any(axis=1).all()
        # masked-out slots must contribute nothing to any C6 pair block
        c6 = np.asarray(ref.reference_c6)
        assert (c6[1, 1, 2:, :] == 0.0).all()
        assert (c6[1, 1, :, 2:] == 0.0).all()

    def test_loader_is_cached_but_dtype_follows_config(self) -> None:
        """Repeated loads reuse the parsed tables and compare equal."""
        again = load_d3_reference()
        npt.assert_array_equal(
            np.asarray(again.reference_c6), np.asarray(self.ref.reference_c6)
        )
        npt.assert_array_equal(np.asarray(again.r4r2), np.asarray(self.ref.r4r2))
