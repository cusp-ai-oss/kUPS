"""The UiO-66 + CO₂ structure that MD and GCMC hand to each other."""

from __future__ import annotations

from dataclasses import dataclass, replace

import ase.io
import numpy as np

from . import EXAMPLES

CO2 = ("C", "O", "O")  # atom order of every guest molecule, as in adsorbate/co2.yaml
MAX_CO2 = (
    15  # CO₂ slots in MD and GCMC; ~9 at 10 bar. 15 keeps MACE's edge buffer at 64/atom
)


@dataclass(frozen=True)
class Structure:
    """Framework atoms first, then one ``(C, O, O)`` triple per CO₂ molecule.

    Positions are Cartesian in kUPS' lower-triangular cell frame.
    ``framework_momenta`` is MD's, carried through GCMC (which has none).
    """

    framework: tuple[str, ...]
    framework_positions: np.ndarray  # (n_framework, 3)
    molecules: np.ndarray  # (n_molecules, 3, 3)
    cell: np.ndarray  # (3, 3), rows are lattice vectors
    framework_momenta: np.ndarray | None = None

    @property
    def symbols(self) -> list[str]:
        return [*self.framework, *CO2 * len(self.molecules)]

    @property
    def positions(self) -> np.ndarray:
        return np.concatenate([self.framework_positions, self.molecules.reshape(-1, 3)])

    def in_cell(self, cell: np.ndarray) -> Structure:
        """The same fractional coordinates in another (slightly strained) cell."""
        to_cell = np.linalg.solve(self.cell, cell)  # frac = r @ inv(h); r' = frac @ h'
        return replace(
            self,
            framework_positions=self.framework_positions @ to_cell,
            molecules=self.molecules @ to_cell,
            cell=cell,
        )


def pristine() -> Structure:
    """UiO-66 primitive cell with one CO₂ in each of its three cages."""
    mof = ase.io.read(EXAMPLES / "host" / "RUBTAK.cif")
    assert isinstance(mof, ase.Atoms)
    cell = mof.cell.array
    assert np.allclose(cell, np.tril(cell)), (
        "kUPS frame expects a lower-triangular cell"
    )
    rng = np.random.default_rng(0)
    molecules: list[list[np.ndarray]] = []
    for center in [(0, 0, 0), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]:
        c = np.array(center) @ cell
        axis = rng.normal(size=3)
        axis *= 1.16 / np.linalg.norm(axis)
        molecules.append([c, c + axis, c - axis])
    return Structure(
        framework=tuple(mof.get_chemical_symbols()),
        framework_positions=mof.positions,
        molecules=np.array(molecules),
        cell=cell,
    )
