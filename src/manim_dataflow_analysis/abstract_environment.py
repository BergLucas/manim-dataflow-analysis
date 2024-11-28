from __future__ import annotations

from typing import TypeVar, Generic
from manim_dataflow_analysis.lattice import Lattice
from dataclasses import dataclass
from frozendict import frozendict


L = TypeVar("L")


@dataclass(frozen=True)
class AbstractEnvironment(Generic[L]):

    lattice: Lattice[L]
    variables: frozendict[str, L]

    def join(self, other: AbstractEnvironment[L]) -> AbstractEnvironment[L]:
        return AbstractEnvironment(
            self.lattice,
            frozendict(
                (
                    variable,
                    self.lattice.join(
                        self.variables.get(variable, self.lattice.bottom()),
                        other.variables.get(variable, self.lattice.bottom()),
                    ),
                )
                for variable in self.variables | other.variables
            ),
        )

    def includes(self, other: AbstractEnvironment[L]) -> bool:
        for variable in self.variables | other.variables:
            self_abstract_value = self.variables.get(variable)
            other_abstract_value = other.variables.get(variable)

            if self_abstract_value is None or (
                other_abstract_value is not None
                and not self.lattice.includes(self_abstract_value, other_abstract_value)
            ):
                return False

        return True

    def set(self, **variables: L) -> AbstractEnvironment[L]:
        return AbstractEnvironment(
            self.lattice, frozendict(**self.variables, **variables)
        )
