from __future__ import annotations

from typing import TypeVar, Generic
from manim_dataflow_analysis.lattice import Lattice


L = TypeVar("L")


class AbstractEnvironment(Generic[L]):

    def __init__(self, lattice: Lattice[L], **variables: L) -> None:
        self.__lattice = lattice
        self.__variables = variables

    def join(self, other: AbstractEnvironment[L]) -> AbstractEnvironment[L]:
        joined_variables: dict[str, L] = {}

        for variable in self.__variables | other.__variables:
            self_abstract_value = self.__variables.get(variable)
            other_abstract_value = other.__variables.get(variable)

            if self_abstract_value is None:
                joined_variables[variable] = other_abstract_value
            elif other_abstract_value is None:
                joined_variables[variable] = self_abstract_value
            else:
                joined_variables[variable] = self.__lattice.join(
                    self_abstract_value, other_abstract_value
                )

        return AbstractEnvironment(self.__lattice, **joined_variables)

    def includes(self, other: AbstractEnvironment[L]) -> bool:
        for variable in self.__variables | other.__variables:
            self_abstract_value = self.__variables.get(variable)
            other_abstract_value = other.__variables.get(variable)

            if self_abstract_value is None or (
                other_abstract_value is not None
                and not self.__lattice.includes(
                    self_abstract_value, other_abstract_value
                )
            ):
                return False

        return True

    def set(self, **variables: L) -> AbstractEnvironment[L]:
        return AbstractEnvironment(self.__lattice, **self.__variables, **variables)

    def get(self, variable: str) -> L | None:
        return self.__variables.get(variable)
