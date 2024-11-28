from __future__ import annotations

from typing import TypeVar, Generic, Iterable
from manim.mobject.types.vectorized_mobject import VGroup, VMobject
from manim.mobject.text.tex_mobject import MathTex
from manim_dataflow_analysis.lattice import Lattice
from dataclasses import dataclass
from frozendict import frozendict


L = TypeVar("L")


@dataclass(frozen=True)
class AbstractEnvironment(Generic[L]):

    lattice: Lattice[L]
    variables: frozendict[str, L] = frozendict()

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

    def get(self, variable: str) -> L:
        return self.variables.get(variable, self.lattice.bottom())


RULE_PART_INDEX = 1
MODIFICATION_PART_INDEX = 3
IF_OTHERWISE_INDEX = 5
CONDITION_PART_INDEX = 7
LINE_LENGTH = 10

EMPTY_CHARACTER = r"\hspace{0pt}"


class AbstractEnvironmentUpdateRules(MathTex, Generic[L]):

    def __init__(self, updates_rules: Iterable[tuple[str, str, str]]) -> None:
        tex_strings: list[str] = []

        for rule_tex, modification_tex, condition_tex in updates_rules:
            tex_strings.extend(
                (
                    "&",
                    rule_tex,
                    r"& = & \quad" if rule_tex else "& &",
                    modification_tex,
                    "& &",
                    "if" if condition_tex else "otherwise",
                    "&",
                    condition_tex if condition_tex else EMPTY_CHARACTER,
                    "&",
                    r"\\",
                )
            )

        super().__init__(*tex_strings)

    def get_rule_part(self, index: int) -> VMobject:
        return self.submobjects[index * LINE_LENGTH + RULE_PART_INDEX]

    def get_condition_part(self, index: int) -> VMobject:
        if_otherwise = self.submobjects[index * LINE_LENGTH + IF_OTHERWISE_INDEX]
        condition_part = self.submobjects[index * LINE_LENGTH + CONDITION_PART_INDEX]

        if condition_part.get_tex_string() == EMPTY_CHARACTER:
            return if_otherwise

        return VGroup(if_otherwise, condition_part)

    def get_modification_part(self, index: int) -> VMobject:
        return self.submobjects[index * LINE_LENGTH + MODIFICATION_PART_INDEX]
