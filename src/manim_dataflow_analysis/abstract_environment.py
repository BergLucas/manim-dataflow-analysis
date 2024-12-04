from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator, Generic, Iterable, Mapping, TypeVar

from frozendict import frozendict
from manim.mobject.text.tex_mobject import MathTex, SingleStringMathTex
from manim.mobject.types.vectorized_mobject import VGroup, VMobject

if TYPE_CHECKING:
    from manim_dataflow_analysis.lattice import Lattice

L = TypeVar("L")


@dataclass(frozen=True)
class AbstractEnvironment(Generic[L]):
    lattice: Lattice[L]
    variables: frozendict[str, L] = frozendict()

    def join_generator(
        self, other: AbstractEnvironment[L]
    ) -> Generator[tuple[str, L], None, None]:
        for variable in self.variables | other.variables:
            yield variable, self.lattice.join(
                self.variables.get(variable, self.lattice.bottom()),
                other.variables.get(variable, self.lattice.bottom()),
            )

    def join(self, other: AbstractEnvironment[L]) -> AbstractEnvironment[L]:
        return AbstractEnvironment(
            self.lattice,
            frozendict(self.join_generator(other)),
        )

    def includes_generator(
        self, other: AbstractEnvironment[L]
    ) -> Generator[tuple[str, bool], None, None]:
        for variable in self.variables | other.variables:
            self_abstract_value = self.variables.get(variable)
            other_abstract_value = other.variables.get(variable)

            yield variable, self_abstract_value is None or (
                other_abstract_value is not None
                and self.lattice.includes(self_abstract_value, other_abstract_value)
            )

    def includes(self, other: AbstractEnvironment[L]) -> bool:
        return all(included for _, included in self.includes_generator(other))

    def set(self, variables: Mapping[str, L]) -> AbstractEnvironment[L]:
        return AbstractEnvironment(self.lattice, self.variables | variables)

    def __getitem__(self, variable: str) -> L:
        return self.variables[variable]

    def __contains__(self, variable: str) -> bool:
        return variable in self.variables


INSTANCE_PART_INDEX = 1
MODIFICATION_PART_INDEX = 3
IF_OTHERWISE_INDEX = 5
CONDITION_PART_INDEX = 7
LINE_LENGTH = 10

EMPTY_CHARACTER = r"\hspace{0pt}"


class AbstractEnvironmentUpdateInstances(MathTex):
    def __init__(
        self, updates_instances: Iterable[tuple[str, str, str | None]]
    ) -> None:
        tex_strings: list[str] = []

        for instance_tex, modification_tex, condition_tex in updates_instances:
            tex_strings.extend(
                (
                    r"&",
                    instance_tex,
                    r"& = & \quad",
                    modification_tex,
                    r"& &",
                    (
                        EMPTY_CHARACTER
                        if condition_tex is None
                        else r"if"
                        if condition_tex
                        else r"otherwise"
                    ),
                    r"&",
                    (
                        EMPTY_CHARACTER
                        if condition_tex is None or not condition_tex
                        else condition_tex
                    ),
                    r"&",
                    r"\\",
                )
            )

        super().__init__(*tex_strings)

    def get_instance_part(self, index: int) -> VMobject:
        return self.submobjects[index * LINE_LENGTH + INSTANCE_PART_INDEX]

    def get_condition_part(self, index: int) -> VMobject | None:
        if_otherwise = self.submobjects[index * LINE_LENGTH + IF_OTHERWISE_INDEX]
        condition_part = self.submobjects[index * LINE_LENGTH + CONDITION_PART_INDEX]

        assert isinstance(if_otherwise, SingleStringMathTex)
        assert isinstance(condition_part, SingleStringMathTex)

        if (
            condition_part.get_tex_string() != EMPTY_CHARACTER
            and if_otherwise.get_tex_string() != EMPTY_CHARACTER
        ):
            return VGroup(if_otherwise, condition_part)
        elif (
            condition_part.get_tex_string() != EMPTY_CHARACTER
            and if_otherwise.get_tex_string() == EMPTY_CHARACTER
        ):
            return condition_part
        elif (
            condition_part.get_tex_string() == EMPTY_CHARACTER
            and if_otherwise.get_tex_string() != EMPTY_CHARACTER
        ):
            return if_otherwise
        else:
            return None

    def get_modification_part(self, index: int) -> VMobject:
        return self.submobjects[index * LINE_LENGTH + MODIFICATION_PART_INDEX]
