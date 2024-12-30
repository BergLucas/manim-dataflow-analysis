from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, TypeVar, Protocol, Sequence, Generator

from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.types.vectorized_mobject import VMobject
from manim_dataflow_analysis.abstract_environment import AbstractEnvironment

L = TypeVar("L")


INSTANCE_PART_INDEX = 1
MODIFICATION_PART_INDEX = 3
LINE_LENGTH = 5


class WideningOperatorTex(MathTex):
    def __init__(self, updates_instances: Iterable[tuple[str, str]]) -> None:
        tex_strings: list[str] = []

        for instance_tex, modification_tex in updates_instances:
            tex_strings.extend(
                (
                    r"&",
                    instance_tex,
                    r"& = & \quad",
                    modification_tex,
                    r"& \\",
                )
            )

        super().__init__(*tex_strings)

    def get_instance_part(self, index: int) -> VMobject:
        return self.submobjects[index * LINE_LENGTH + INSTANCE_PART_INDEX]

    def get_modification_part(self, index: int) -> VMobject:
        return self.submobjects[index * LINE_LENGTH + MODIFICATION_PART_INDEX]


class WideningOperator(Protocol[L]):
    @property
    @abstractmethod
    def instances(self) -> Sequence[tuple[str, str]]: ...

    @abstractmethod
    def apply(
        self,
        last_value: L,
        new_value: L,
    ) -> tuple[L, int]: ...

    def join_generator(
        self,
        last_environment: AbstractEnvironment[L],
        new_environment: AbstractEnvironment[L],
    ) -> Generator[tuple[str, L, int], None, None]:
        for variable in last_environment.variables | new_environment.variables:
            yield (
                variable,
                *self.apply(
                    last_environment.variables.get(
                        variable, last_environment.lattice.bottom()
                    ),
                    new_environment.variables.get(
                        variable, new_environment.lattice.bottom()
                    ),
                ),
            )
