from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint
from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.mobject import Group
from typing import TypeVar, Generic, Protocol
from abc import abstractmethod

L = TypeVar("L")


class ControlFlowFunctions(Protocol[L]):
    @abstractmethod
    def functions(self) -> tuple[tuple[str, str], ...]: ...

    @abstractmethod
    def apply(
        self,
        abstract_environment: AbstractEnvironment[L],
        program_point: ProgramPoint,
    ) -> tuple[AbstractEnvironment[L], int] | None: ...


LINE_LENGTH = 9
CONDITION_PART_INDEX = 6
MODIFICATION_PART_INDEX = 2


class ControlFlowFunctionsTex(MathTex, Generic[L]):

    def __init__(self, control_flow_functions: ControlFlowFunctions[L]):
        tex_strings: list[str] = []

        for modification_tex, condition_tex in control_flow_functions.functions():
            tex_strings.extend(
                (
                    "&",
                    r"fg[[p]](\phi) = &" if not tex_strings else "&",
                    modification_tex,
                    r"& \quad",
                    "if" if condition_tex else "otherwise",
                    "&",
                    condition_tex,
                    "&",
                    r"\\",
                )
            )

        super().__init__(*tex_strings)

    def get_condition_part(self, index: int) -> Group:
        return self[index * LINE_LENGTH + CONDITION_PART_INDEX]

    def get_modification_part(self, index: int) -> Group:
        return self[index * LINE_LENGTH + MODIFICATION_PART_INDEX]
