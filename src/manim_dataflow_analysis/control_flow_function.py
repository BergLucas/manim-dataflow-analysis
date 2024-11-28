from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint
from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.mobject import Group
from typing import TypeVar, Generic, Protocol
from dataclasses import dataclass
from abc import abstractmethod

L = TypeVar("L")


class ControlFlowFunctionApplyFunction(Protocol[L]):
    @abstractmethod
    def __call__(
        self, abstract_environment: AbstractEnvironment[L], program_point: ProgramPoint
    ) -> AbstractEnvironment[L] | None: ...


@dataclass(frozen=True)
class ControlFlowFunction(Generic[L]):
    condition_tex: str
    modification_tex: str
    apply_function: ControlFlowFunctionApplyFunction[L]


@dataclass(init=False, frozen=True)
class ControlFlowFunctions(Generic[L]):
    functions: tuple[ControlFlowFunction[L], ...]

    def __init__(self, *functions: ControlFlowFunction[L]):
        object.__setattr__(self, "functions", functions)

    def apply(
        self, abstract_environment: AbstractEnvironment[L], program_point: ProgramPoint
    ) -> tuple[AbstractEnvironment[L], ControlFlowFunction[L]] | None:
        for function in self.functions:
            new_abstract_environment = function.apply_function(
                abstract_environment, program_point
            )
            if new_abstract_environment is not None:
                return new_abstract_environment, function

        return None


LINE_LENGTH = 9
CONDITION_PART_INDEX = 6
MODIFICATION_PART_INDEX = 2


class ControlFlowFunctionsTex(MathTex, Generic[L]):

    def __init__(self, control_flow_functions: ControlFlowFunctions[L]):
        tex_strings: list[str] = []

        for function in control_flow_functions.functions:
            tex_strings.extend(
                (
                    "&",
                    r"fg[[p]](\phi) = &" if not tex_strings else "&",
                    function.modification_tex,
                    r"& \quad",
                    "if" if function.condition_tex else "otherwise",
                    "&",
                    function.condition_tex,
                    "&",
                    r"\\",
                )
            )

        super().__init__(*tex_strings)
        self.control_flow_functions = control_flow_functions

    def get_condition_part(self, function: ControlFlowFunction[L]) -> Group:
        i = self.control_flow_functions.functions.index(function)

        return self[i * LINE_LENGTH + CONDITION_PART_INDEX]

    def get_modification_part(self, function: ControlFlowFunction[L]) -> Group:
        i = self.control_flow_functions.functions.index(function)

        return self[i * LINE_LENGTH + MODIFICATION_PART_INDEX]
