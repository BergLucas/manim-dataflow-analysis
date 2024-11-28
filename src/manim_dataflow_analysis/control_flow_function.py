from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint
from manim.mobject.text.tex_mobject import MathTex
from typing import TypeVar, Protocol
from abc import abstractmethod

L = TypeVar("L")


class ControlFlowFunction(Protocol[L]):
    @abstractmethod
    def condition_tex(self) -> MathTex: ...

    @abstractmethod
    def modification_tex(self) -> MathTex: ...

    @abstractmethod
    def apply(
        self,
        abstract_environment: AbstractEnvironment,
        program_point: ProgramPoint,
    ) -> AbstractEnvironment | None: ...
