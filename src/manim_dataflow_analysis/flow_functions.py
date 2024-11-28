from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint
from manim_dataflow_analysis.ast import AstStatement
from typing import TypeVar, Protocol, Sequence
from abc import abstractmethod

L = TypeVar("L")


class FlowFunctions(Protocol[L]):
    @abstractmethod
    def rules(self) -> Sequence[tuple[str, str, str]]: ...

    @abstractmethod
    def apply(
        self,
        abstract_environment: AbstractEnvironment[L],
        statement: AstStatement,
    ) -> tuple[AbstractEnvironment[L], int]: ...


class ControlFlowFunctions(Protocol[L]):
    @abstractmethod
    def rules(self) -> Sequence[tuple[str, str, str]]: ...

    @abstractmethod
    def flow_functions() -> FlowFunctions[L] | None: ...

    @abstractmethod
    def apply(
        self,
        abstract_environment: AbstractEnvironment[L],
        program_point: ProgramPoint,
    ) -> tuple[AbstractEnvironment[L], int | tuple[int, int]]: ...
