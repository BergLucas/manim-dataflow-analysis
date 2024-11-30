from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint
from manim_dataflow_analysis.ast import AstStatement
from typing import TypeVar, Protocol, Sequence
from abc import abstractmethod

L = TypeVar("L")


class FlowFunctions(Protocol[L]):
    @property
    @abstractmethod
    def rules(self) -> Sequence[tuple[str, str, str | None]]: ...

    @abstractmethod
    def apply(
        self,
        statement: AstStatement,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], int]: ...


class ControlFlowFunctions(Protocol[L]):
    @property
    @abstractmethod
    def rules(self) -> Sequence[tuple[str, str, str | None]]: ...

    @property
    @abstractmethod
    def flow_functions(self) -> FlowFunctions[L] | None: ...

    @abstractmethod
    def apply(
        self,
        program_point: ProgramPoint,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], int | tuple[int, int]]: ...
