from abc import abstractmethod
from typing import Mapping, Protocol, Sequence, TypeVar

from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.ast import AstStatement
from manim_dataflow_analysis.cfg import ProgramPoint

L = TypeVar("L")


class FlowFunction(Protocol[L]):
    @property
    @abstractmethod
    def instances(self) -> Sequence[tuple[str, str, str | None]]:
        ...

    @abstractmethod
    def get_variables(
        self,
        statement: AstStatement,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[Mapping[str, L], int]:
        ...

    def apply(
        self,
        statement: AstStatement,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], int]:
        variables, instance_id = self.get_variables(statement, abstract_environment)
        return abstract_environment.set(**variables), instance_id

    def apply_and_get_variables(
        self,
        statement: AstStatement,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], Mapping[str, L], int]:
        variables, instance_id = self.get_variables(statement, abstract_environment)
        return abstract_environment.set(**variables), variables, instance_id


class ControlFlowFunction(Protocol[L]):
    @property
    @abstractmethod
    def instances(self) -> Sequence[tuple[str, str, str | None]]:
        ...

    @property
    @abstractmethod
    def flow_function(self) -> FlowFunction[L] | None:
        ...

    @abstractmethod
    def get_variables(
        self,
        program_point: ProgramPoint,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[Mapping[str, L], int | tuple[int, int]]:
        ...

    def apply(
        self,
        program_point: ProgramPoint,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], int | tuple[int, int]]:
        variables, instance_id = self.get_variables(program_point, abstract_environment)
        return abstract_environment.set(**variables), instance_id

    def apply_and_get_variables(
        self,
        program_point: ProgramPoint,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], Mapping[str, L], int | tuple[int, int]]:
        variables, instance_id = self.get_variables(program_point, abstract_environment)
        return abstract_environment.set(**variables), variables, instance_id
