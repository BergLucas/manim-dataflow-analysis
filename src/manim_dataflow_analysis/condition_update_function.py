from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from typing import TypeVar, Protocol, Sequence, Mapping
from abc import abstractmethod

L = TypeVar("L")
E = TypeVar("E")


class ConditionUpdateFunction(Protocol[L, E]):
    @property
    @abstractmethod
    def instances(self) -> Sequence[tuple[str, str, str | None]]: ...

    @abstractmethod
    def get_variables(
        self,
        expression: E,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[Mapping[str, L], int]: ...

    def apply(
        self,
        expression: E,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], int]:
        variables, instance_id = self.get_variables(expression, abstract_environment)
        return abstract_environment.set(**variables), instance_id

    def apply_and_get_variables(
        self,
        expression: E,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], Mapping[str, L], int]:
        variables, instance_id = self.get_variables(expression, abstract_environment)
        return abstract_environment.set(**variables), variables, instance_id
