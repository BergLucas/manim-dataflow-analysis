from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from typing import TypeVar, Protocol, Sequence
from abc import abstractmethod

L = TypeVar("L")
E = TypeVar("E")


class ConditionUpdateFunction(Protocol[L, E]):
    @property
    @abstractmethod
    def instances(self) -> Sequence[tuple[str, str, str | None]]: ...

    @abstractmethod
    def apply(
        self,
        expression: E,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L], int]: ...
