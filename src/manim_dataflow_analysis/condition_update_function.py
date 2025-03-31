from abc import abstractmethod
from typing import Mapping, Protocol, Sequence, TypeVar

from manim_dataflow_analysis.abstract_environment import AbstractEnvironment

L = TypeVar("L")
E_contra = TypeVar("E_contra", contravariant=True)


class ConditionUpdateFunction(Protocol[L, E_contra]):
    @property
    @abstractmethod
    def instances(self) -> Sequence[tuple[str, str, str | None]]: ...

    @abstractmethod
    def get_variables(
        self,
        expression: E_contra,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[Mapping[str, L] | None, int]: ...

    def apply(
        self,
        expression: E_contra,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L] | None, int]:
        variables, instance_id = self.get_variables(expression, abstract_environment)

        if variables is None:
            return None, instance_id
        else:
            return abstract_environment.set(variables), instance_id

    def apply_and_get_variables(
        self,
        expression: E_contra,
        abstract_environment: AbstractEnvironment[L],
    ) -> tuple[AbstractEnvironment[L] | None, Mapping[str, L] | None, int]:
        variables, instance_id = self.get_variables(expression, abstract_environment)

        if variables is None:
            return None, None, instance_id
        else:
            return abstract_environment.set(variables), variables, instance_id
