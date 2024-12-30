from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import networkx as nx

    from manim_dataflow_analysis.cfg import ProgramPoint


class AstStatement(Protocol):
    """A statement in its AST form."""

    @property
    @abstractmethod
    def header(self) -> str:
        """The header of the AST statement.

        Returns:
            The header of the AST statement.
        """


class AstFunction(Protocol):
    """A function in its AST form."""

    @property
    def line_number(self) -> int:
        """The line number of the AST function.

        Returns:
            The line number of the AST function.
        """
        return 1

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the AST function.

        Returns:
            The name of the AST function.
        """

    @property
    @abstractmethod
    def parameters(self) -> tuple[str, ...]:
        """The parameters of the AST function.

        Returns:
            The parameters of the AST function.
        """

    @property
    @abstractmethod
    def variables(self) -> frozenset[str]:
        """Get the variable names used in the AST program.

        Returns:
            The set of variable names used in the AST program.
        """

    @abstractmethod
    def to_cfg(self) -> tuple[ProgramPoint, nx.DiGraph[ProgramPoint]]:
        """Convert the AST program to its control flow graph form.

        Returns:
            A tuple containing the entry point and the CFG of the program.
        """

    @abstractmethod
    def __str__(self) -> str:
        """Convert the AST program to its string form.

        Returns:
            The string representation of the AST program.
        """
