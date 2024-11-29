from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from abc import abstractmethod
import networkx as nx

if TYPE_CHECKING:
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

    @abstractmethod
    def __str__(self) -> str:
        """Convert the AST statement to its string form.

        Returns:
            The string representation of the AST statement.
        """


class AstProgram(Protocol):
    """A program in its AST form."""

    @property
    @abstractmethod
    def language(self) -> str:
        """The language of the AST program.

        Returns:
            The language of the AST program.
        """

    @property
    @abstractmethod
    def variables(self) -> set[str]:
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
