from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Hashable
from manim.scene.scene import Scene
from manim_dataflow_analysis.ast import AstProgram
from manim_dataflow_analysis.cfg import ControlFlowGraph, ProgramPoint
from manim_dataflow_analysis.lattice import LatticeGraph, Lattice
from manim.mobject.geometry.line import Arrow
from manim.mobject.text.code_mobject import Code
from manim.mobject.text.text_mobject import Text
from manim.animation.creation import Create, Uncreate, Write, Unwrite
from manim.constants import LEFT, RIGHT, DOWN
import networkx as nx
import numpy as np


L = TypeVar("L", bound=Hashable)
E = TypeVar("E", bound=Hashable)


class AbstractAnalysisScene(ABC, Scene, Generic[L, E]):

    title_wait_time: int = 2.5

    program_wait_time: int = 2.5

    program_conversion_wait_time: int = 2.5

    cfg_wait_time: int = 5

    lattice_wait_time: int = 5

    sorting_function = lambda it: list(it)

    @property
    def title(self) -> str:
        """The title of the scene."""
        return "Dataflow Analysis"

    @property
    def program_subtitle(self) -> str:
        """The subtitle of the program."""
        return "Here is the program that we are going to analyse."

    @property
    def program_conversion_subtitle(self) -> str:
        """The subtitle of the program conversion."""
        return "First, we need to convert it into a control flow graph."

    @property
    @abstractmethod
    def program(self) -> AstProgram:
        """The program to analyse."""

    @property
    @abstractmethod
    def lattice(self) -> Lattice[L]:
        """The lattice of the analysis."""

    @abstractmethod
    def condition_update(self, condition: E):
        """Update the condition."""

    def show_title(self) -> None:
        title = Text(self.title)

        self.play(Write(title))

        self.wait(self.title_wait_time)

        self.play(Unwrite(title))

    def show_program(self, scale: float = 0.75) -> tuple[Code, Text]:
        program = Code(
            code=str(self.program),
            language="c",
            background="window",
            tab_width=4,
            font="Monospace",
            style="monokai",
        )
        program.scale(scale)

        program_subtitle = Text(self.program_subtitle)
        program_subtitle.scale_to_fit_width(self.camera.frame_width * 0.8)

        program_subtitle.move_to(
            self.camera.frame_height * np.array([0, 0.5, 0]) + DOWN
        )

        self.play(Write(program_subtitle))

        self.play(Create(program))

        self.wait(self.program_wait_time)

        return program, program_subtitle

    def show_cfg(
        self, position: tuple[int, int, int] = (0, 0, 0), scale: float = 0.5
    ) -> tuple[ProgramPoint, ControlFlowGraph, nx.DiGraph[ProgramPoint]]:
        entry_point, program_cfg = self.program.to_cfg()

        cfg = ControlFlowGraph.from_cfg(entry_point, program_cfg)

        cfg.move_to(position)
        cfg.scale(scale)

        self.play(Create(cfg))

        self.wait(self.cfg_wait_time)

        return entry_point, program_cfg, cfg

    def show_program_conversion(
        self, program: Code, program_subtitle: Text
    ) -> tuple[ProgramPoint, ControlFlowGraph, nx.DiGraph[ProgramPoint]]:
        self.play(Unwrite(program_subtitle))

        program_conversion_subtitle = Text(self.program_conversion_subtitle)
        program_conversion_subtitle.scale_to_fit_width(self.camera.frame_width * 0.8)
        program_conversion_subtitle.move_to(program_subtitle.get_center())

        self.play(Write(program_conversion_subtitle))

        self.wait(self.program_conversion_wait_time)

        self.play(
            program.animate.move_to(self.camera.frame_width * np.array([-0.25, 0, 0]))
        )

        arrow = Arrow(start=LEFT, end=RIGHT)

        self.play(Create(arrow))

        entry_point, program_cfg, cfg = self.show_cfg(
            position=program.get_right() + RIGHT * 5, scale=0.33
        )

        self.play(Uncreate(arrow))

        self.remove(program_cfg)

        return entry_point, program_cfg, cfg

    def show_lattice(
        self,
        position: tuple[int, int, int] = (0, 0, 0),
        scale: float = 0.25,
        max_horizontal_size: int = 8,
        max_vertical_size: int = 8,
    ):
        lattice = LatticeGraph(
            self.lattice,
            max_horizontal_size=max_horizontal_size,
            max_vertical_size=max_vertical_size,
            layout_config=dict(sorting_function=self.sorting_function),
        )

        lattice.move_to(position)
        lattice.scale(scale)

        self.play(Create(lattice))

        self.wait(self.lattice_wait_time)

    def construct(self):
        self.show_title()

        program, program_subtitle = self.show_program()

        entry_point, program_cfg, cfg = self.show_program_conversion(
            program, program_subtitle
        )

        self.clear()

        self.show_lattice()
