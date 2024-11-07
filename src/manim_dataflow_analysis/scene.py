from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Hashable
from manim.scene.scene import Scene
from manim_dataflow_analysis.ast import AstProgram
from manim_dataflow_analysis.cfg import ControlFlowGraph
from manim.mobject.text.code_mobject import Code
from manim.animation.creation import Create

E = TypeVar("E", bound=Hashable)


class AbstractAnalysisScene(ABC, Scene, Generic[E]):

    wait_time: int = 10

    @property
    @abstractmethod
    def program(self) -> AstProgram:
        """The program to analyze."""

    @abstractmethod
    def condition_update(self, condition: E):
        """Update the condition."""

    def show_code(self):
        self.play(
            Create(
                Code(
                    code=str(self.program),
                    language="c",
                    background="window",
                    tab_width=4,
                    font="Monospace",
                    style="monokai",
                )
            )
        )

    def show_cfg(self):
        entry_point, program_cfg = self.program.to_cfg()

        cfg = ControlFlowGraph.from_cfg(entry_point, program_cfg)

        cfg.move_to((0, 0, 0))
        cfg.scale(0.5)

        self.play(Create(cfg))

        return entry_point, cfg

    def construct(self):
        self.show_code()

        self.wait(self.wait_time)

        self.clear()

        entry_point, cfg = self.show_cfg()

        self.wait(self.wait_time)
