from __future__ import annotations
from typing import TypeVar, Generic, Hashable, Collection, Iterable, Callable
from manim.scene.scene import Scene
from manim_dataflow_analysis.ast import AstProgram
from manim_dataflow_analysis.cfg import ControlFlowGraph, ProgramPoint, succ, cond
from manim_dataflow_analysis.condition_update_function import ConditionUpdateFunction
from manim_dataflow_analysis.lattice import (
    LatticeGraph,
    Lattice,
    default_sorting_function,
)
from manim_dataflow_analysis.abstract_environment import (
    AbstractEnvironment,
    AbstractEnvironmentUpdateRules,
)
from manim_dataflow_analysis.flow_function import ControlFlowFunction
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.types.vectorized_mobject import VMobject
from manim.mobject.geometry.line import Arrow
from manim.mobject.text.code_mobject import Code
from manim.mobject.text.text_mobject import Text
from manim.animation.creation import Create, Uncreate, Write, Unwrite
from manim.animation.transform import FadeTransform, Transform
from manim.constants import LEFT, RIGHT, DOWN
from frozendict import frozendict
import networkx as nx
import numpy as np


L = TypeVar("L", bound=Hashable)
E = TypeVar("E", bound=Hashable)


class AbstractAnalysisScene(Scene, Generic[L, E]):

    title_wait_time: float = 2.5

    program_wait_time: float = 2.5

    program_conversion_wait_time: float = 2.5

    cfg_wait_time: float = 5.0

    lattice_wait_time: float = 5.0

    lattice_transform_wait_time: float = 2.5

    lattice_join_wait_time: float = 5.0

    flow_function_wait_time: float = 2.5

    flow_function_set_wait_time: float = 2.5

    control_flow_function_wait_time: float = 2.5

    control_flow_function_highlight_wait_time: float = 2.5

    control_flow_function_set_wait_time: float = 2.5

    sorting_function: Callable[[Iterable[Hashable]], list[Hashable]] = (
        default_sorting_function
    )

    title: str = "Dataflow Analysis"

    program_subtitle: str = "Here is the program that we are going to analyse."

    program_conversion_subtitle: str = (
        "First, we need to convert it into a control flow graph."
    )

    program: AstProgram

    lattice: Lattice[L]

    control_flow_function: ControlFlowFunction[L]

    condition_update_function: ConditionUpdateFunction[L, E]

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

    def create_lattice_graph(
        self,
        position: tuple[int, int, int] = (0, 0, 0),
        scale: float = 0.25,
        visible_vertices: set[L] | None = None,
        max_horizontal_size_per_vertex: int = 8,
        max_vertical_size: int = 8,
    ) -> LatticeGraph[L]:
        lattice_graph = LatticeGraph.from_lattice(
            self.lattice,
            visible_vertices=visible_vertices,
            max_horizontal_size_per_vertex=max_horizontal_size_per_vertex,
            max_vertical_size=max_vertical_size,
            layout_config=dict(sorting_function=type(self).sorting_function),
        )

        lattice_graph.move_to(position)
        lattice_graph.scale(scale)

        return lattice_graph

    def show_lattice_graph(
        self,
        position: tuple[int, int, int] = (0, 0, 0),
        scale: float = 0.25,
        visible_vertices: set[L] | None = None,
        max_horizontal_size_per_vertex: int = 8,
        max_vertical_size: int = 8,
    ) -> LatticeGraph[L]:
        lattice_graph = self.create_lattice_graph(
            position=position,
            scale=scale,
            visible_vertices=visible_vertices,
            max_horizontal_size_per_vertex=max_horizontal_size_per_vertex,
            max_vertical_size=max_vertical_size,
        )

        self.play(Create(lattice_graph))

        self.wait(self.lattice_wait_time)

        return lattice_graph

    def show_lattice_join(
        self,
        lattice_graph: LatticeGraph[L],
        value1: L,
        value2: L,
        position: tuple[int, int, int] = (0, 0, 0),
        scale: float = 0.25,
        max_horizontal_size_per_vertex: int = 8,
        max_vertical_size: int = 8,
    ) -> LatticeGraph[L]:
        joined_value = self.lattice.join(value1, value2)

        visible_vertices = {
            value1,
            value2,
            joined_value,
        }

        new_lattice_graph = self.create_lattice_graph(
            position=position,
            scale=scale,
            visible_vertices=visible_vertices,
            max_horizontal_size_per_vertex=max_horizontal_size_per_vertex,
            max_vertical_size=max_vertical_size,
        )

        self.play(FadeTransform(lattice_graph, new_lattice_graph))

        self.wait(self.lattice_transform_wait_time)

        new_lattice_graph.color_path(value1, joined_value)
        new_lattice_graph.color_path(value2, joined_value)

        self.wait(self.lattice_join_wait_time)

        return new_lattice_graph

    def show_flow_functions_instance(
        self,
        instance_id: int,
        control_flow_modification: VMobject,
        control_flow_modification_rectangle: SurroundingRectangle,
    ):
        rules = AbstractEnvironmentUpdateRules(
            self.control_flow_function.flow_function.instances
        )
        rules.scale_to_fit_width(self.camera.frame_width * 0.8)

        rule = rules.get_rule_part(instance_id)
        condition = rules.get_condition_part(instance_id)
        modification = rules.get_modification_part(instance_id)

        rule_rectangle = SurroundingRectangle(rule)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        self.play(Write(rules))

        if condition_rectangle is not None:
            self.play(
                Create(condition_rectangle),
                Transform(control_flow_modification, rule),
                Transform(control_flow_modification_rectangle, rule_rectangle),
            )
        else:
            self.play(
                Transform(control_flow_modification, rule),
                Transform(control_flow_modification_rectangle, rule_rectangle),
            )

        self.wait(self.flow_function_wait_time)

        self.remove(control_flow_modification, control_flow_modification_rectangle)

        if condition_rectangle is not None:
            self.play(
                Transform(rule_rectangle, modification_rectangle),
                Transform(condition_rectangle, modification_rectangle),
            )
        else:
            self.play(Transform(rule_rectangle, modification_rectangle))

        self.wait(self.flow_function_set_wait_time)

        if condition_rectangle is not None:
            self.remove(rule_rectangle, condition_rectangle)
        else:
            self.remove(rule_rectangle)

        self.play(Unwrite(rules), Uncreate(modification_rectangle))

    def show_control_flow_functions_instance(
        self,
        instance_id: int | tuple[int, int],
    ):
        rules = AbstractEnvironmentUpdateRules(self.control_flow_function.instances)
        rules.scale_to_fit_width(self.camera.frame_width * 0.8)

        if isinstance(instance_id, int):
            flow_instance_id = None
        else:
            instance_id, flow_instance_id = instance_id

        rule = rules.get_rule_part(instance_id)
        condition = rules.get_condition_part(instance_id)
        modification = rules.get_modification_part(instance_id)

        rule_rectangle = SurroundingRectangle(rule)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        self.play(Write(rules))

        self.wait(self.control_flow_function_wait_time)

        if condition_rectangle is not None:
            self.play(Create(rule_rectangle), Create(condition_rectangle))
        else:
            self.play(Create(rule_rectangle))

        self.wait(self.control_flow_function_highlight_wait_time)

        if condition_rectangle is not None:
            self.play(
                Transform(rule_rectangle, modification_rectangle),
                Transform(condition_rectangle, modification_rectangle),
            )
        else:
            self.play(
                Transform(rule_rectangle, modification_rectangle),
            )

        self.wait(self.control_flow_function_set_wait_time)

        if flow_instance_id is not None:
            modification_copy = modification.copy()
            modification_rectangle_copy = modification_rectangle.copy()

            self.add(modification_copy, modification_rectangle_copy)

            self.play(Unwrite(rules))

            if condition_rectangle is not None:
                self.remove(rule_rectangle, condition_rectangle, modification_rectangle)
            else:
                self.remove(rule_rectangle, modification_rectangle)

            self.show_flow_functions_instance(
                flow_instance_id,
                modification_copy,
                modification_rectangle_copy,
            )
        else:
            if condition_rectangle is not None:
                self.remove(rule_rectangle, condition_rectangle)
            else:
                self.remove(rule_rectangle)

            self.play(Unwrite(rules), Uncreate(modification_rectangle))

    def worklist(
        self,
        entry_point: ProgramPoint,
        cfg: nx.DiGraph[ProgramPoint],
        variables: Collection[str],
        lattice_graph: LatticeGraph[L],
    ):
        abstract_environments = {
            p: AbstractEnvironment(
                self.lattice,
                frozendict((variable, self.lattice.bottom()) for variable in variables),
            )
            for p in cfg.nodes
        }

        worklist = {entry_point}
        while worklist:
            program_point = worklist.pop()

            res, control_flow_instance_id = self.control_flow_function.apply(
                program_point, abstract_environments[program_point]
            )

            self.show_control_flow_functions_instance(control_flow_instance_id)

            for successor in succ(cfg, program_point):
                resCond, condition_update_instance_id = (
                    self.condition_update_function.apply(
                        cond(cfg, program_point, successor),
                        res,
                    )
                )

                if not abstract_environments[successor].includes(resCond):
                    abstract_environments[successor] = abstract_environments[
                        successor
                    ].join(resCond)

                    worklist.add(successor)

    def construct(self):
        self.show_title()

        program, program_subtitle = self.show_program()

        entry_point, program_cfg, cfg = self.show_program_conversion(
            program, program_subtitle
        )

        self.clear()

        lattice_graph = self.show_lattice_graph()

        self.clear()

        self.worklist(
            entry_point,
            program_cfg,
            self.program.variables,
            lattice_graph,
        )
