from __future__ import annotations
from typing import TypeVar, Generic, Hashable, Collection, Iterable, Callable, Generator
from manim.scene.zoomed_scene import MovingCameraScene
from manim.mobject.mobject import Mobject
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
    AbstractEnvironmentUpdateInstances,
)
from manim_dataflow_analysis.flow_function import ControlFlowFunction
from manim_dataflow_analysis.worklist import WorklistTex, WorklistTable, ResTable
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.types.vectorized_mobject import VMobject
from manim.animation.animation import Animation
from manim.mobject.geometry.line import Arrow
from manim.mobject.text.code_mobject import Code
from manim.mobject.text.text_mobject import Text
from manim.animation.creation import Create, Uncreate, Write, Unwrite
from manim.animation.transform import FadeTransform, Transform
from manim.constants import LEFT, RIGHT, DOWN
from manim.utils.color import ORANGE
from contextlib import contextmanager
from frozendict import frozendict
import networkx as nx
import numpy as np


L = TypeVar("L", bound=Hashable)
E = TypeVar("E", bound=Hashable)


class AbstractAnalysisScene(MovingCameraScene, Generic[L, E]):

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

    condition_update_function_wait_time: float = 2.5

    condition_update_function_highlight_wait_time: float = 2.5

    condition_update_function_set_wait_time: float = 2.5

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
    ) -> tuple[ProgramPoint, nx.DiGraph[ProgramPoint], ControlFlowGraph]:
        entry_point, program_cfg = self.program.to_cfg()

        cfg = ControlFlowGraph.from_cfg(entry_point, program_cfg)

        cfg.move_to(position)
        cfg.scale(scale)

        self.play(Create(cfg))

        self.wait(self.cfg_wait_time)

        return entry_point, program_cfg, cfg

    def show_program_conversion(
        self, program: Code, program_subtitle: Text
    ) -> tuple[ProgramPoint, nx.DiGraph[ProgramPoint], ControlFlowGraph]:
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

    def show_flow_function_instance(
        self,
        instance_id: int,
        control_flow_modification: VMobject,
        control_flow_modification_rectangle: SurroundingRectangle,
    ):
        instances_tex = AbstractEnvironmentUpdateInstances(
            self.control_flow_function.flow_function.instances
        )
        instances_tex.scale_to_fit_width(self.camera.frame_width * 0.8)

        instance = instances_tex.get_instance_part(instance_id)
        condition = instances_tex.get_condition_part(instance_id)
        modification = instances_tex.get_modification_part(instance_id)

        instance_rectangle = SurroundingRectangle(instance)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        self.play(Write(instances_tex))

        if condition_rectangle is not None:
            self.play(
                Create(condition_rectangle),
                Transform(control_flow_modification, instance),
                Transform(control_flow_modification_rectangle, instance_rectangle),
            )
        else:
            self.play(
                Transform(control_flow_modification, instance),
                Transform(control_flow_modification_rectangle, instance_rectangle),
            )

        self.wait(self.flow_function_wait_time)

        self.remove(control_flow_modification, control_flow_modification_rectangle)

        if condition_rectangle is not None:
            self.play(
                Transform(instance_rectangle, modification_rectangle),
                Transform(condition_rectangle, modification_rectangle),
            )
        else:
            self.play(Transform(instance_rectangle, modification_rectangle))

        self.wait(self.flow_function_set_wait_time)

        if condition_rectangle is not None:
            self.remove(instance_rectangle, condition_rectangle)
        else:
            self.remove(instance_rectangle)

        self.play(Unwrite(instances_tex), Uncreate(modification_rectangle))

    def show_control_flow_function_instance(
        self,
        instance_id: int | tuple[int, int],
    ):
        instances_tex = AbstractEnvironmentUpdateInstances(
            self.control_flow_function.instances
        )
        self.scale_mobject(
            instances_tex,
            self.camera.frame_width * 0.95,
            self.camera.frame_height * 0.95,
        )

        if isinstance(instance_id, int):
            flow_instance_id = None
        else:
            instance_id, flow_instance_id = instance_id

        instance = instances_tex.get_instance_part(instance_id)
        condition = instances_tex.get_condition_part(instance_id)
        modification = instances_tex.get_modification_part(instance_id)

        instance_rectangle = SurroundingRectangle(instance)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        self.play(Write(instances_tex))

        self.wait(self.control_flow_function_wait_time)

        if condition_rectangle is not None:
            self.play(Create(instance_rectangle), Create(condition_rectangle))
        else:
            self.play(Create(instance_rectangle))

        self.wait(self.control_flow_function_highlight_wait_time)

        if condition_rectangle is not None:
            self.play(
                Transform(instance_rectangle, modification_rectangle),
                Transform(condition_rectangle, modification_rectangle),
            )
        else:
            self.play(
                Transform(instance_rectangle, modification_rectangle),
            )

        self.wait(self.control_flow_function_set_wait_time)

        if flow_instance_id is not None:
            modification_copy = modification.copy()
            modification_rectangle_copy = modification_rectangle.copy()

            self.add(modification_copy, modification_rectangle_copy)

            self.play(Unwrite(instances_tex))

            if condition_rectangle is not None:
                self.remove(
                    instance_rectangle, condition_rectangle, modification_rectangle
                )
            else:
                self.remove(instance_rectangle, modification_rectangle)

            self.show_flow_function_instance(
                flow_instance_id,
                modification_copy,
                modification_rectangle_copy,
            )
        else:
            if condition_rectangle is not None:
                self.remove(instance_rectangle, condition_rectangle)
            else:
                self.remove(instance_rectangle)

            self.play(Unwrite(instances_tex), Uncreate(modification_rectangle))

    def show_condition_update_function_instance(self, instance_id: int):
        instances_tex = AbstractEnvironmentUpdateInstances(
            self.condition_update_function.instances
        )
        self.scale_mobject(
            instances_tex,
            self.camera.frame_width * 0.95,
            self.camera.frame_height * 0.95,
        )

        instance = instances_tex.get_instance_part(instance_id)
        condition = instances_tex.get_condition_part(instance_id)
        modification = instances_tex.get_modification_part(instance_id)

        instance_rectangle = SurroundingRectangle(instance)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        self.play(Write(instances_tex))

        self.wait(self.condition_update_function_wait_time)

        if condition_rectangle is not None:
            self.play(Create(instance_rectangle), Create(condition_rectangle))
        else:
            self.play(Create(instance_rectangle))

        self.wait(self.condition_update_function_highlight_wait_time)

        if condition_rectangle is not None:
            self.play(
                Transform(instance_rectangle, modification_rectangle),
                Transform(condition_rectangle, modification_rectangle),
            )
        else:
            self.play(
                Transform(instance_rectangle, modification_rectangle),
            )

        self.wait(self.condition_update_function_set_wait_time)

        if condition_rectangle is not None:
            self.remove(instance_rectangle, condition_rectangle)
        else:
            self.remove(instance_rectangle)

        self.play(Unwrite(instances_tex), Uncreate(modification_rectangle))

    @contextmanager
    def animate_mobject(
        self,
        previous_mobject: Mobject | None,
        new_mobject: Mobject | None,
    ) -> Generator[tuple[Animation | None, Mobject | None], None, None]:
        if previous_mobject is not None and new_mobject is not None:
            yield Transform(previous_mobject, new_mobject), new_mobject
            self.remove(previous_mobject)
            self.add(new_mobject)
        elif previous_mobject is not None and new_mobject is None:
            yield Uncreate(previous_mobject), None
        elif new_mobject is not None and previous_mobject is None:
            yield Create(new_mobject), new_mobject
        else:
            yield None, None

    def scale_mobject(self, mobject: Mobject, max_x: float, max_y: float) -> None:
        if mobject.width >= mobject.height:
            if max_y / max_x >= mobject.height / mobject.width:
                mobject.scale(max_x / mobject.width)
            else:
                mobject.scale(max_y / mobject.height)
        else:
            if max_x / max_y >= mobject.width / mobject.height:
                mobject.scale(max_y / mobject.height)
            else:
                mobject.scale(max_x / mobject.width)

    def create_worklist_table(
        self,
        variables: Collection[str],
        abstract_environments: dict[ProgramPoint, AbstractEnvironment[L]],
    ) -> WorklistTable[L]:
        table = WorklistTable(variables, abstract_environments)

        self.scale_mobject(
            table,
            self.camera.frame_width * 0.45,
            self.camera.frame_height * 0.55,
        )

        table.move_to(
            (self.camera.frame_width * 0.25, self.camera.frame_height * -0.2, 0)
        )

        return table

    def create_res_table(
        self,
        variables: Collection[str],
        res: AbstractEnvironment[L] | None = None,
        res_cond: AbstractEnvironment[L] | None = None,
    ) -> ResTable[L]:
        res_table = ResTable(variables, res, res_cond)

        self.scale_mobject(
            res_table,
            self.camera.frame_width * 0.45,
            self.camera.frame_height * 0.25,
        )

        res_table.move_to(
            (self.camera.frame_width * 0.25, self.camera.frame_height * 0.225, 0)
        )

        return res_table

    def create_worklist_tex(self, worklist: set[ProgramPoint]) -> WorklistTex:
        worklist_tex = WorklistTex(worklist)

        self.scale_mobject(
            worklist_tex,
            self.camera.frame_width * 0.45,
            self.camera.frame_height * 0.10,
        )

        worklist_tex.move_to(
            (self.camera.frame_width * 0.25, self.camera.frame_height * 0.425, 0)
        )

        return worklist_tex

    def worklist(
        self,
        entry_point: ProgramPoint,
        program_cfg: nx.DiGraph[ProgramPoint],
        cfg: ControlFlowGraph,
        variables: Collection[str],
    ):
        abstract_environments = {
            p: AbstractEnvironment(
                self.lattice,
                frozendict((variable, self.lattice.bottom()) for variable in variables),
            )
            for p in program_cfg.nodes
        }

        worklist = {entry_point}

        table = self.create_worklist_table(variables, abstract_environments)

        self.scale_mobject(
            cfg,
            self.camera.frame_width * 0.45,
            self.camera.frame_height,
        )

        cfg.move_to((self.camera.frame_width * -0.25, 0, 0))

        res_table = self.create_res_table(variables)

        worklist_tex = self.create_worklist_tex(worklist)

        self.play(Create(table), Create(cfg), Create(res_table), Create(worklist_tex))

        while worklist:
            program_point = worklist.pop()

            worklist_program_point_rectangle = SurroundingRectangle(
                worklist_tex.get_program_point_part(program_point)
            )

            self.play(Create(worklist_program_point_rectangle))

            table_program_point_rectangle = None
            program_point_rectangle = None

            with (
                self.animate_mobject(
                    worklist_tex,
                    self.create_worklist_tex(worklist),
                ) as (
                    worklist_animation,
                    worklist_tex,
                ),
                self.animate_mobject(
                    program_point_rectangle,
                    SurroundingRectangle(cfg[program_point]),
                ) as (
                    program_point_animation,
                    program_point_rectangle,
                ),
                self.animate_mobject(
                    table_program_point_rectangle,
                    SurroundingRectangle(table.get_program_point_part(program_point)),
                ) as (
                    table_program_point_animation,
                    table_program_point_rectangle,
                ),
            ):
                self.play(
                    Uncreate(worklist_program_point_rectangle),
                    worklist_animation,
                    program_point_animation,
                    table_program_point_animation,
                )

            res, res_instance_id = self.control_flow_function.apply(
                program_point, abstract_environments[program_point]
            )

            self.camera.frame.save_state()

            self.play(
                self.camera.frame.animate.move_to((0, -self.camera.frame_height, 0))
            )

            self.show_control_flow_function_instance(res_instance_id)

            self.play(self.camera.frame.animate.restore())

            with (
                self.animate_mobject(
                    res_table,
                    self.create_res_table(variables, res),
                ) as (res_table_animation, res_table),
            ):
                self.play(res_table_animation)

            table_successor_program_point_rectangle = None
            successor_program_point_rectangle = None

            for successor in succ(program_cfg, program_point):
                res_cond, res_cond_instance_id = self.condition_update_function.apply(
                    cond(program_cfg, program_point, successor),
                    res,
                )

                with (
                    self.animate_mobject(
                        successor_program_point_rectangle,
                        SurroundingRectangle(cfg[successor], color=ORANGE),
                    ) as (
                        successor_program_point_animation,
                        successor_program_point_rectangle,
                    ),
                    self.animate_mobject(
                        table_successor_program_point_rectangle,
                        SurroundingRectangle(
                            table.get_program_point_part(successor), color=ORANGE
                        ),
                    ) as (
                        table_successor_program_point_animation,
                        table_successor_program_point_rectangle,
                    ),
                ):
                    self.play(
                        successor_program_point_animation,
                        table_successor_program_point_animation,
                    )

                self.camera.frame.save_state()

                self.play(
                    self.camera.frame.animate.move_to((0, self.camera.frame_height, 0))
                )

                self.show_condition_update_function_instance(res_cond_instance_id)

                self.play(self.camera.frame.animate.restore())

                with (
                    self.animate_mobject(
                        res_table,
                        self.create_res_table(variables, res, res_cond),
                    ) as (
                        res_table_animation,
                        res_table,
                    ),
                ):
                    self.play(res_table_animation)

                if not abstract_environments[successor].includes(res_cond):
                    abstract_environments[successor] = abstract_environments[
                        successor
                    ].join(res_cond)

                    worklist.add(successor)

                    with (
                        self.animate_mobject(
                            worklist_tex,
                            self.create_worklist_tex(worklist),
                        ) as (
                            worklist_animation,
                            worklist_tex,
                        ),
                        self.animate_mobject(
                            table,
                            self.create_worklist_table(
                                variables, abstract_environments
                            ),
                        ) as (
                            table_animation,
                            table,
                        ),
                    ):
                        self.play(worklist_animation, table_animation)

                with (
                    self.animate_mobject(
                        res_table,
                        self.create_res_table(variables, res),
                    ) as (
                        res_table_animation,
                        res_table,
                    ),
                ):
                    self.play(res_table_animation)

            with (
                self.animate_mobject(
                    res_table,
                    self.create_res_table(variables),
                ) as (
                    res_table_animation,
                    res_table,
                ),
                self.animate_mobject(
                    program_point_rectangle,
                    None,
                ) as (
                    program_point_animation,
                    program_point_rectangle,
                ),
                self.animate_mobject(
                    table_program_point_rectangle,
                    None,
                ) as (
                    table_program_point_animation,
                    table_program_point_rectangle,
                ),
                self.animate_mobject(
                    successor_program_point_rectangle,
                    None,
                ) as (
                    successor_program_point_animation,
                    successor_program_point_rectangle,
                ),
                self.animate_mobject(
                    table_successor_program_point_rectangle,
                    None,
                ) as (
                    table_successor_program_point_animation,
                    table_successor_program_point_rectangle,
                ),
            ):
                if (
                    successor_program_point_animation is not None
                    and table_successor_program_point_animation is not None
                ):
                    self.play(
                        successor_program_point_animation,
                        table_successor_program_point_animation,
                    )

                self.play(
                    res_table_animation,
                    program_point_animation,
                    table_program_point_animation,
                )

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
            cfg,
            self.program.variables,
        )

        self.wait(5)
