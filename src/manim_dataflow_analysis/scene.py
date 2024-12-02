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
from manim.utils.color import ORANGE
from contextlib import contextmanager
from frozendict import frozendict
from manim import config
import networkx as nx


L = TypeVar("L", bound=Hashable)
E = TypeVar("E", bound=Hashable)


def fw(scale_w: float):
    return scale_w * config.frame_width


def fh(scale_y: float):
    return scale_y * config.frame_height


class AbstractAnalysisScene(MovingCameraScene, Generic[L, E]):

    # Title
    title: str = "Dataflow Analysis"
    title_width: float = fw(0.5)
    title_height: float = fh(0.5)
    title_position: tuple[float, float, float] = (0, 0, 0)
    title_camera_position: tuple[float, float, float] = (0, 0, 0)
    title_wait_time: float = 2.5

    # Lattice
    lattice: Lattice[L]
    lattice_title: str = "Here is the lattice that we are going to use."
    lattice_title_width: float = fw(0.95)
    lattice_title_height: float = fh(0.175)
    lattice_title_position: tuple[float, float, float] = (fw(1), fh(0.3875), 0)
    lattice_width: float = fw(0.95)
    lattice_height: float = fh(0.775)
    lattice_position: tuple[float, float, float] = (fw(1), fh(-0.0775), 0)
    lattice_camera_position: tuple[float, float, float] = (fw(1), 0, 0)
    lattice_wait_time: float = 5.0
    lattice_max_horizontal_size_per_vertex: int = 8
    lattice_max_vertical_size: int = 8
    lattice_transform_wait_time: float = 2.5
    lattice_join_wait_time: float = 5.0
    sorting_function: Callable[[Iterable[Hashable]], list[Hashable]] = (
        default_sorting_function
    )

    # Control-flow function
    control_flow_function: ControlFlowFunction[L]
    control_flow_function_title: str = (
        "We will use the following control-flow function."
    )
    control_flow_function_title_width: float = fw(0.95)
    control_flow_function_title_height: float = fw(0.2)
    control_flow_function_title_position: tuple[float, float, float] = (0, fh(-0.6), 0)
    control_flow_function_width: float = fw(0.95)
    control_flow_function_height: float = fh(0.8)
    control_flow_function_position: tuple[float, float, float] = (0, fh(-1.1), 0)
    control_flow_function_camera_position: tuple[float, float, float] = (0, fh(-1), 0)
    control_flow_function_wait_time: float = 2.5
    control_flow_function_highlight_wait_time: float = 2.5
    control_flow_function_set_wait_time: float = 2.5

    # Flow function
    flow_function_title: str = "And the following flow function"
    flow_function_title_width: float = fw(0.95)
    flow_function_title_height: float = fh(0.2)
    flow_function_title_position: tuple[float, float, float] = (0, fh(-1.6), 0)
    condition_update_function_width: float = fw(0.95)
    condition_update_function_height: float = fh(0.8)
    flow_function_position: tuple[float, float, float] = (0, fh(-2.1), 0)
    flow_function_camera_position: tuple[float, float, float] = (0, fh(-2), 0)
    flow_function_wait_time: float = 2.5
    flow_function_set_wait_time: float = 2.5

    # Condition update function
    condition_update_function: ConditionUpdateFunction[L, E]
    condition_update_function_title: str = (
        "We will also use this condition update function"
    )
    condition_update_function_title_width: float = fw(0.95)
    condition_update_function_title_height: float = fh(0.2)
    condition_update_function_title_position: tuple[float, float, float] = (
        0,
        fh(1.4),
        0,
    )
    condition_update_function_width: float = fw(0.95)
    condition_update_function_height: float = fh(0.8)
    condition_update_function_position: tuple[float, float, float] = (0, fh(0.9), 0)
    condition_update_function_camera_position: tuple[float, float, float] = (
        0,
        fh(1),
        0,
    )
    condition_update_function_wait_time: float = 2.5
    condition_update_function_highlight_wait_time: float = 2.5
    condition_update_function_set_wait_time: float = 2.5

    # Program
    program: AstProgram
    program_title: str = "Here is the program that we are going to analyse."
    program_title_width: float = fw(0.95)
    program_title_height: float = fh(0.2)
    program_title_position: tuple[float, float, float] = (fw(-1), fh(0.4), 0)
    program_width: float = fw(0.5)
    program_height: float = fh(0.8)
    program_position: tuple[float, float, float] = (fw(-1), fh(-0.1), 0)
    program_camera_position: tuple[float, float, float] = (fw(-1), 0, 0)
    program_wait_time: float = 2.5

    # Program conversion
    program_conversion_title: str = (
        "First, we need to convert it into a control flow graph."
    )
    program_conversion_title_width: float = fw(0.95)
    program_conversion_title_height: float = fh(0.2)
    program_conversion_title_position: tuple[float, float, float] = (
        fw(-0.45),
        fh(0.4),
        0,
    )
    program_new_width: float = fw(0.25)
    program_new_height: float = fh(0.8)
    program_new_position: tuple[float, float, float] = (fw(-0.75), fh(-0.1), 0)
    program_arrow_start_position: tuple[float, float, float] = (fw(-0.6), fh(-0.1), 0)
    program_arrow_end_position: tuple[float, float, float] = (fw(-0.5), fh(-0.1), 0)
    program_conversion_camera_position: tuple[float, float, float] = (fw(-0.45), 0, 0)
    program_conversion_wait_time: float = 2.5

    # CFG
    cfg_width: float = fw(0.45)
    cfg_height: float = fh(0.8)
    cfg_position: tuple[float, float, float] = (fw(-0.25), fh(-0.1), 0)
    cfg_wait_time: float = 5.0

    # Worklist
    worklist_camera_position: tuple[float, float, float] = (0, 0, 0)

    def show_title(self) -> None:
        title = Text(self.title)

        self.scale_mobject(title, self.title_width, self.title_height)
        title.move_to(self.title_position)

        self.play(Write(title))

        self.wait(self.title_wait_time)

        self.play(Unwrite(title))

    def create_lattice_graph(
        self,
        visible_vertices: set[L] | None = None,
    ) -> LatticeGraph[L]:
        lattice_graph = LatticeGraph.from_lattice(
            self.lattice,
            visible_vertices=visible_vertices,
            max_horizontal_size_per_vertex=self.lattice_max_horizontal_size_per_vertex,
            max_vertical_size=self.lattice_max_vertical_size,
            layout_config=dict(sorting_function=type(self).sorting_function),
        )

        self.scale_mobject(lattice_graph, self.lattice_width, self.lattice_height)
        lattice_graph.move_to(self.lattice_position)

        return lattice_graph

    def show_lattice_graph(self) -> LatticeGraph[L]:
        lattice_title = Text(self.lattice_title)

        self.scale_mobject(
            lattice_title, self.lattice_title_width, self.lattice_title_height
        )
        lattice_title.move_to(self.lattice_title_position)

        lattice_graph = self.create_lattice_graph()

        self.play(Write(lattice_title), Create(lattice_graph))

        self.wait(self.lattice_wait_time)

        self.play(Unwrite(lattice_title))

        return lattice_graph

    def show_control_flow_function(self) -> AbstractEnvironmentUpdateInstances[L]:
        control_flow_function_title = Text(self.control_flow_function_title)

        self.scale_mobject(
            control_flow_function_title,
            self.control_flow_function_title_width,
            self.control_flow_function_title_width,
        )
        control_flow_function_title.move_to(self.control_flow_function_title_position)

        control_flow_function_tex = AbstractEnvironmentUpdateInstances(
            self.control_flow_function.instances
        )

        self.scale_mobject(
            control_flow_function_tex,
            self.control_flow_function_width,
            self.control_flow_function_height,
        )
        control_flow_function_tex.move_to(self.control_flow_function_position)

        self.play(Create(control_flow_function_title))
        self.play(Create(control_flow_function_tex))

        self.wait(self.control_flow_function_wait_time)

        self.play(Uncreate(control_flow_function_title))

        return control_flow_function_tex

    def show_flow_function(self) -> AbstractEnvironmentUpdateInstances[L]:
        flow_function_title = Text(self.flow_function_title)

        self.scale_mobject(
            flow_function_title,
            self.flow_function_title_width,
            self.flow_function_title_width,
        )
        flow_function_title.move_to(self.flow_function_title_position)

        flow_function_tex = AbstractEnvironmentUpdateInstances(
            self.control_flow_function.flow_function.instances
        )

        self.scale_mobject(
            flow_function_tex,
            self.condition_update_function_width,
            self.condition_update_function_height,
        )
        flow_function_tex.move_to(self.flow_function_position)

        self.play(Create(flow_function_title))
        self.play(Create(flow_function_tex))

        self.wait(self.flow_function_wait_time)

        self.play(Uncreate(flow_function_title))

        return flow_function_tex

    def show_condition_update_function(self) -> AbstractEnvironmentUpdateInstances[L]:
        condition_update_function_title = Text(self.condition_update_function_title)

        self.scale_mobject(
            condition_update_function_title,
            self.condition_update_function_title_width,
            self.condition_update_function_title_width,
        )
        condition_update_function_title.move_to(
            self.condition_update_function_title_position
        )

        condition_update_function_tex = AbstractEnvironmentUpdateInstances(
            self.condition_update_function.instances
        )

        self.scale_mobject(
            condition_update_function_tex,
            self.condition_update_function_width,
            self.condition_update_function_height,
        )
        condition_update_function_tex.move_to(self.condition_update_function_position)

        self.play(Create(condition_update_function_title))
        self.play(Create(condition_update_function_tex))

        self.wait(self.condition_update_function_wait_time)

        self.play(Uncreate(condition_update_function_title))

        return condition_update_function_tex

    def show_program(self) -> Code:
        program_title = Text(self.program_title)

        self.scale_mobject(
            program_title,
            self.program_title_width,
            self.program_title_height,
        )
        program_title.move_to(self.program_title_position)

        program = Code(
            code=str(self.program),
            language="c",
            background="window",
            tab_width=4,
            font="Monospace",
            style="monokai",
        )

        self.scale_mobject(program, self.program_width, self.program_height)
        program.move_to(self.program_position)

        self.play(Write(program_title), Create(program))

        self.wait(self.program_wait_time)

        new_program = program.copy()

        self.scale_mobject(
            new_program,
            self.program_new_width,
            self.program_new_height,
        )
        new_program.move_to(self.program_new_position)

        self.play(Unwrite(program_title), Transform(program, new_program))

        self.remove(program)
        self.add(new_program)

        return new_program

    def show_program_conversion(
        self, program: Code
    ) -> tuple[ProgramPoint, nx.DiGraph[ProgramPoint], ControlFlowGraph]:
        arrow = Arrow(
            start=self.program_arrow_start_position,
            end=self.program_arrow_end_position,
        )

        program_conversion_title = Text(self.program_conversion_title)

        self.scale_mobject(
            program_conversion_title,
            self.program_conversion_title_width,
            self.program_conversion_title_height,
        )
        program_conversion_title.move_to(self.program_conversion_title_position)

        entry_point, program_cfg = self.program.to_cfg()

        cfg = ControlFlowGraph.from_cfg(entry_point, program_cfg)

        self.scale_mobject(cfg, self.cfg_width, self.cfg_height)
        cfg.move_to(self.cfg_position)

        self.play(Write(program_conversion_title))
        self.play(Create(arrow))
        self.play(Create(cfg))

        self.wait(self.program_conversion_wait_time)

        self.remove(program)

        self.play(Uncreate(arrow), Uncreate(program), Unwrite(program_conversion_title))

        return entry_point, program_cfg, cfg

    def show_lattice_join(
        self,
        lattice_graph: LatticeGraph[L],
        value1: L,
        value2: L,
        position: tuple[float, float, float] = (0, 0, 0),
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

        res_table = self.create_res_table(variables)

        worklist_tex = self.create_worklist_tex(worklist)

        self.play(Create(table), Create(res_table), Create(worklist_tex))

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

            # self.camera.frame.save_state()

            # self.play(
            #    self.camera.frame.animate.move_to((0, -self.camera.frame_height, 0))
            # )

            # self.show_control_flow_function_instance(res_instance_id)

            # self.play(self.camera.frame.animate.restore())

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

                # self.camera.frame.save_state()

                # self.play(
                #    self.camera.frame.animate.move_to((0, self.camera.frame_height, 0))
                # )

                # self.show_condition_update_function_instance(res_cond_instance_id)

                # self.play(self.camera.frame.animate.restore())

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
        self.play(self.camera.frame.animate.move_to(self.title_camera_position))

        self.show_title()

        self.play(self.camera.frame.animate.move_to(self.lattice_camera_position))

        lattice_graph = self.show_lattice_graph()

        self.play(
            self.camera.frame.animate.move_to(
                self.control_flow_function_camera_position
            )
        )

        control_flow_function_tex = self.show_control_flow_function()

        self.play(self.camera.frame.animate.move_to(self.flow_function_camera_position))

        flow_function_tex = self.show_flow_function()

        self.play(
            self.camera.frame.animate.move_to(
                self.condition_update_function_camera_position
            )
        )

        condition_update_function_tex = self.show_condition_update_function()

        self.play(self.camera.frame.animate.move_to(self.program_camera_position))

        program = self.show_program()

        self.play(
            self.camera.frame.animate.move_to(self.program_conversion_camera_position)
        )

        entry_point, program_cfg, cfg = self.show_program_conversion(program)

        self.play(self.camera.frame.animate.move_to(self.worklist_camera_position))

        self.worklist(
            entry_point,
            program_cfg,
            cfg,
            self.program.variables,
        )

        self.wait(5)
