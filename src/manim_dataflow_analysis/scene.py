from __future__ import annotations

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Generator,
    Generic,
    Hashable,
    Iterable,
    TypeVar,
    Set,
    Mapping,
)

from frozendict import frozendict
from manim.animation.creation import Create, Uncreate, Unwrite, Write
from manim.animation.transform import FadeTransform, Transform
from manim.camera.moving_camera import MovingCamera
from manim.mobject.geometry.line import Arrow
from manim.mobject.geometry.shape_matchers import SurroundingRectangle
from manim.mobject.mobject import Mobject
from manim.mobject.text.code_mobject import Code
from manim.mobject.text.text_mobject import Text
from manim.renderer.opengl_renderer import OpenGLCamera
from manim.scene.zoomed_scene import MovingCameraScene
from manim.utils.color import ORANGE, WHITE

from manim_dataflow_analysis.abstract_environment import (
    AbstractEnvironment,
    AbstractEnvironmentUpdateInstances,
)
from manim_dataflow_analysis.cfg import ControlFlowGraph, ProgramPoint
from manim_dataflow_analysis.lattice import (
    Lattice,
    LatticeGraph,
    default_sorting_function,
)
from manim_dataflow_analysis.worklist import (
    ResTable,
    WorklistTable,
    WorklistTex,
    WorklistListener,
    worklist_algorithm,
)
from manim_dataflow_analysis.scale import fh, fw, scale_mobject

if TYPE_CHECKING:
    import networkx as nx
    from manim.animation.animation import Animation
    from manim.mobject.types.vectorized_mobject import VMobject
    from manim.mobject.text.tex_mobject import SingleStringMathTex, Tex

    from manim_dataflow_analysis.ast import AstFunction, AstStatement
    from manim_dataflow_analysis.condition_update_function import (
        ConditionUpdateFunction,
    )
    from manim_dataflow_analysis.flow_function import ControlFlowFunction


L = TypeVar("L", bound=Hashable)
E = TypeVar("E", bound=Hashable)
M = TypeVar("M", bound=Mobject | None)


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
    lattice_title: str = "We will use the following lattice for our analysis :"
    lattice_title_width: float = fw(0.95)
    lattice_title_height: float = fh(0.175)
    lattice_title_position: tuple[float, float, float] = (fw(1), fh(0.3875), 0)
    lattice_width: float = fw(0.95)
    lattice_height: float = fh(0.775)
    lattice_position: tuple[float, float, float] = (fw(1), fh(-0.0775), 0)
    lattice_camera_position: tuple[float, float, float] = (fw(1), 0, 0)
    lattice_wait_time: float = 5.0
    lattice_join_title_template: str = (
        "We join {abstract_value1} and {abstract_value2} which results in {joined_abstract_value}"  # noqa: E501
    )
    lattice_join_wait_time: float = 5.0
    lattice_max_horizontal_size_per_vertex: int = 8
    lattice_max_vertical_size: int = 8
    sorting_function: Callable[[Iterable[Hashable]], list[Hashable]] = (
        default_sorting_function
    )

    # Control-flow function
    control_flow_function: ControlFlowFunction[L]
    control_flow_function_title: str = (
        "We will use the following control-flow function for our analysis :"
    )
    control_flow_function_title_width: float = fw(0.95)
    control_flow_function_title_height: float = fw(0.175)
    control_flow_function_title_position: tuple[float, float, float] = (
        0,
        fh(-0.6125),
        0,
    )
    control_flow_function_width: float = fw(0.95)
    control_flow_function_height: float = fh(0.775)
    control_flow_function_position: tuple[float, float, float] = (0, fh(-1.0775), 0)
    control_flow_function_camera_position: tuple[float, float, float] = (0, fh(-1), 0)
    control_flow_function_wait_time: float = 5.0
    control_flow_function_highlight_wait_time: float = 2.5
    control_flow_function_set_wait_time: float = 2.5

    # Flow function
    flow_function_title: str = (
        "We will use the following flow function to help us for our analysis :"
    )
    flow_function_title_width: float = fw(0.95)
    flow_function_title_height: float = fh(0.175)
    flow_function_title_position: tuple[float, float, float] = (0, fh(-1.6125), 0)
    flow_function_position: tuple[float, float, float] = (0, fh(-2.0775), 0)
    flow_function_camera_position: tuple[float, float, float] = (0, fh(-2), 0)
    flow_function_wait_time: float = 5.0
    flow_function_highlight_time: float = 2.5
    flow_function_set_wait_time: float = 2.5

    # Condition update function
    condition_update_function: ConditionUpdateFunction[L, E]
    condition_update_function_title: str = (
        "We will also use the following condition update function for our analysis :"
    )
    condition_update_function_title_width: float = fw(0.95)
    condition_update_function_title_height: float = fh(0.175)
    condition_update_function_title_position: tuple[float, float, float] = (
        0,
        fh(1.3875),
        0,
    )
    condition_update_function_width: float = fw(0.95)
    condition_update_function_height: float = fh(0.775)
    condition_update_function_position: tuple[float, float, float] = (0, fh(0.9225), 0)
    condition_update_function_camera_position: tuple[float, float, float] = (
        0,
        fh(1),
        0,
    )
    condition_update_function_wait_time: float = 5.0
    condition_update_function_highlight_wait_time: float = 2.5
    condition_update_function_set_wait_time: float = 2.5

    # Program
    program: AstFunction
    program_title: str = "Here is the program that we are going to analyse :"
    program_title_width: float = fw(0.95)
    program_title_height: float = fh(0.175)
    program_title_position: tuple[float, float, float] = (fw(-1), fh(0.3875), 0)
    program_width: float = fw(0.5)
    program_height: float = fh(0.775)
    program_position: tuple[float, float, float] = (fw(-1), fh(-0.0775), 0)
    program_camera_position: tuple[float, float, float] = (fw(-1), 0, 0)
    program_wait_time: float = 2.5

    # Program conversion
    program_conversion_title: str = (
        "First, we need to convert it into a control flow graph :"
    )
    program_conversion_title_width: float = fw(0.95)
    program_conversion_title_height: float = fh(0.175)
    program_conversion_title_position: tuple[float, float, float] = (
        fw(-0.45),
        fh(0.3875),
        0,
    )
    program_new_width: float = fw(0.25)
    program_new_height: float = fh(0.775)
    program_new_position: tuple[float, float, float] = (fw(-0.75), fh(-0.0775), 0)
    program_arrow_start_position: tuple[float, float, float] = (fw(-0.6), fh(-0.1), 0)
    program_arrow_end_position: tuple[float, float, float] = (fw(-0.5), fh(-0.1), 0)
    program_conversion_camera_position: tuple[float, float, float] = (fw(-0.45), 0, 0)
    program_conversion_wait_time: float = 2.5

    # CFG
    cfg_title_width: float = fw(0.45)
    cfg_title_height: float = fh(0.175)
    cfg_title_position: tuple[float, float, float] = (fw(-0.25), fh(0.3875), 0)
    cfg_width: float = fw(0.45)
    cfg_height: float = fh(0.775)
    cfg_position: tuple[float, float, float] = (fw(-0.25), fh(-0.0775), 0)
    cfg_wait_time: float = 5.0

    # Worklist
    worklist_pop_title_template: str = (
        "We remove the program point {program_point} from the worklist :"
    )
    worklist_control_flow_function_title_template: str = (
        "We use the control-flow function on our program point {program_point} which is the statement {statement} :"  # noqa: E501
    )
    worklist_flow_function_title_template: str = (
        "We use the flow function on our statement {statement} :"
    )
    worklist_condition_update_function_title_template: str = (
        "We use the condition update function on our condition {condition} :"
    )
    worklist_control_flow_variables_title_template: str = (
        "We update the res abstract environment with the variables\n{variables} coming from the control flow function :"  # noqa: E501
    )
    worklist_table_variables_title_template: str = (
        "We update the rest of the res abstract environment with the variables\n{variables} coming from the abstract environment {program_point} :"  # noqa: E501
    )
    worklist_successor_title_template: str = (
        "We try to check if we need to process the successor {successor_program_point} :"  # noqa: E501
    )
    worklist_condition_update_variables_title_template: str = (
        "We update the res[COND(p,p')] abstract environment with the variables\n{variables} coming from the condition update function :"  # noqa: E501
    )
    worklist_res_variables_title_template: str = (
        "We update the rest of the res[COND(p,p')] abstract environment with the variables\n{variables} coming from the res abstract environment :"  # noqa: E501
    )
    worklist_is_included_title_template: str = (
        "res[COND(p,p')] is included in the abstract environment {successor_program_point}\nso we reached a fixed point :"  # noqa: E501
    )
    worklist_not_included_title_template: str = (
        "res[COND(p,p')] is not included in the abstract environment {successor_program_point}\nso we must process the successor {successor_program_point} :"  # noqa: E501
    )
    worklist_joined_values_title_template: str = (
        "We join the values from the abstract environment res[COND(p,p')] with\nthe abstract environment {program_point} :"  # noqa: E501
    )
    worklist_add_successor_title_template: str = (
        "We add the successor {program_point} to the worklist :"
    )
    worklist_camera_position: tuple[float, float, float] = (0, 0, 0)
    worklist_pop_wait_time: float = 5.0
    worklist_control_flow_variables_wait_time: float = 5.0
    worklist_table_variables_wait_time: float = 5.0
    worklist_successor_wait_time: float = 5.0
    worklist_condition_update_variables_wait_time: float = 5.0
    worklist_res_variables_wait_time: float = 5.0
    worklist_included_wait_time: float = 5.0
    worklist_joined_values_wait_time: float = 5.0
    worklist_add_successor_wait_time: float = 5.0
    worklist_wait_time: float = 5.0

    def show_title(self) -> None:
        title = Text(self.title)

        scale_mobject(title, self.title_width, self.title_height)
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
            layout_config={"sorting_function": type(self).sorting_function},
        )

        scale_mobject(lattice_graph, self.lattice_width, self.lattice_height)
        lattice_graph.move_to(self.lattice_position)

        return lattice_graph

    def show_lattice_graph(self) -> LatticeGraph[L]:
        lattice_title = Text(self.lattice_title)

        scale_mobject(
            lattice_title, self.lattice_title_width, self.lattice_title_height
        )
        lattice_title.move_to(self.lattice_title_position)

        lattice_graph = self.create_lattice_graph()

        self.play(Write(lattice_title), Create(lattice_graph))

        self.wait(self.lattice_wait_time)

        self.play(Unwrite(lattice_title))

        return lattice_graph

    def show_control_flow_function(self) -> AbstractEnvironmentUpdateInstances:
        control_flow_function_title = Text(self.control_flow_function_title)

        scale_mobject(
            control_flow_function_title,
            self.control_flow_function_title_width,
            self.control_flow_function_title_width,
        )
        control_flow_function_title.move_to(self.control_flow_function_title_position)

        control_flow_function_tex = AbstractEnvironmentUpdateInstances(
            self.control_flow_function.instances
        )

        scale_mobject(
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

    def show_flow_function(self) -> AbstractEnvironmentUpdateInstances | None:
        if self.control_flow_function.flow_function is None:
            return None

        flow_function_title = Text(self.flow_function_title)

        scale_mobject(
            flow_function_title,
            self.flow_function_title_width,
            self.flow_function_title_width,
        )
        flow_function_title.move_to(self.flow_function_title_position)

        flow_function_tex = AbstractEnvironmentUpdateInstances(
            self.control_flow_function.flow_function.instances
        )

        scale_mobject(
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

    def show_condition_update_function(self) -> AbstractEnvironmentUpdateInstances:
        condition_update_function_title = Text(self.condition_update_function_title)

        scale_mobject(
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

        scale_mobject(
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

        scale_mobject(
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
            line_no_from=self.program.line_number,
        )

        scale_mobject(program, self.program_width, self.program_height)
        program.move_to(self.program_position)

        self.play(Write(program_title), Create(program))

        self.wait(self.program_wait_time)

        new_program = program.copy()

        scale_mobject(
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

        scale_mobject(
            program_conversion_title,
            self.program_conversion_title_width,
            self.program_conversion_title_height,
        )
        program_conversion_title.move_to(self.program_conversion_title_position)

        entry_point, program_cfg = self.program.to_cfg()

        cfg = ControlFlowGraph.from_cfg(entry_point, program_cfg)

        scale_mobject(cfg, self.cfg_width, self.cfg_height)
        cfg.move_to(self.cfg_position)

        self.play(Write(program_conversion_title))
        self.play(Create(arrow))
        self.play(Create(cfg))

        self.wait(self.program_conversion_wait_time)

        self.remove(program)

        self.play(Uncreate(arrow), Uncreate(program), Unwrite(program_conversion_title))

        return entry_point, program_cfg, cfg

    def show_flow_function_instance(
        self,
        flow_function_tex: AbstractEnvironmentUpdateInstances,
        instance_id: int,
        statement: AstStatement,
        control_flow_modification: VMobject,
        control_flow_modification_rectangle: SurroundingRectangle,
    ):
        instance = flow_function_tex.get_instance_part(instance_id)
        condition = flow_function_tex.get_condition_part(instance_id)
        modification = flow_function_tex.get_modification_part(instance_id)

        worklist_flow_function_title = Text(
            self.worklist_flow_function_title_template.format(
                statement=str(statement.header),
            )
        )
        scale_mobject(
            worklist_flow_function_title,
            self.flow_function_title_width,
            self.flow_function_title_height,
        )
        worklist_flow_function_title.move_to(self.flow_function_title_position)

        instance_rectangle = SurroundingRectangle(instance)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        if condition_rectangle is not None:
            self.play(
                self.move_camera_animation(self.flow_function_camera_position),
                Create(worklist_flow_function_title),
                Create(condition_rectangle),
                Transform(control_flow_modification, instance),
                Transform(control_flow_modification_rectangle, instance_rectangle),
            )
        else:
            self.play(
                Create(worklist_flow_function_title),
                Transform(control_flow_modification, instance),
                Transform(control_flow_modification_rectangle, instance_rectangle),
            )

        self.wait(self.flow_function_highlight_time)

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

        self.play(
            Uncreate(worklist_flow_function_title),
            Uncreate(modification_rectangle),
        )

    def show_control_flow_function_instance(
        self,
        control_flow_function_tex: AbstractEnvironmentUpdateInstances,
        flow_function_tex: AbstractEnvironmentUpdateInstances | None,
        instance_id: int | tuple[int, int],
        program_point: ProgramPoint,
    ):
        if isinstance(instance_id, int):
            flow_instance_id = None
        else:
            instance_id, flow_instance_id = instance_id

        instance = control_flow_function_tex.get_instance_part(instance_id)
        condition = control_flow_function_tex.get_condition_part(instance_id)
        modification = control_flow_function_tex.get_modification_part(instance_id)

        worklist_control_flow_function_title = Text(
            self.worklist_control_flow_function_title_template.format(
                program_point=str(program_point.point),
                statement=str(program_point.statement.header),
            )
        )
        scale_mobject(
            worklist_control_flow_function_title,
            self.control_flow_function_title_width,
            self.control_flow_function_title_height,
        )
        worklist_control_flow_function_title.move_to(
            self.control_flow_function_title_position
        )

        instance_rectangle = SurroundingRectangle(instance)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        if condition_rectangle is not None:
            self.play(
                Create(worklist_control_flow_function_title),
                Create(instance_rectangle),
                Create(condition_rectangle),
            )
        else:
            self.play(
                Create(worklist_control_flow_function_title),
                Create(instance_rectangle),
            )

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

            if condition_rectangle is not None:
                self.remove(
                    instance_rectangle, condition_rectangle, modification_rectangle
                )
            else:
                self.remove(instance_rectangle, modification_rectangle)

            assert flow_function_tex is not None

            self.show_flow_function_instance(
                flow_function_tex,
                flow_instance_id,
                program_point.statement,
                modification_copy,
                modification_rectangle_copy,
            )

            self.play(Uncreate(worklist_control_flow_function_title))
        else:
            if condition_rectangle is not None:
                self.remove(instance_rectangle, condition_rectangle)
            else:
                self.remove(instance_rectangle)

            self.play(
                Uncreate(worklist_control_flow_function_title),
                Uncreate(modification_rectangle),
            )

    def show_condition_update_function_instance(
        self,
        condition_update_function_tex: AbstractEnvironmentUpdateInstances,
        instance_id: int,
        condition_expression: E,
    ):
        instance = condition_update_function_tex.get_instance_part(instance_id)
        condition = condition_update_function_tex.get_condition_part(instance_id)
        modification = condition_update_function_tex.get_modification_part(instance_id)

        worklist_condition_update_function_title = Text(
            self.worklist_condition_update_function_title_template.format(
                condition=str(condition_expression),
            )
        )
        scale_mobject(
            worklist_condition_update_function_title,
            self.condition_update_function_title_width,
            self.condition_update_function_title_height,
        )
        worklist_condition_update_function_title.move_to(
            self.condition_update_function_title_position
        )

        instance_rectangle = SurroundingRectangle(instance)
        modification_rectangle = SurroundingRectangle(modification)
        if condition is not None:
            condition_rectangle = SurroundingRectangle(condition)
        else:
            condition_rectangle = None

        if condition_rectangle is not None:
            self.play(
                Create(worklist_condition_update_function_title),
                Create(instance_rectangle),
                Create(condition_rectangle),
            )
        else:
            self.play(
                Create(worklist_condition_update_function_title),
                Create(instance_rectangle),
            )

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

        self.play(
            Uncreate(worklist_condition_update_function_title),
            Uncreate(modification_rectangle),
        )

    @contextmanager
    def animate_mobject(
        self,
        previous_mobject: Mobject | None,
        new_mobject: M,
    ) -> Generator[tuple[Animation | None, M], None, None]:
        if previous_mobject is not None and new_mobject is not None:
            yield Transform(previous_mobject, new_mobject), new_mobject
            self.remove(previous_mobject)
            self.add(new_mobject)
        elif previous_mobject is not None and new_mobject is None:
            yield Uncreate(previous_mobject), new_mobject  # type: ignore[misc]
        elif new_mobject is not None and previous_mobject is None:
            yield Create(new_mobject), new_mobject
        else:
            yield None, new_mobject

    def create_worklist_table(
        self,
        variables: Collection[str],
        abstract_environments: dict[ProgramPoint, AbstractEnvironment[L]],
    ) -> WorklistTable[L]:
        table = WorklistTable(variables, abstract_environments)

        scale_mobject(
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

        scale_mobject(
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

        scale_mobject(
            worklist_tex,
            self.camera.frame_width * 0.45,
            self.camera.frame_height * 0.10,
        )

        worklist_tex.move_to(
            (self.camera.frame_width * 0.25, self.camera.frame_height * 0.425, 0)
        )

        return worklist_tex

    def move_camera_animation(self, position: tuple[float, float, float]) -> Animation:
        if isinstance(self.camera, MovingCamera):
            return self.camera.frame.animate.move_to(position)
        elif isinstance(self.camera, OpenGLCamera):
            return self.camera.animate.move_to(position)
        else:
            raise NotImplementedError("Unsupported camera type")

    def create_worklist_listener(
        self,
        lattice_graph: LatticeGraph[L],
        cfg: ControlFlowGraph,
        control_flow_function_tex: AbstractEnvironmentUpdateInstances,
        flow_function_tex: AbstractEnvironmentUpdateInstances | None,
        condition_update_function_tex: AbstractEnvironmentUpdateInstances,
    ) -> WorklistListener[L, E]:
        class WorklistAnimation:

            @classmethod
            def before_worklist_creation(
                cls,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
            ):
                pass

            table: WorklistTable[L]
            res_table: ResTable[L]
            worklist_tex: WorklistTex

            @classmethod
            def after_worklist_creation(
                cls,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
            ):
                cls.table = self.create_worklist_table(variables, abstract_environments)

                cls.res_table = self.create_res_table(variables)

                cls.worklist_tex = self.create_worklist_tex(worklist)

                self.play(
                    Create(cls.table), Create(cls.res_table), Create(cls.worklist_tex)
                )

            @classmethod
            def before_iteration(
                cls,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
            ):
                pass

            worklist_pop_title: Text
            worklist_program_point_rectangle: SurroundingRectangle
            table_program_point_rectangle: SurroundingRectangle | None
            program_point_rectangle: SurroundingRectangle | None

            @classmethod
            def after_program_point_selection(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
            ):
                cls.worklist_pop_title = Text(
                    self.worklist_pop_title_template.format(
                        program_point=str(program_point.point)
                    )
                )
                scale_mobject(
                    cls.worklist_pop_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_pop_title.move_to(self.cfg_title_position)

                cls.worklist_program_point_rectangle = SurroundingRectangle(
                    cls.worklist_tex.get_program_point_part(program_point)
                )

                self.play(
                    Create(cls.worklist_pop_title),
                    Create(cls.worklist_program_point_rectangle),
                )

                cls.table_program_point_rectangle = None
                cls.program_point_rectangle = None

                with (
                    self.animate_mobject(
                        cls.worklist_tex,
                        self.create_worklist_tex(worklist),
                    ) as (
                        worklist_animation,
                        cls.worklist_tex,
                    ),
                    self.animate_mobject(
                        cls.program_point_rectangle,
                        SurroundingRectangle(cfg[program_point]),
                    ) as (
                        cls.program_point_animation,
                        cls.program_point_rectangle,
                    ),
                    self.animate_mobject(
                        cls.table_program_point_rectangle,
                        SurroundingRectangle(
                            cls.table.get_program_point_part(program_point)
                        ),
                    ) as (
                        cls.table_program_point_animation,
                        cls.table_program_point_rectangle,
                    ),
                ):
                    self.play(
                        Uncreate(cls.worklist_program_point_rectangle),
                        worklist_animation,
                        cls.program_point_animation,
                        cls.table_program_point_animation,
                    )

                self.wait(self.worklist_pop_wait_time)

            control_flow_function_instance: VMobject
            control_flow_function_result: VMobject
            program_point_label: SingleStringMathTex | Text | Tex
            worklist_control_flow_variables_title: Text
            res_table: ResTable[L]
            worklist_table_variables_title: Text
            table_successor_program_point_rectangle: SurroundingRectangle | None
            successor_program_point_rectangle: SurroundingRectangle | None

            @classmethod
            def after_control_flow_function_application(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
            ):
                if isinstance(res_instance_id, int):
                    cls.control_flow_function_instance = (
                        control_flow_function_tex.get_instance_part(res_instance_id)
                    )
                    cls.control_flow_function_result = (
                        control_flow_function_tex.get_modification_part(res_instance_id)
                    )
                else:
                    instance_id, flow_instance_id = res_instance_id
                    cls.control_flow_function_instance = (
                        control_flow_function_tex.get_instance_part(instance_id)
                    )
                    assert flow_function_tex is not None
                    cls.control_flow_function_result = (
                        flow_function_tex.get_modification_part(flow_instance_id)
                    )

                cls.program_point_label = cfg.labels[program_point].copy()
                cls.program_point_label.color = WHITE

                self.add(control_flow_function_tex, flow_function_tex)

                self.play(
                    Uncreate(cls.worklist_pop_title),
                    Transform(
                        cls.program_point_label, cls.control_flow_function_instance
                    ),
                    self.move_camera_animation(
                        self.control_flow_function_camera_position
                    ),
                )
                self.remove(cls.program_point_label)

                self.show_control_flow_function_instance(
                    control_flow_function_tex,
                    flow_function_tex,
                    res_instance_id,
                    program_point,
                )

                cls.worklist_control_flow_variables_title = Text(
                    self.worklist_control_flow_variables_title_template.format(
                        variables=", ".join(variable for variable in res_variables)
                    )
                )
                scale_mobject(
                    cls.worklist_control_flow_variables_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_control_flow_variables_title.move_to(
                    self.cfg_title_position
                )

                with (
                    self.animate_mobject(
                        cls.res_table,
                        self.create_res_table(
                            variables,
                            AbstractEnvironment(
                                self.lattice, frozendict(res_variables)
                            ),
                        ),
                    ) as (res_table_animation, cls.res_table),
                ):
                    res_parts = tuple(
                        (
                            cls.control_flow_function_result.copy(),
                            cls.res_table.get_res_variable_part(variable),
                        )
                        for variable in res_variables
                    )
                    self.play(
                        Create(cls.worklist_control_flow_variables_title),
                        self.move_camera_animation(self.worklist_camera_position),
                        res_table_animation,
                        *(Transform(*res_part) for res_part in res_parts),
                    )
                    self.remove(
                        control_flow_function_tex,
                        flow_function_tex,
                        *(part for part, _ in res_parts),
                    )

                self.wait(self.worklist_control_flow_variables_wait_time)

                cls.worklist_table_variables_title = Text(
                    self.worklist_table_variables_title_template.format(
                        program_point=str(program_point.point),
                        variables=", ".join(
                            variable
                            for variable in variables
                            if variable not in res_variables
                        ),
                    )
                )
                scale_mobject(
                    cls.worklist_table_variables_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_table_variables_title.move_to(self.cfg_title_position)

                with (
                    self.animate_mobject(
                        cls.res_table,
                        self.create_res_table(variables, res),
                    ) as (res_table_animation, cls.res_table),
                ):
                    control_flow_parts = tuple(
                        (
                            cls.table.get_variable_part(program_point, variable).copy(),
                            cls.res_table.get_res_variable_part(variable),
                        )
                        for variable in variables
                        if variable not in res_variables
                    )
                    self.play(
                        Transform(
                            cls.worklist_control_flow_variables_title,
                            cls.worklist_table_variables_title,
                        ),
                        res_table_animation,
                        *(
                            Transform(*control_flow_part)
                            for control_flow_part in control_flow_parts
                        ),
                    )
                    self.remove(*(part for part, _ in control_flow_parts))

                self.add(cls.worklist_table_variables_title)
                self.remove(cls.worklist_control_flow_variables_title)

                self.wait(self.worklist_table_variables_wait_time)

                self.play(Uncreate(cls.worklist_table_variables_title))

                cls.table_successor_program_point_rectangle = None
                cls.successor_program_point_rectangle = None

            worklist_successor_title: Text

            @classmethod
            def before_successor_iteration(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                successor: ProgramPoint,
            ):
                cls.worklist_successor_title = Text(
                    self.worklist_successor_title_template.format(
                        successor_program_point=str(successor.point),
                    )
                )
                scale_mobject(
                    cls.worklist_successor_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_successor_title.move_to(self.cfg_title_position)

                with (
                    self.animate_mobject(
                        cls.successor_program_point_rectangle,
                        SurroundingRectangle(cfg[successor], color=ORANGE),
                    ) as (
                        successor_program_point_animation,
                        cls.successor_program_point_rectangle,
                    ),
                    self.animate_mobject(
                        cls.table_successor_program_point_rectangle,
                        SurroundingRectangle(
                            cls.table.get_program_point_part(successor), color=ORANGE
                        ),
                    ) as (
                        table_successor_program_point_animation,
                        cls.table_successor_program_point_rectangle,
                    ),
                ):
                    self.play(
                        Create(cls.worklist_successor_title),
                        successor_program_point_animation,
                        table_successor_program_point_animation,
                    )

                self.wait(self.worklist_successor_wait_time)

                self.play(Uncreate(cls.worklist_successor_title))

            successor_program_point_label: SingleStringMathTex | Text | Tex
            condition_update_function_instance: VMobject
            condition_update_function_result: VMobject
            worklist_condition_update_variables_title: Text
            worklist_res_variables_title: Text

            @classmethod
            def after_condition_update_function_application(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
            ):
                cls.program_point_label = cfg.labels[program_point].copy()
                cls.program_point_label.color = WHITE
                cls.successor_program_point_label = cfg.labels[successor].copy()
                cls.successor_program_point_label.color = WHITE

                cls.condition_update_function_instance = (
                    condition_update_function_tex.get_instance_part(
                        res_cond_instance_id
                    )
                )
                cls.condition_update_function_result = (
                    condition_update_function_tex.get_modification_part(
                        res_cond_instance_id
                    )
                )

                self.add(condition_update_function_tex)

                self.play(
                    Transform(
                        cls.program_point_label, cls.condition_update_function_instance
                    ),
                    Transform(
                        cls.successor_program_point_label,
                        cls.condition_update_function_instance,
                    ),
                    self.move_camera_animation(
                        self.condition_update_function_camera_position
                    ),
                )
                self.remove(cls.program_point_label, cls.successor_program_point_label)

                self.show_condition_update_function_instance(
                    condition_update_function_tex,
                    res_cond_instance_id,
                    condition,
                )

                cls.worklist_condition_update_variables_title = Text(
                    self.worklist_condition_update_variables_title_template.format(
                        variables=", ".join(variable for variable in res_cond_variables)
                    )
                )
                scale_mobject(
                    cls.worklist_condition_update_variables_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_condition_update_variables_title.move_to(
                    self.cfg_title_position
                )

                with (
                    self.animate_mobject(
                        cls.res_table,
                        self.create_res_table(
                            variables,
                            res,
                            AbstractEnvironment(
                                self.lattice, frozendict(res_cond_variables)
                            ),
                        ),
                    ) as (res_cond_table_animation, cls.res_table),
                ):
                    res_cond_parts = tuple(
                        (
                            cls.condition_update_function_result.copy(),
                            cls.res_table.get_res_cond_variable_part(variable),
                        )
                        for variable in res_cond_variables
                    )
                    self.play(
                        Create(cls.worklist_condition_update_variables_title),
                        self.move_camera_animation(self.worklist_camera_position),
                        res_cond_table_animation,
                        *(
                            Transform(*res_cond_part)
                            for res_cond_part in res_cond_parts
                        ),
                    )
                    self.remove(*(part for part, _ in res_cond_parts))

                self.remove(condition_update_function_tex)

                self.wait(self.worklist_condition_update_variables_wait_time)

                cls.worklist_res_variables_title = Text(
                    self.worklist_res_variables_title_template.format(
                        variables=", ".join(
                            variable
                            for variable in variables
                            if variable not in res_cond_variables
                        ),
                    )
                )
                scale_mobject(
                    cls.worklist_res_variables_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_res_variables_title.move_to(self.cfg_title_position)

                with (
                    self.animate_mobject(
                        cls.res_table,
                        self.create_res_table(variables, res, res_cond),
                    ) as (res_cond_table_animation, cls.res_table),
                ):
                    condition_update_parts = tuple(
                        (
                            cls.res_table.get_res_variable_part(variable).copy(),
                            cls.res_table.get_res_cond_variable_part(variable),
                        )
                        for variable in variables
                        if variable not in res_cond_variables
                    )
                    self.play(
                        Transform(
                            cls.worklist_condition_update_variables_title,
                            cls.worklist_res_variables_title,
                        ),
                        res_cond_table_animation,
                        *(
                            Transform(*condition_update_part)
                            for condition_update_part in condition_update_parts
                        ),
                    )
                    self.remove(*(part for part, _ in condition_update_parts))

                self.add(cls.worklist_res_variables_title)
                self.remove(cls.worklist_condition_update_variables_title)

                self.wait(self.worklist_res_variables_wait_time)

                self.play(Uncreate(cls.worklist_res_variables_title))

            worklist_included_title: Text

            @classmethod
            def after_included(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
                included: bool,
            ):
                if included:
                    cls.worklist_included_title = Text(
                        self.worklist_is_included_title_template.format(
                            successor_program_point=str(successor.point)
                        )
                    )
                else:
                    cls.worklist_included_title = Text(
                        self.worklist_not_included_title_template.format(
                            successor_program_point=str(successor.point),
                        )
                    )
                scale_mobject(
                    cls.worklist_included_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_included_title.move_to(self.cfg_title_position)

                self.play(Create(cls.worklist_included_title))

                self.wait(self.worklist_included_wait_time)

                self.play(Uncreate(cls.worklist_included_title))

            @classmethod
            def after_not_included(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
            ):
                cls.worklist_joined_values_title = Text(
                    self.worklist_joined_values_title_template.format(
                        program_point=str(successor.point)
                    )
                )
                scale_mobject(
                    cls.worklist_joined_values_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_joined_values_title.move_to(self.cfg_title_position)

                self.play(Create(cls.worklist_joined_values_title))

                self.wait(self.worklist_joined_values_wait_time)

            new_lattice_graph: LatticeGraph[L]
            successor_program_point_part: VMobject
            res_cond_part: VMobject
            lattice_res_cond_part: VMobject
            lattice_successor_part: VMobject
            lattice_join_title: Text
            lattice_joined_part: VMobject

            @classmethod
            def while_join(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
                variable: str,
                joined_abstract_value: L,
                current_abstract_value: L,
                successor_abstract_value: L,
            ):
                cls.new_lattice_graph = self.create_lattice_graph(
                    {
                        current_abstract_value,
                        successor_abstract_value,
                        joined_abstract_value,
                    }
                )

                self.add(lattice_graph)
                self.play(FadeTransform(lattice_graph, cls.new_lattice_graph))

                cls.new_lattice_graph.color_path(
                    current_abstract_value, joined_abstract_value
                )
                cls.new_lattice_graph.color_path(
                    successor_abstract_value, joined_abstract_value
                )

                self.add(cls.new_lattice_graph)
                self.remove(lattice_graph)

                cls.successor_program_point_part = cls.table.get_variable_part(
                    successor, variable
                ).copy()
                cls.res_cond_part = cls.res_table.get_res_cond_variable_part(
                    variable
                ).copy()

                cls.lattice_res_cond_part = cls.new_lattice_graph.labels[
                    current_abstract_value
                ]
                cls.lattice_successor_part = cls.new_lattice_graph.labels[
                    successor_abstract_value
                ]

                cls.lattice_join_title = Text(
                    self.lattice_join_title_template.format(
                        abstract_value1=current_abstract_value,
                        abstract_value2=successor_abstract_value,
                        joined_abstract_value=joined_abstract_value,
                    )
                )
                scale_mobject(
                    cls.lattice_join_title,
                    self.lattice_title_width,
                    self.lattice_title_height,
                )
                cls.lattice_join_title.move_to(self.lattice_title_position)

                self.play(
                    Create(cls.lattice_join_title),
                    Transform(
                        cls.successor_program_point_part, cls.lattice_successor_part
                    ),
                    Transform(cls.res_cond_part, cls.lattice_res_cond_part),
                    self.move_camera_animation(self.lattice_camera_position),
                )

                self.remove(cls.successor_program_point_part, cls.res_cond_part)

                self.wait(self.lattice_join_wait_time)

                cls.lattice_joined_part = cls.new_lattice_graph.labels[
                    joined_abstract_value
                ].copy()

                with (
                    self.animate_mobject(
                        cls.table,
                        self.create_worklist_table(variables, abstract_environments),
                    ) as (
                        table_animation,
                        cls.table,
                    ),
                ):
                    self.play(
                        Uncreate(cls.lattice_join_title),
                        Transform(
                            cls.lattice_joined_part,
                            cls.table.get_variable_part(successor, variable),
                        ),
                        self.move_camera_animation(self.worklist_camera_position),
                        table_animation,
                    )

                self.remove(cls.lattice_joined_part, cls.new_lattice_graph)

            @classmethod
            def after_join(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
            ):
                self.play(Uncreate(cls.worklist_joined_values_title))

            @classmethod
            def after_add(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
            ):
                cls.worklist_add_successor_title = Text(
                    self.worklist_add_successor_title_template.format(
                        program_point=str(successor.point)
                    )
                )
                scale_mobject(
                    cls.worklist_add_successor_title,
                    self.cfg_title_width,
                    self.cfg_title_height,
                )
                cls.worklist_add_successor_title.move_to(self.cfg_title_position)

                with (
                    self.animate_mobject(
                        cls.worklist_tex,
                        self.create_worklist_tex(worklist),
                    ) as (
                        worklist_animation,
                        cls.worklist_tex,
                    ),
                ):
                    self.play(
                        Create(cls.worklist_add_successor_title),
                        worklist_animation,
                    )

                self.wait(self.worklist_add_successor_wait_time)

                self.play(Uncreate(cls.worklist_add_successor_title))

            @classmethod
            def after_successor_iteration(
                cls,
                program_point: ProgramPoint,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
                successor: ProgramPoint,
                condition: E,
                res_cond: AbstractEnvironment[L] | None,
                res_cond_variables: Mapping[str, L] | None,
                res_cond_instance_id: int,
            ):
                with (
                    self.animate_mobject(
                        cls.res_table,
                        self.create_res_table(variables, res),
                    ) as (
                        res_table_animation,
                        cls.res_table,
                    ),
                ):
                    self.play(res_table_animation)

            @classmethod
            def after_iteration(
                cls,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
                worklist: Set[ProgramPoint],
                res: AbstractEnvironment[L],
                res_variables: Mapping[str, L],
                res_instance_id: int,
            ):
                with (
                    self.animate_mobject(
                        cls.res_table,
                        self.create_res_table(variables),
                    ) as (
                        res_table_animation,
                        cls.res_table,
                    ),
                    self.animate_mobject(
                        cls.program_point_rectangle,
                        None,
                    ) as (
                        program_point_animation,
                        cls.program_point_rectangle,
                    ),
                    self.animate_mobject(
                        cls.table_program_point_rectangle,
                        None,
                    ) as (
                        table_program_point_animation,
                        cls.table_program_point_rectangle,
                    ),
                    self.animate_mobject(
                        cls.successor_program_point_rectangle,
                        None,
                    ) as (
                        successor_program_point_animation,
                        cls.successor_program_point_rectangle,
                    ),
                    self.animate_mobject(
                        cls.table_successor_program_point_rectangle,
                        None,
                    ) as (
                        table_successor_program_point_animation,
                        cls.table_successor_program_point_rectangle,
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

            @classmethod
            def after_worklist_algorithm(
                cls,
                variables: Set[str],
                abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
            ):
                self.wait(self.worklist_wait_time)

        return WorklistAnimation

    def worklist(
        self,
        entry_point: ProgramPoint,
        program_cfg: nx.DiGraph[ProgramPoint],
        cfg: ControlFlowGraph,
        lattice_graph: LatticeGraph[L],
        control_flow_function_tex: AbstractEnvironmentUpdateInstances,
        flow_function_tex: AbstractEnvironmentUpdateInstances | None,
        condition_update_function_tex: AbstractEnvironmentUpdateInstances,
    ):
        listener = self.create_worklist_listener(
            lattice_graph,
            cfg,
            control_flow_function_tex,
            flow_function_tex,
            condition_update_function_tex,
        )

        worklist_algorithm(
            self.program.parameters,
            self.program.variables,
            self.lattice,
            self.control_flow_function,
            self.condition_update_function,
            entry_point,
            program_cfg,
            listener,
        )

    def construct(self):
        self.play(self.move_camera_animation(self.title_camera_position))

        self.show_title()

        self.play(self.move_camera_animation(self.lattice_camera_position))

        lattice_graph = self.show_lattice_graph()

        self.play(
            self.move_camera_animation(self.control_flow_function_camera_position)
        )

        # Temporary remove tex to improve performance
        self.remove(lattice_graph)

        control_flow_function_tex = self.show_control_flow_function()

        self.play(self.move_camera_animation(self.flow_function_camera_position))

        flow_function_tex = self.show_flow_function()

        self.play(
            self.move_camera_animation(self.condition_update_function_camera_position)
        )

        condition_update_function_tex = self.show_condition_update_function()

        self.play(self.move_camera_animation(self.program_camera_position))

        # Temporary remove tex to improve performance
        if flow_function_tex is not None:
            self.remove(flow_function_tex)

        self.remove(
            control_flow_function_tex,
            condition_update_function_tex,
        )

        program = self.show_program()

        self.play(self.move_camera_animation(self.program_conversion_camera_position))

        entry_point, program_cfg, cfg = self.show_program_conversion(program)

        self.play(self.move_camera_animation(self.worklist_camera_position))

        self.worklist(
            entry_point,
            program_cfg,
            cfg,
            lattice_graph,
            control_flow_function_tex,
            flow_function_tex,
            condition_update_function_tex,
        )
