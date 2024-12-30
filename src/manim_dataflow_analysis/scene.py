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
    TypedDict,
    TypeVar,
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
from manim_dataflow_analysis.scale import fh, fw, scale_mobject
from manim_dataflow_analysis.widening_operator import (
    WideningOperator,
    WideningOperatorTex,
)
from manim_dataflow_analysis.worklist import (
    AfterConditionUpdateFunctionApplicationDict,
    AfterControlFlowFunctionApplicationDict,
    AfterIncludedDict,
    AfterProgramPointSelectionDict,
    AfterWorklistCreationDict,
    BeforeSuccessorIterationDict,
    BeforeWorklistCreationDict,
    ResTable,
    WhileJoinDict,
    WhileWidenDict,
    WorklistListener,
    WorklistTable,
    WorklistTex,
    worklist_algorithm,
)

if TYPE_CHECKING:
    import networkx as nx
    from manim.animation.animation import Animation
    from manim.mobject.text.tex_mobject import SingleStringMathTex, Tex
    from manim.mobject.types.vectorized_mobject import VMobject

    from manim_dataflow_analysis.ast import AstFunction, AstStatement
    from manim_dataflow_analysis.condition_update_function import (
        ConditionUpdateFunction,
    )
    from manim_dataflow_analysis.flow_function import ControlFlowFunction


L = TypeVar("L", bound=Hashable)
E = TypeVar("E", bound=Hashable)
M = TypeVar("M", bound=Mobject | None)


class WorklistExtraDataDict(TypedDict, Generic[L], total=False):
    cfg: ControlFlowGraph
    lattice_graph: LatticeGraph[L]
    widening_operator_tex: WideningOperatorTex
    control_flow_function_tex: AbstractEnvironmentUpdateInstances
    flow_function_tex: AbstractEnvironmentUpdateInstances | None
    condition_update_function_tex: AbstractEnvironmentUpdateInstances

    table: WorklistTable[L]
    res_table: ResTable[L]
    worklist_tex: WorklistTex

    worklist_pop_title: Text
    worklist_program_point_rectangle: SurroundingRectangle
    table_program_point_rectangle: SurroundingRectangle | None
    program_point_rectangle: SurroundingRectangle | None

    control_flow_function_instance: VMobject
    control_flow_function_result: VMobject
    program_point_label: SingleStringMathTex | Text | Tex
    worklist_control_flow_variables_title: Text
    worklist_table_variables_title: Text
    table_successor_program_point_rectangle: SurroundingRectangle | None
    successor_program_point_rectangle: SurroundingRectangle | None

    worklist_successor_title: Text

    successor_program_point_label: SingleStringMathTex | Text | Tex
    condition_update_function_instance: VMobject
    condition_update_function_result: VMobject
    worklist_condition_update_variables_title: Text
    worklist_res_variables_title: Text

    worklist_unreachable_title: Text
    worklist_included_title: Text

    new_lattice_graph: LatticeGraph[L]
    successor_program_point_part: VMobject
    res_cond_part: VMobject
    lattice_res_cond_part: VMobject
    lattice_successor_part: VMobject
    lattice_join_title: Text
    lattice_joined_part: VMobject
    widening_operator_instance_part: VMobject
    widening_operator_modification_part: VMobject
    widening_operator_widen_title: Text
    widening_operator_widened_part: VMobject


class AbstractAnalysisScene(
    MovingCameraScene,
    WorklistListener[L, E, WorklistExtraDataDict[L]],  # type: ignore
    Generic[L, E],
):
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
    lattice_join_title_template: str = "We join {abstract_value1} and {abstract_value2} which results in {joined_abstract_value}"  # noqa: E501
    lattice_join_wait_time: float = 5.0
    lattice_max_horizontal_size_per_vertex: int = 8
    lattice_max_vertical_size: int = 8
    sorting_function: Callable[
        [Iterable[Hashable]], list[Hashable]
    ] = default_sorting_function

    # Widening operator
    widening_operator: WideningOperator[L] | None = None
    widening_operator_title: str = (
        "We will use the following widening operator for our analysis :"  # noqa: E501
    )
    widening_operator_title_width: float = fw(0.95)
    widening_operator_title_height: float = fh(0.175)
    widening_operator_title_position: tuple[float, float, float] = (
        fw(1),
        fh(-0.6125),
        0,
    )
    widening_operator_width: float = fw(0.95)
    widening_operator_height: float = fh(0.775)
    widening_operator_position: tuple[float, float, float] = (fw(1), fh(-1.0775), 0)
    widening_operator_camera_position: tuple[float, float, float] = (fw(1), fh(-1), 0)
    widening_operator_wait_time: float = 5.0
    widening_operator_widen_title_template: str = "We apply the widening operator on {last_value} and {new_value} which results in {widened_abstract_value}"  # noqa: E501
    widening_operator_widen_highlight_wait_time: float = 2.5
    widening_operator_widen_set_time: float = 2.5

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
    worklist_control_flow_function_title_template: str = "We use the control-flow function on our program point {program_point} which is the statement {statement} :"  # noqa: E501
    worklist_flow_function_title_template: str = (
        "We use the flow function on our statement {statement} :"
    )
    worklist_condition_update_function_title_template: str = (
        "We use the condition update function on our condition {condition} :"
    )
    worklist_control_flow_variables_title_template: str = "We update the res abstract environment with the variables\n{variables} coming from the control flow function :"  # noqa: E501
    worklist_table_variables_title_template: str = "We update the rest of the res abstract environment with the variables\n{variables} coming from the abstract environment {program_point} :"  # noqa: E501
    worklist_successor_title_template: str = "We try to check if we need to process the successor {successor_program_point} :"  # noqa: E501
    worklist_unreachable_title_template: str = (
        "We found the unreachable successor {successor_program_point} so we skip it :"
    )
    worklist_condition_update_variables_title_template: str = "We update the res[COND(p,p')] abstract environment with the variables\n{variables} coming from the condition update function :"  # noqa: E501
    worklist_res_variables_title_template: str = "We update the rest of the res[COND(p,p')] abstract environment with the variables\n{variables} coming from the res abstract environment :"  # noqa: E501
    worklist_is_included_title_template: str = "res[COND(p,p')] is included in the abstract environment {successor_program_point}\nso we reached a fixed point :"  # noqa: E501
    worklist_not_included_title_template: str = "res[COND(p,p')] is not included in the abstract environment {successor_program_point}\nso we must process the successor {successor_program_point} :"  # noqa: E501
    worklist_joined_values_title_template: str = "We join the values from the abstract environment res[COND(p,p')] with\nthe abstract environment {program_point} :"  # noqa: E501
    worklist_add_successor_title_template: str = (
        "We add the successor {program_point} to the worklist :"
    )
    worklist_camera_position: tuple[float, float, float] = (0, 0, 0)
    worklist_pop_wait_time: float = 5.0
    worklist_unreachable_wait_time: float = 5.0
    worklist_control_flow_variables_wait_time: float = 5.0
    worklist_table_width: float = fw(0.45)
    worklist_table_height: float = fh(0.55)
    worklist_table_position: tuple[float, float, float] = (fw(0.25), fh(-0.2), 0)
    worklist_table_variables_wait_time: float = 5.0
    worklist_successor_wait_time: float = 5.0
    worklist_condition_update_variables_wait_time: float = 5.0
    worklist_res_table_width: float = fw(0.45)
    worklist_res_table_height: float = fh(0.25)
    worklist_res_table_position: tuple[float, float, float] = (fw(0.25), fh(0.225), 0)
    worklist_res_variables_wait_time: float = 5.0
    worklist_included_wait_time: float = 5.0
    worklist_joined_values_wait_time: float = 5.0
    worklist_add_successor_wait_time: float = 5.0
    worklist_wait_time: float = 5.0
    worklist_tex_width: float = fw(0.45)
    worklist_tex_height: float = fh(0.10)
    worklist_tex_position: tuple[float, float, float] = (fw(0.25), fh(0.425), 0)

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

    def show_widening_operator(self) -> WideningOperatorTex:
        widening_operator_title = Text(self.widening_operator_title)

        scale_mobject(
            widening_operator_title,
            self.widening_operator_title_width,
            self.widening_operator_title_height,
        )
        widening_operator_title.move_to(self.widening_operator_title_position)

        assert self.widening_operator is not None

        widening_operator_tex = WideningOperatorTex(self.widening_operator.instances)

        scale_mobject(
            widening_operator_tex,
            self.widening_operator_width,
            self.widening_operator_height,
        )
        widening_operator_tex.move_to(self.widening_operator_position)

        self.play(Write(widening_operator_title), Create(widening_operator_tex))

        self.wait(self.widening_operator_wait_time)

        self.play(Unwrite(widening_operator_title))

        return widening_operator_tex

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
            self.worklist_table_width,
            self.worklist_table_height,
        )

        table.move_to(self.worklist_table_position)

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
            self.worklist_res_table_width,
            self.worklist_res_table_height,
        )

        res_table.move_to(self.worklist_res_table_position)

        return res_table

    def create_worklist_tex(self, worklist: set[ProgramPoint]) -> WorklistTex:
        worklist_tex = WorklistTex(worklist)

        scale_mobject(
            worklist_tex,
            self.worklist_tex_width,
            self.worklist_tex_height,
        )

        worklist_tex.move_to(self.worklist_tex_position)

        return worklist_tex

    def move_camera_animation(self, position: tuple[float, float, float]) -> Animation:
        if isinstance(self.camera, MovingCamera):
            return self.camera.frame.animate.move_to(position)
        elif isinstance(self.camera, OpenGLCamera):
            return self.camera.animate.move_to(position)
        else:
            raise NotImplementedError("Unsupported camera type")

    def before_worklist_creation(
        self,
        data: BeforeWorklistCreationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        pass

    def after_worklist_creation(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["table"] = self.create_worklist_table(
            data["variables"],
            data["abstract_environments"],
        )
        extra_data["res_table"] = self.create_res_table(data["variables"])
        extra_data["worklist_tex"] = self.create_worklist_tex(data["worklist"])

        self.play(
            Create(extra_data["table"]),
            Create(extra_data["res_table"]),
            Create(extra_data["worklist_tex"]),
        )

    def before_iteration(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        pass

    def after_program_point_selection(
        self,
        data: AfterProgramPointSelectionDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["worklist_pop_title"] = Text(
            self.worklist_pop_title_template.format(
                program_point=str(data["program_point"].point)
            )
        )
        scale_mobject(
            extra_data["worklist_pop_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_pop_title"].move_to(self.cfg_title_position)

        extra_data["worklist_program_point_rectangle"] = SurroundingRectangle(
            extra_data["worklist_tex"].get_program_point_part(data["program_point"])
        )

        self.play(
            Create(extra_data["worklist_pop_title"]),
            Create(extra_data["worklist_program_point_rectangle"]),
        )

        extra_data["table_program_point_rectangle"] = None
        extra_data["program_point_rectangle"] = None

        with (
            self.animate_mobject(
                extra_data["worklist_tex"],
                self.create_worklist_tex(data["worklist"]),
            ) as (
                worklist_animation,
                extra_data["worklist_tex"],
            ),
            self.animate_mobject(
                extra_data["program_point_rectangle"],
                SurroundingRectangle(extra_data["cfg"][data["program_point"]]),
            ) as (
                program_point_animation,
                extra_data["program_point_rectangle"],
            ),
            self.animate_mobject(
                extra_data["table_program_point_rectangle"],
                SurroundingRectangle(
                    extra_data["table"].get_program_point_part(data["program_point"])
                ),
            ) as (
                table_program_point_animation,
                extra_data["table_program_point_rectangle"],
            ),
        ):
            self.play(
                Uncreate(extra_data["worklist_program_point_rectangle"]),
                worklist_animation,
                program_point_animation,
                table_program_point_animation,
            )

        self.wait(self.worklist_pop_wait_time)

        return extra_data

    def after_control_flow_function_application(
        self,
        data: AfterControlFlowFunctionApplicationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        if isinstance(data["res_instance_id"], int):
            extra_data["control_flow_function_instance"] = extra_data[
                "control_flow_function_tex"
            ].get_instance_part(data["res_instance_id"])
            extra_data["control_flow_function_result"] = extra_data[
                "control_flow_function_tex"
            ].get_modification_part(data["res_instance_id"])
        else:
            instance_id, flow_instance_id = data["res_instance_id"]
            extra_data["control_flow_function_instance"] = extra_data[
                "control_flow_function_tex"
            ].get_instance_part(instance_id)
            assert extra_data["flow_function_tex"] is not None
            extra_data["control_flow_function_result"] = extra_data[
                "flow_function_tex"
            ].get_modification_part(flow_instance_id)

        extra_data["program_point_label"] = (
            extra_data["cfg"].labels[data["program_point"]].copy()
        )
        extra_data["program_point_label"].color = WHITE

        self.add(
            extra_data["control_flow_function_tex"], extra_data["flow_function_tex"]
        )

        self.play(
            Uncreate(extra_data["worklist_pop_title"]),
            Transform(
                extra_data["program_point_label"],
                extra_data["control_flow_function_instance"],
            ),
            self.move_camera_animation(self.control_flow_function_camera_position),
        )
        self.remove(extra_data["program_point_label"])

        self.show_control_flow_function_instance(
            extra_data["control_flow_function_tex"],
            extra_data["flow_function_tex"],
            data["res_instance_id"],
            data["program_point"],
        )

        extra_data["worklist_control_flow_variables_title"] = Text(
            self.worklist_control_flow_variables_title_template.format(
                variables=", ".join(variable for variable in data["res_variables"])
            )
        )
        scale_mobject(
            extra_data["worklist_control_flow_variables_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_control_flow_variables_title"].move_to(
            self.cfg_title_position
        )

        with (
            self.animate_mobject(
                extra_data["res_table"],
                self.create_res_table(
                    data["variables"],
                    AbstractEnvironment(
                        self.lattice,
                        frozendict(data["res_variables"]),
                    ),
                ),
            ) as (res_table_animation, extra_data["res_table"]),
        ):
            res_parts = tuple(
                (
                    extra_data["control_flow_function_result"].copy(),
                    extra_data["res_table"].get_res_variable_part(variable),
                )
                for variable in data["res_variables"]
            )
            self.play(
                Create(extra_data["worklist_control_flow_variables_title"]),
                self.move_camera_animation(self.worklist_camera_position),
                res_table_animation,
                *(Transform(*res_part) for res_part in res_parts),
            )
            self.remove(
                extra_data["control_flow_function_tex"],
                extra_data["flow_function_tex"],
                *(part for part, _ in res_parts),
            )

        self.wait(self.worklist_control_flow_variables_wait_time)

        extra_data["worklist_table_variables_title"] = Text(
            self.worklist_table_variables_title_template.format(
                program_point=str(data["program_point"].point),
                variables=", ".join(
                    variable
                    for variable in data["variables"]
                    if variable not in data["res_variables"]
                ),
            )
        )
        scale_mobject(
            extra_data["worklist_table_variables_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_table_variables_title"].move_to(self.cfg_title_position)

        with (
            self.animate_mobject(
                extra_data["res_table"],
                self.create_res_table(data["variables"], data["res"]),
            ) as (res_table_animation, extra_data["res_table"]),
        ):
            control_flow_parts = tuple(
                (
                    extra_data["table"]
                    .get_variable_part(data["program_point"], variable)
                    .copy(),
                    extra_data["res_table"].get_res_variable_part(variable),
                )
                for variable in data["variables"]
                if variable not in data["res_variables"]
            )
            self.play(
                Transform(
                    extra_data["worklist_control_flow_variables_title"],
                    extra_data["worklist_table_variables_title"],
                ),
                res_table_animation,
                *(
                    Transform(*control_flow_part)
                    for control_flow_part in control_flow_parts
                ),
            )
            self.remove(*(part for part, _ in control_flow_parts))

        self.add(extra_data["worklist_table_variables_title"])
        self.remove(extra_data["worklist_control_flow_variables_title"])

        self.wait(self.worklist_table_variables_wait_time)

        self.play(Uncreate(extra_data["worklist_table_variables_title"]))

        extra_data["table_successor_program_point_rectangle"] = None
        extra_data["successor_program_point_rectangle"] = None

    def before_successor_iteration(
        self,
        data: BeforeSuccessorIterationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["worklist_successor_title"] = Text(
            self.worklist_successor_title_template.format(
                successor_program_point=str(data["successor"].point),
            )
        )
        scale_mobject(
            extra_data["worklist_successor_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_successor_title"].move_to(self.cfg_title_position)

        with (
            self.animate_mobject(
                extra_data["successor_program_point_rectangle"],
                SurroundingRectangle(
                    extra_data["cfg"][data["successor"]],
                    color=ORANGE,
                ),
            ) as (
                successor_program_point_animation,
                extra_data["successor_program_point_rectangle"],
            ),
            self.animate_mobject(
                extra_data["table_successor_program_point_rectangle"],
                SurroundingRectangle(
                    extra_data["table"].get_program_point_part(data["successor"]),
                    color=ORANGE,
                ),
            ) as (
                table_successor_program_point_animation,
                extra_data["table_successor_program_point_rectangle"],
            ),
        ):
            self.play(
                Create(extra_data["worklist_successor_title"]),
                successor_program_point_animation,
                table_successor_program_point_animation,
            )

        self.wait(self.worklist_successor_wait_time)

        self.play(Uncreate(extra_data["worklist_successor_title"]))

    def after_condition_update_function_application(
        self,
        data: AfterConditionUpdateFunctionApplicationDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["program_point_label"] = (
            extra_data["cfg"].labels[data["program_point"]].copy()
        )
        extra_data["program_point_label"].color = WHITE
        extra_data["successor_program_point_label"] = (
            extra_data["cfg"].labels[data["successor"]].copy()
        )
        extra_data["successor_program_point_label"].color = WHITE

        extra_data["condition_update_function_instance"] = extra_data[
            "condition_update_function_tex"
        ].get_instance_part(data["res_cond_instance_id"])
        extra_data["condition_update_function_result"] = extra_data[
            "condition_update_function_tex"
        ].get_modification_part(data["res_cond_instance_id"])

        self.add(extra_data["condition_update_function_tex"])

        self.play(
            Transform(
                extra_data["program_point_label"],
                extra_data["condition_update_function_instance"],
            ),
            Transform(
                extra_data["successor_program_point_label"],
                extra_data["condition_update_function_instance"],
            ),
            self.move_camera_animation(self.condition_update_function_camera_position),
        )
        self.remove(
            extra_data["program_point_label"],
            extra_data["successor_program_point_label"],
        )

        self.show_condition_update_function_instance(
            extra_data["condition_update_function_tex"],
            data["res_cond_instance_id"],
            data["condition"],
        )

        if data["res_cond_variables"] is None:
            self.play(self.move_camera_animation(self.worklist_camera_position))
            return

        extra_data["worklist_condition_update_variables_title"] = Text(
            self.worklist_condition_update_variables_title_template.format(
                variables=", ".join(variable for variable in data["res_cond_variables"])
            )
        )
        scale_mobject(
            extra_data["worklist_condition_update_variables_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_condition_update_variables_title"].move_to(
            self.cfg_title_position
        )

        with (
            self.animate_mobject(
                extra_data["res_table"],
                self.create_res_table(
                    data["variables"],
                    data["res"],
                    AbstractEnvironment(
                        self.lattice,
                        frozendict(data["res_cond_variables"]),
                    ),
                ),
            ) as (res_cond_table_animation, extra_data["res_table"]),
        ):
            res_cond_parts = tuple(
                (
                    extra_data["condition_update_function_result"].copy(),
                    extra_data["res_table"].get_res_cond_variable_part(variable),
                )
                for variable in data["res_cond_variables"]
            )
            self.play(
                Create(extra_data["worklist_condition_update_variables_title"]),
                self.move_camera_animation(self.worklist_camera_position),
                res_cond_table_animation,
                *(Transform(*res_cond_part) for res_cond_part in res_cond_parts),
            )
            self.remove(*(part for part, _ in res_cond_parts))

        self.remove(extra_data["condition_update_function_tex"])

        self.wait(self.worklist_condition_update_variables_wait_time)

        extra_data["worklist_res_variables_title"] = Text(
            self.worklist_res_variables_title_template.format(
                variables=", ".join(
                    variable
                    for variable in data["variables"]
                    if variable not in data["res_cond_variables"]
                ),
            )
        )
        scale_mobject(
            extra_data["worklist_res_variables_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_res_variables_title"].move_to(self.cfg_title_position)

        with (
            self.animate_mobject(
                extra_data["res_table"],
                self.create_res_table(
                    data["variables"],
                    data["res"],
                    data["res_cond"],
                ),
            ) as (res_cond_table_animation, extra_data["res_table"]),
        ):
            condition_update_parts = tuple(
                (
                    extra_data["res_table"].get_res_variable_part(variable).copy(),
                    extra_data["res_table"].get_res_cond_variable_part(variable),
                )
                for variable in data["variables"]
                if variable not in data["res_cond_variables"]
            )
            self.play(
                Transform(
                    extra_data["worklist_condition_update_variables_title"],
                    extra_data["worklist_res_variables_title"],
                ),
                res_cond_table_animation,
                *(
                    Transform(*condition_update_part)
                    for condition_update_part in condition_update_parts
                ),
            )
            self.remove(*(part for part, _ in condition_update_parts))

        self.add(extra_data["worklist_res_variables_title"])
        self.remove(extra_data["worklist_condition_update_variables_title"])

        self.wait(self.worklist_res_variables_wait_time)

        self.play(Uncreate(extra_data["worklist_res_variables_title"]))

    def after_unreachable_code(
        self,
        data: AfterConditionUpdateFunctionApplicationDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["worklist_unreachable_title"] = Text(
            self.worklist_unreachable_title_template.format(
                successor_program_point=str(data["successor"].point)
            )
        )
        scale_mobject(
            extra_data["worklist_unreachable_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_unreachable_title"].move_to(self.cfg_title_position)

        self.play(Create(extra_data["worklist_unreachable_title"]))

        self.wait(self.worklist_unreachable_wait_time)

        self.play(Uncreate(extra_data["worklist_unreachable_title"]))

    def after_included(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        if data["included"]:
            extra_data["worklist_included_title"] = Text(
                self.worklist_is_included_title_template.format(
                    successor_program_point=str(data["successor"].point)
                )
            )
        else:
            extra_data["worklist_included_title"] = Text(
                self.worklist_not_included_title_template.format(
                    successor_program_point=str(data["successor"].point),
                )
            )
        scale_mobject(
            extra_data["worklist_included_title"],
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_included_title"].move_to(self.cfg_title_position)

        self.play(Create(extra_data["worklist_included_title"]))

        self.wait(self.worklist_included_wait_time)

        self.play(Uncreate(extra_data["worklist_included_title"]))

    def after_not_included(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["worklist_joined_values_title"] = Text(  # type: ignore
            self.worklist_joined_values_title_template.format(
                program_point=str(data["successor"].point)
            )
        )
        scale_mobject(
            extra_data["worklist_joined_values_title"],  # type: ignore
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_joined_values_title"].move_to(self.cfg_title_position)  # type: ignore

        self.play(Create(extra_data["worklist_joined_values_title"]))  # type: ignore

        self.wait(self.worklist_joined_values_wait_time)

    def while_join(
        self,
        data: WhileJoinDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["new_lattice_graph"] = self.create_lattice_graph(  # type: ignore
            {
                data["current_abstract_value"],
                data["successor_abstract_value"],
                data["joined_abstract_value"],
            }
        )

        self.add(extra_data["lattice_graph"])  # type: ignore
        self.play(
            FadeTransform(
                extra_data["lattice_graph"],  # type: ignore
                extra_data["new_lattice_graph"],  # type: ignore
            )
        )

        extra_data["new_lattice_graph"].color_path(  # type: ignore
            data["current_abstract_value"], data["joined_abstract_value"]
        )
        extra_data["new_lattice_graph"].color_path(  # type: ignore
            data["successor_abstract_value"], data["joined_abstract_value"]
        )

        self.add(extra_data["new_lattice_graph"])
        self.remove(extra_data["lattice_graph"])

        extra_data["successor_program_point_part"] = (  # type: ignore
            extra_data["table"]  # type: ignore
            .get_variable_part(data["successor"], data["variable"])
            .copy()
        )
        extra_data["res_cond_part"] = (  # type: ignore
            extra_data["res_table"].get_res_cond_variable_part(data["variable"]).copy()
        )

        extra_data["lattice_res_cond_part"] = extra_data["new_lattice_graph"].labels[  # type: ignore
            data["current_abstract_value"]
        ]
        extra_data["lattice_successor_part"] = extra_data["new_lattice_graph"].labels[  # type: ignore
            data["successor_abstract_value"]
        ]

        extra_data["lattice_join_title"] = Text(
            self.lattice_join_title_template.format(
                abstract_value1=data["successor_abstract_value"],
                abstract_value2=data["current_abstract_value"],
                joined_abstract_value=data["joined_abstract_value"],
            )
        )
        scale_mobject(
            extra_data["lattice_join_title"],
            self.lattice_title_width,
            self.lattice_title_height,
        )
        extra_data["lattice_join_title"].move_to(self.lattice_title_position)

        self.play(
            Create(extra_data["lattice_join_title"]),
            Transform(
                extra_data["successor_program_point_part"],
                extra_data["lattice_successor_part"],
            ),
            Transform(
                extra_data["res_cond_part"],
                extra_data["lattice_res_cond_part"],
            ),
            self.move_camera_animation(self.lattice_camera_position),
        )

        self.remove(
            extra_data["successor_program_point_part"],
            extra_data["res_cond_part"],
        )

        self.wait(self.lattice_join_wait_time)

        extra_data["lattice_joined_part"] = (
            extra_data["new_lattice_graph"].labels[data["joined_abstract_value"]].copy()
        )

        with (
            self.animate_mobject(
                extra_data["table"],
                self.create_worklist_table(
                    data["variables"], data["abstract_environments"]
                ),
            ) as (
                table_animation,
                extra_data["table"],
            ),
        ):
            self.play(
                Uncreate(extra_data["lattice_join_title"]),
                Transform(
                    extra_data["lattice_joined_part"],
                    extra_data["table"].get_variable_part(
                        data["successor"], data["variable"]
                    ),
                ),
                self.move_camera_animation(self.worklist_camera_position),
                table_animation,
            )

        self.remove(
            extra_data["lattice_joined_part"],
            extra_data["new_lattice_graph"],
        )

    def while_widen(
        self,
        data: WhileWidenDict,
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["successor_program_point_part"] = (
            extra_data["table"]
            .get_variable_part(data["successor"], data["variable"])
            .copy()
        )
        extra_data["res_cond_part"] = (
            extra_data["res_table"].get_res_cond_variable_part(data["variable"]).copy()
        )

        extra_data["widening_operator_instance_part"] = extra_data[
            "widening_operator_tex"
        ].get_instance_part(data["widened_instance_id"])

        extra_data["widening_operator_widen_title"] = Text(
            self.widening_operator_widen_title_template.format(
                last_value=data["successor_abstract_value"],
                new_value=data["current_abstract_value"],
                widened_abstract_value=data["widened_abstract_value"],
            )
        )
        scale_mobject(
            extra_data["widening_operator_widen_title"],
            self.widening_operator_title_width,
            self.widening_operator_title_height,
        )
        extra_data["widening_operator_widen_title"].move_to(
            self.widening_operator_title_position
        )

        instance_rectangle = SurroundingRectangle(
            extra_data["widening_operator_instance_part"]
        )

        self.play(
            Create(instance_rectangle),
            Create(extra_data["widening_operator_widen_title"]),
            Transform(
                extra_data["successor_program_point_part"],
                extra_data["widening_operator_instance_part"],
            ),
            Transform(
                extra_data["res_cond_part"],
                extra_data["widening_operator_instance_part"],
            ),
            self.move_camera_animation(self.widening_operator_camera_position),
        )

        self.remove(
            extra_data["successor_program_point_part"],
            extra_data["res_cond_part"],
        )

        self.wait(self.widening_operator_widen_highlight_wait_time)

        extra_data["widening_operator_modification_part"] = (
            extra_data["widening_operator_tex"]
            .get_modification_part(data["widened_instance_id"])
            .copy()
        )

        modification_rectangle = SurroundingRectangle(
            extra_data["widening_operator_modification_part"]
        )

        self.play(
            Transform(
                instance_rectangle,
                modification_rectangle,
            ),
        )

        self.wait(self.widening_operator_widen_set_time)

        self.remove(instance_rectangle)

        with (
            self.animate_mobject(
                extra_data["table"],
                self.create_worklist_table(
                    data["variables"], data["abstract_environments"]
                ),
            ) as (
                table_animation,
                extra_data["table"],
            ),
        ):
            self.play(
                Uncreate(modification_rectangle),
                Uncreate(extra_data["widening_operator_widen_title"]),
                Transform(
                    extra_data["widening_operator_modification_part"],
                    extra_data["table"].get_variable_part(
                        data["successor"], data["variable"]
                    ),
                ),
                self.move_camera_animation(self.worklist_camera_position),
                table_animation,
            )

        self.remove(extra_data["widening_operator_modification_part"])

    def after_join(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        self.play(Uncreate(extra_data["worklist_joined_values_title"]))  # type: ignore

    def after_add(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        extra_data["worklist_add_successor_title"] = Text(  # type: ignore
            self.worklist_add_successor_title_template.format(
                program_point=str(data["successor"].point)
            )
        )
        scale_mobject(
            extra_data["worklist_add_successor_title"],  # type: ignore
            self.cfg_title_width,
            self.cfg_title_height,
        )
        extra_data["worklist_add_successor_title"].move_to(self.cfg_title_position)  # type: ignore

        with (
            self.animate_mobject(
                extra_data["worklist_tex"],  # type: ignore
                self.create_worklist_tex(data["worklist"]),
            ) as (
                worklist_animation,
                extra_data["worklist_tex"],  # type: ignore
            ),
        ):
            self.play(
                Create(extra_data["worklist_add_successor_title"]),  # type: ignore
                worklist_animation,
            )

        self.wait(self.worklist_add_successor_wait_time)

        self.play(Uncreate(extra_data["worklist_add_successor_title"]))  # type: ignore

    def after_successor_iteration(
        self,
        data: AfterConditionUpdateFunctionApplicationDict[L, E],
        extra_data: WorklistExtraDataDict[L],
    ):
        with (
            self.animate_mobject(
                extra_data["res_table"],  # type: ignore
                self.create_res_table(data["variables"], data["res"]),
            ) as (
                res_table_animation,
                extra_data["res_table"],  # type: ignore
            ),
        ):
            self.play(res_table_animation)

    def after_iteration(
        self,
        data: AfterControlFlowFunctionApplicationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        with (
            self.animate_mobject(
                extra_data["res_table"],  # type: ignore
                self.create_res_table(data["variables"]),
            ) as (
                res_table_animation,
                extra_data["res_table"],  # type: ignore
            ),
            self.animate_mobject(
                extra_data["program_point_rectangle"],  # type: ignore
                None,
            ) as (
                program_point_animation,
                extra_data["program_point_rectangle"],  # type: ignore
            ),
            self.animate_mobject(
                extra_data["table_program_point_rectangle"],  # type: ignore
                None,
            ) as (
                table_program_point_animation,
                extra_data["table_program_point_rectangle"],  # type: ignore
            ),
            self.animate_mobject(
                extra_data["successor_program_point_rectangle"],  # type: ignore
                None,
            ) as (
                successor_program_point_animation,
                extra_data["successor_program_point_rectangle"],  # type: ignore
            ),
            self.animate_mobject(
                extra_data["table_successor_program_point_rectangle"],  # type: ignore
                None,
            ) as (
                table_successor_program_point_animation,
                extra_data["table_successor_program_point_rectangle"],  # type: ignore
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

    def after_worklist_algorithm(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: WorklistExtraDataDict[L],
    ):
        self.wait(self.worklist_wait_time)

    def worklist(
        self,
        entry_point: ProgramPoint,
        program_cfg: nx.DiGraph[ProgramPoint],
        cfg: ControlFlowGraph,
        lattice_graph: LatticeGraph[L],
        widening_operator_tex: WideningOperatorTex | None,
        control_flow_function_tex: AbstractEnvironmentUpdateInstances,
        flow_function_tex: AbstractEnvironmentUpdateInstances | None,
        condition_update_function_tex: AbstractEnvironmentUpdateInstances,
    ):
        worklist_algorithm(
            set(self.program.parameters),
            self.program.variables,
            self.lattice,  # type: ignore
            self.widening_operator,  # type: ignore
            self.control_flow_function,
            self.condition_update_function,
            entry_point,
            program_cfg,
            self,
            {
                "cfg": cfg,
                "lattice_graph": lattice_graph,
                "widening_operator_tex": widening_operator_tex,
                "control_flow_function_tex": control_flow_function_tex,
                "flow_function_tex": flow_function_tex,
                "condition_update_function_tex": condition_update_function_tex,
            },
        )

    def construct(self):
        self.play(self.move_camera_animation(self.title_camera_position))

        self.show_title()

        self.play(self.move_camera_animation(self.lattice_camera_position))

        lattice_graph = self.show_lattice_graph()

        if self.widening_operator is not None:
            self.play(
                self.move_camera_animation(self.widening_operator_camera_position)
            )

            widening_operator_tex = self.show_widening_operator()
        else:
            widening_operator_tex = None

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
            widening_operator_tex,
            control_flow_function_tex,
            flow_function_tex,
            condition_update_function_tex,
        )
