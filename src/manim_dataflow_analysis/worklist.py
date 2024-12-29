from __future__ import annotations

from typing import Collection, Generic, Mapping, Set, TypeVar, Protocol, TypedDict

from manim.mobject.table import MobjectTable
from manim.mobject.text.tex_mobject import MathTex

from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint, succ, cond
from manim_dataflow_analysis.lattice import Lattice
from manim_dataflow_analysis.flow_function import ControlFlowFunction
from manim_dataflow_analysis.condition_update_function import ConditionUpdateFunction

from frozendict import frozendict
import networkx as nx

L = TypeVar("L")
E = TypeVar("E")


class WorklistTex(MathTex):
    def __init__(self, worklist: Set[ProgramPoint]):
        self.worklist = tuple(worklist)

        tex_strings = [r"W = \{"]
        if self.worklist:
            for program_point in self.worklist[:-1]:
                tex_strings.append(str(program_point.point))
                tex_strings.append(",")
            tex_strings.append(str(self.worklist[-1].point))
        tex_strings.append(r"\}")

        super().__init__(*tex_strings)

    def get_program_point_part(self, program_point: ProgramPoint) -> MathTex:
        return self.submobjects[1 + self.worklist.index(program_point) * 2]


class WorklistTable(MobjectTable, Generic[L]):
    def __init__(
        self,
        variables: Collection[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
    ):
        self.values_mobjects = {
            program_point: {
                variable: MathTex(str(abstract_environment[variable]))
                for variable in variables
            }
            for program_point, abstract_environment in abstract_environments.items()
        }
        self.program_points_mobjects = {
            program_point: MathTex(str(program_point.point))
            for program_point in abstract_environments
        }
        table = [
            [
                MathTex("p"),
                *(MathTex(rf"\phi_p({variable})") for variable in variables),
            ],
            *(
                [
                    self.program_points_mobjects[program_point],
                    *(mobjects[variable] for variable in variables),
                ]
                for program_point, mobjects in self.values_mobjects.items()
            ),
        ]

        super().__init__(table, include_outer_lines=True)

    def get_variable_part(self, program_point: ProgramPoint, variable: str) -> MathTex:
        return self.values_mobjects[program_point][variable]

    def get_program_point_part(self, program_point: ProgramPoint) -> MathTex:
        return self.program_points_mobjects[program_point]


class ResTable(MobjectTable, Generic[L]):
    def __init__(
        self,
        variables: Collection[str],
        res: AbstractEnvironment[L] | None = None,
        res_cond: AbstractEnvironment[L] | None = None,
    ):
        self.res_mobjects = {
            variable: MathTex(
                str(res[variable]) if res is not None and variable in res else "?"
            )
            for variable in variables
        }
        self.res_cond_mobjects = {
            variable: MathTex(
                str(res_cond[variable])
                if res_cond is not None and variable in res_cond
                else "?"
            )
            for variable in variables
        }

        table = [
            [
                MathTex("Variables"),
                *(MathTex(rf"\phi_p({variable})") for variable in variables),
            ],
            [
                MathTex("res"),
                *(self.res_mobjects[variable] for variable in variables),
            ],
            [
                MathTex("res[COND(p, p')]"),
                *(self.res_cond_mobjects[variable] for variable in variables),
            ],
        ]

        super().__init__(table, include_outer_lines=True)

    def get_res_variable_part(self, variable: str) -> MathTex:
        return self.res_mobjects[variable]

    def get_res_cond_variable_part(self, variable: str) -> MathTex:
        return self.res_cond_mobjects[variable]


class BeforeWorklistCreationDict(TypedDict, Generic[L]):
    variables: Set[str]
    abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]]


class AfterWorklistCreationDict(BeforeWorklistCreationDict[L]):
    worklist: Set[ProgramPoint]


class AfterProgramPointSelectionDict(AfterWorklistCreationDict[L]):
    program_point: ProgramPoint


class AfterControlFlowFunctionApplicationDict(AfterProgramPointSelectionDict[L]):
    res: AbstractEnvironment[L]
    res_variables: Mapping[str, L]
    res_instance_id: int


class BeforeSuccessorIterationDict(AfterControlFlowFunctionApplicationDict[L]):
    successor: ProgramPoint


class AfterConditionUpdateFunctionApplicationDict(
    AfterControlFlowFunctionApplicationDict[L], Generic[L, E]
):
    condition: E
    res_cond: AbstractEnvironment[L] | None
    res_cond_variables: Mapping[str, L] | None
    res_cond_instance_id: int


class AfterIncludedDict(AfterConditionUpdateFunctionApplicationDict[L, E]):
    included: bool


class WhileJoinDict(AfterIncludedDict[L, E]):
    variable: str
    joined_abstract_value: L
    current_abstract_value: L
    successor_abstract_value: L


EX = TypeVar("EX", bound=dict[str, object])


class WorklistListener(Protocol[L, E, EX]):
    def create_extra_data(self) -> EX:
        """Create the extra data that is passed through the worklist algorithm."""
        return {}

    def before_worklist_creation(
        self,
        data: BeforeWorklistCreationDict[L],
        extra_data: EX,
    ) -> None:
        """Called before the worklist is created."""

    def after_worklist_creation(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: EX,
    ) -> None:
        """Called after the worklist is created."""

    def before_iteration(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: EX,
    ) -> None:
        """Called before an iteration of the worklist algorithm."""

    def after_program_point_selection(
        self,
        data: AfterProgramPointSelectionDict[L],
        extra_data: EX,
    ) -> None:
        """Called after a program point is selected from the worklist."""

    def after_control_flow_function_application(
        self,
        data: AfterControlFlowFunctionApplicationDict[L],
        extra_data: EX,
    ) -> None:
        """Called after the control flow function is applied."""

    def before_successor_iteration(
        self,
        data: BeforeSuccessorIterationDict[L],
        extra_data: EX,
    ) -> None:
        """Called before a successor is added to the worklist."""

    def after_condition_update_function_application(
        self,
        data: AfterConditionUpdateFunctionApplicationDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called after the condition update function is applied."""

    def after_included(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called after the res_cond abstract environment is included in the successor abstract environment."""

    def after_not_included(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called after the res_cond abstract environment is not included in the successor abstract environment."""

    def while_join(
        self,
        data: WhileJoinDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called while the join operation is applied."""

    def after_join(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called after the join operation is applied."""

    def after_add(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called after a successor is added to the worklist."""

    def after_successor_iteration(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX,
    ) -> None:
        """Called after a successor is added to the worklist."""

    def after_iteration(
        self,
        data: AfterControlFlowFunctionApplicationDict[L],
        extra_data: EX,
    ) -> None:
        """Called after an iteration of the worklist algorithm."""

    def after_worklist_algorithm(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: EX,
    ) -> None:
        """Called after the worklist algorithm is finished."""


def worklist_algorithm(
    parameters: set[str],
    variables: set[str],
    lattice: Lattice[L],
    control_flow_function: ControlFlowFunction[L],
    condition_update_function: ConditionUpdateFunction[L, E],
    entry_point: ProgramPoint,
    program_cfg: nx.DiGraph[ProgramPoint],
    listener: WorklistListener[L, E, EX],
):
    extra_data = listener.create_extra_data()

    data: BeforeWorklistCreationDict[L] = {
        "variables": variables.union(parameters),
        "abstract_environments": {
            p: AbstractEnvironment(
                lattice,
                frozendict(
                    (
                        *((variable, lattice.bottom()) for variable in variables),
                        *((parameter, lattice.top()) for parameter in parameters),
                    )
                ),
            )
            for p in program_cfg.nodes
        },
    }

    listener.before_worklist_creation(data, extra_data)

    data: BeforeWorklistCreationDict[L] | AfterWorklistCreationDict[L]

    data["worklist"] = {entry_point}

    listener.after_worklist_creation(data, extra_data)

    while data["worklist"]:
        listener.before_iteration(data, extra_data)

        data: AfterWorklistCreationDict[L] | AfterProgramPointSelectionDict[L]

        data["program_point"] = data["worklist"].pop()

        listener.after_program_point_selection(data, extra_data)

        data: (
            AfterProgramPointSelectionDict[L]
            | AfterControlFlowFunctionApplicationDict[L]
        )

        (
            data["res"],
            data["res_variables"],
            data["res_instance_id"],
        ) = control_flow_function.apply_and_get_variables(
            data["program_point"],
            data["abstract_environments"][data["program_point"]],
        )

        listener.after_control_flow_function_application(data, extra_data)

        for successor in succ(program_cfg, data["program_point"]):
            data: (
                AfterControlFlowFunctionApplicationDict[L]
                | BeforeSuccessorIterationDict[L]
            )

            data["successor"] = successor

            listener.before_successor_iteration(data, extra_data)

            data: (
                BeforeSuccessorIterationDict[L]
                | AfterConditionUpdateFunctionApplicationDict[L, E]
            )

            data["condition"] = cond(
                program_cfg,
                data["program_point"],
                data["successor"],
            )

            (
                data["res_cond"],
                data["res_cond_variables"],
                data["res_cond_instance_id"],
            ) = condition_update_function.apply_and_get_variables(
                data["condition"],
                data["res"],
            )

            listener.after_condition_update_function_application(data, extra_data)

            data: (
                AfterConditionUpdateFunctionApplicationDict[L, E]
                | AfterIncludedDict[L, E]
            )

            data["included"] = data["abstract_environments"][
                data["successor"]
            ].includes(data["res_cond"])

            listener.after_included(data, extra_data)

            if not data["included"]:
                listener.after_not_included(data, extra_data)

                for variable, joined_abstract_value in data["abstract_environments"][
                    data["successor"]
                ].join_generator(data["res_cond"]):
                    data: AfterIncludedDict[L, E] | WhileJoinDict[L, E]

                    data["variable"] = variable
                    data["joined_abstract_value"] = joined_abstract_value
                    data["current_abstract_value"] = data["res_cond"][data["variable"]]
                    data["successor_abstract_value"] = data["abstract_environments"][
                        data["successor"]
                    ][data["variable"]]

                    data["abstract_environments"][data["successor"]] = data[
                        "abstract_environments"
                    ][data["successor"]].set(
                        {data["variable"]: data["joined_abstract_value"]}
                    )

                    listener.while_join(data, extra_data)

                listener.after_join(data, extra_data)

                data["worklist"].add(data["successor"])

                listener.after_add(data, extra_data)

            listener.after_successor_iteration(data, extra_data)

        listener.after_iteration(data, extra_data)

    listener.after_worklist_algorithm(data, extra_data)

    return data, extra_data
