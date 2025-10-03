from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Collection,
    Generic,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
)

from frozendict import frozendict
from manim.mobject.table import MobjectTable
from manim.mobject.text.tex_mobject import MathTex
from manim.mobject.text.text_mobject import Text
from sortedcontainers import SortedDict, SortedSet

from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim_dataflow_analysis.cfg import ProgramPoint, cond, succ

if TYPE_CHECKING:
    import networkx as nx

    from manim_dataflow_analysis.condition_update_function import (
        ConditionUpdateFunction,
    )
    from manim_dataflow_analysis.flow_function import ControlFlowFunction
    from manim_dataflow_analysis.lattice import Lattice
    from manim_dataflow_analysis.widening_operator import WideningOperator

L = TypeVar("L")
E = TypeVar("E")


class WorklistTex(MathTex):
    def __init__(self, worklist: SortedSet[ProgramPoint]):
        self.worklist = worklist.copy()

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
        variables: SortedSet[str],
        abstract_environments: SortedDict[ProgramPoint, AbstractEnvironment[L]],
    ):
        self.values_mobjects = SortedDict(
            lambda p: p.point,
            {
                program_point: SortedDict(
                    lambda v: v,
                    {
                        variable: Text(str(abstract_environment[variable]))
                        for variable in variables
                    },
                )
                for program_point, abstract_environment in abstract_environments.items()
            },
        )
        self.program_points_mobjects = SortedDict(
            lambda p: p.point,
            {
                program_point: Text(str(program_point.point))
                for program_point in abstract_environments
            },
        )
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

    def get_variable_part(self, program_point: ProgramPoint, variable: str) -> Text:
        return self.values_mobjects[program_point][variable]

    def get_program_point_part(self, program_point: ProgramPoint) -> Text:
        return self.program_points_mobjects[program_point]


class ResTable(MobjectTable, Generic[L]):
    def __init__(
        self,
        variables: Collection[str],
        res: AbstractEnvironment[L] | None = None,
        res_cond: AbstractEnvironment[L] | None = None,
    ):
        self.res_mobjects = {
            variable: Text(
                str(res[variable]) if res is not None and variable in res else "?"
            )
            for variable in variables
        }
        self.res_cond_mobjects = {
            variable: Text(
                str(res_cond[variable])
                if res_cond is not None and variable in res_cond
                else "?"
            )
            for variable in variables
        }

        table = [
            [
                Text("Variables"),
                *(MathTex(rf"\phi_p({variable})") for variable in variables),
            ],
            [
                Text("res"),
                *(self.res_mobjects[variable] for variable in variables),
            ],
            [
                Text("res[COND(p, p')]"),
                *(self.res_cond_mobjects[variable] for variable in variables),
            ],
        ]

        super().__init__(table, include_outer_lines=True)

    def get_res_variable_part(self, variable: str) -> Text:
        return self.res_mobjects[variable]

    def get_res_cond_variable_part(self, variable: str) -> Text:
        return self.res_cond_mobjects[variable]


class BeforeWorklistCreationDict(TypedDict, Generic[L]):
    variables: SortedSet[str]
    abstract_environments: SortedDict[ProgramPoint, AbstractEnvironment[L]]


class AfterWorklistCreationDict(BeforeWorklistCreationDict[L]):
    worklist: SortedSet[ProgramPoint]


class AfterProgramPointSelectionDict(AfterWorklistCreationDict[L]):
    program_point: ProgramPoint


class AfterControlFlowFunctionApplicationDict(AfterProgramPointSelectionDict[L]):
    res: AbstractEnvironment[L]
    res_variables: dict[str, L]
    res_instance_id: int | tuple[int, int]


class BeforeSuccessorIterationDict(AfterControlFlowFunctionApplicationDict[L]):
    successor: ProgramPoint


class AfterConditionUpdateFunctionApplicationDict(
    BeforeSuccessorIterationDict[L], Generic[L, E]
):
    condition: E
    res_cond: AbstractEnvironment[L] | None
    res_cond_variables: dict[str, L] | None
    res_cond_instance_id: int


class AfterIncludedDict(AfterConditionUpdateFunctionApplicationDict[L, E]):
    included: bool


class WhileJoinDict(AfterIncludedDict[L, E]):
    variable: str
    joined_abstract_value: L
    current_abstract_value: L
    successor_abstract_value: L


class WhileWidenDict(AfterIncludedDict[L, E]):
    variable: str
    widened_abstract_value: L
    current_abstract_value: L
    successor_abstract_value: L
    widened_instance_id: int


EX_contra = TypeVar("EX_contra", contravariant=True)


class WorklistListener(Protocol[L, E, EX_contra]):
    def before_worklist_creation(
        self,
        data: BeforeWorklistCreationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called before the worklist is created."""

    def after_worklist_creation(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after the worklist is created."""

    def before_iteration(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called before an iteration of the worklist algorithm."""

    def after_program_point_selection(
        self,
        data: AfterProgramPointSelectionDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after a program point is selected from the worklist."""

    def after_control_flow_function_application(
        self,
        data: AfterControlFlowFunctionApplicationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after the control flow function is applied."""

    def before_successor_iteration(
        self,
        data: BeforeSuccessorIterationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called before a successor is processed."""

    def after_not_defined(
        self,
        data: BeforeSuccessorIterationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called if the successor abstract environment is not defined."""

    def after_condition_update_function_application(
        self,
        data: AfterConditionUpdateFunctionApplicationDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called after the condition update function is applied."""

    def after_unreachable_code(
        self,
        data: AfterConditionUpdateFunctionApplicationDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called after the condition update function detects unreachable code."""

    def after_included(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called after the res_cond abstract environment is included in the successor abstract environment."""  # noqa: E501

    def after_not_included(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called after the res_cond abstract environment is not included in the successor abstract environment."""  # noqa: E501

    def while_join(
        self,
        data: WhileJoinDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called while the join operation is applied."""

    def after_join(
        self,
        data: AfterIncludedDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called after the join operation is applied."""

    def while_widen(
        self,
        data: WhileWidenDict[L, E],
        extra_data: EX_contra,
    ) -> None:
        """Called while the widening operation is applied."""

    def after_add(
        self,
        data: BeforeSuccessorIterationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after a successor is added to the worklist."""

    def after_successor_iteration(
        self,
        data: BeforeSuccessorIterationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after a successor is processed."""

    def after_iteration(
        self,
        data: AfterControlFlowFunctionApplicationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after an iteration of the worklist algorithm."""

    def after_worklist_algorithm(
        self,
        data: AfterWorklistCreationDict[L],
        extra_data: EX_contra,
    ) -> None:
        """Called after the worklist algorithm is finished."""


def worklist_algorithm(
    parameters: tuple[str, ...],
    variables: frozenset[str],
    lattice: Lattice[L],
    widening_operator: WideningOperator[L],
    control_flow_function: ControlFlowFunction[L],
    condition_update_function: ConditionUpdateFunction[L, E],
    entry_point: ProgramPoint,
    program_cfg: nx.DiGraph[ProgramPoint],
    listener: WorklistListener[L, E, EX_contra],
    extra_data: EX_contra,
):
    data: BeforeWorklistCreationDict[L] = {
        "variables": SortedSet((*variables, *parameters)),
        "abstract_environments": SortedDict(
            lambda p: p.point,
            {
                entry_point: AbstractEnvironment(
                    lattice,
                    frozendict(
                        (
                            *((variable, lattice.bottom()) for variable in variables),
                            *((parameter, lattice.top()) for parameter in parameters),
                        )
                    ),
                )
            },
        ),
    }

    listener.before_worklist_creation(data, extra_data)

    data = cast(AfterWorklistCreationDict[L], data)

    data["worklist"] = SortedSet((entry_point,), key=lambda p: p.point)

    listener.after_worklist_creation(data, extra_data)

    while data["worklist"]:
        listener.before_iteration(data, extra_data)

        data = cast(AfterProgramPointSelectionDict[L], data)

        data["program_point"] = data["worklist"].pop(0)

        listener.after_program_point_selection(data, extra_data)

        data = cast(AfterControlFlowFunctionApplicationDict[L], data)

        (
            data["res"],
            res_variables,
            data["res_instance_id"],
        ) = control_flow_function.apply_and_get_variables(
            data["program_point"],
            data["abstract_environments"][data["program_point"]],
        )

        data["res_variables"] = dict(res_variables)

        listener.after_control_flow_function_application(data, extra_data)

        for successor in succ(program_cfg, data["program_point"]):
            data = cast(BeforeSuccessorIterationDict[L], data)

            data["successor"] = successor

            listener.before_successor_iteration(data, extra_data)

            data = cast(AfterConditionUpdateFunctionApplicationDict[L, E], data)

            data["condition"] = cond(
                program_cfg,
                data["program_point"],
                data["successor"],
            )

            (
                data["res_cond"],
                res_cond_variables,
                data["res_cond_instance_id"],
            ) = condition_update_function.apply_and_get_variables(
                data["condition"],
                data["res"],
            )

            data["res_cond_variables"] = (
                None if res_cond_variables is None else dict(res_cond_variables)
            )

            listener.after_condition_update_function_application(data, extra_data)

            if data["res_cond"] is None:
                listener.after_unreachable_code(data, extra_data)
            elif data["successor"] not in data["abstract_environments"]:
                assert data["res_cond"] is not None

                data["abstract_environments"][data["successor"]] = data["res_cond"]
                listener.after_not_defined(data, extra_data)

                data["worklist"].add(data["successor"])

                listener.after_add(data, extra_data)
            else:
                assert data["res_cond"] is not None

                data = cast(AfterIncludedDict[L, E], data)

                data["included"] = data["abstract_environments"][
                    data["successor"]  # type: ignore
                ].includes(
                    data["res_cond"]  # type: ignore
                )

                listener.after_included(data, extra_data)

                if not data["included"]:
                    listener.after_not_included(data, extra_data)

                    if widening_operator is None:
                        for variable, joined_abstract_value in data[
                            "abstract_environments"
                        ][
                            data["successor"]  # type: ignore
                        ].join_generator(
                            data["res_cond"]  # type: ignore
                        ):
                            data = cast(WhileJoinDict[L, E], data)

                            data["variable"] = variable
                            data["joined_abstract_value"] = joined_abstract_value
                            data["current_abstract_value"] = data["res_cond"][  # type: ignore
                                data["variable"]
                            ]
                            data["successor_abstract_value"] = data[
                                "abstract_environments"
                            ][data["successor"]][data["variable"]]

                            data["abstract_environments"][data["successor"]] = data[
                                "abstract_environments"
                            ][data["successor"]].set(
                                {data["variable"]: data["joined_abstract_value"]}
                            )

                            listener.while_join(data, extra_data)
                    else:
                        for (
                            variable,
                            widened_abstract_value,
                            widened_instance_id,
                        ) in widening_operator.join_generator(
                            data["abstract_environments"][data["successor"]],
                            data["res_cond"],  # type: ignore
                        ):
                            data = cast(WhileWidenDict[L, E], data)

                            data["variable"] = variable
                            data["widened_abstract_value"] = widened_abstract_value
                            data["widened_instance_id"] = widened_instance_id
                            data["current_abstract_value"] = data["res_cond"][  # type: ignore
                                data["variable"]
                            ]
                            data["successor_abstract_value"] = data[
                                "abstract_environments"
                            ][data["successor"]][data["variable"]]

                            data["abstract_environments"][data["successor"]] = data[
                                "abstract_environments"
                            ][data["successor"]].set(
                                {data["variable"]: data["widened_abstract_value"]}
                            )

                            listener.while_widen(data, extra_data)

                    listener.after_join(data, extra_data)

                    data["worklist"].add(data["successor"])

                    listener.after_add(data, extra_data)

            listener.after_successor_iteration(data, extra_data)

        listener.after_iteration(data, extra_data)

    listener.after_worklist_algorithm(data, extra_data)

    return data["abstract_environments"]
