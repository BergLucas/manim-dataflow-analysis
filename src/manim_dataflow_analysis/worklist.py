from __future__ import annotations

from typing import Collection, Generic, Mapping, Set, TypeVar, Protocol

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


class WorklistListener(Protocol[L, E]):
    def before_worklist_creation(
        self,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
    ):
        """Called before the worklist is created."""

    def after_worklist_creation(
        self,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
        worklist: Set[ProgramPoint],
    ):
        """Called after the worklist is created."""

    def before_iteration(
        self,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
        worklist: Set[ProgramPoint],
    ):
        """Called before an iteration of the worklist algorithm."""

    def after_program_point_selection(
        self,
        program_point: ProgramPoint,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
        worklist: Set[ProgramPoint],
    ):
        """Called after a program point is selected from the worklist."""

    def after_control_flow_function_application(
        self,
        program_point: ProgramPoint,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
        worklist: Set[ProgramPoint],
        res: AbstractEnvironment[L],
        res_variables: Mapping[str, L],
        res_instance_id: int,
    ):
        """Called after the control flow function is applied."""

    def before_successor_iteration(
        self,
        program_point: ProgramPoint,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
        worklist: Set[ProgramPoint],
        successor: ProgramPoint,
    ):
        """Called before a successor is added to the worklist."""

    def after_condition_update_function_application(
        self,
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
        """Called after the condition update function is applied."""

    def after_included(
        self,
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
        """Called after the res_cond abstract environment is included in the successor abstract environment."""

    def after_not_included(
        self,
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
        """Called after the res_cond abstract environment is not included in the successor abstract environment."""

    def while_join(
        self,
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
        """Called while the join operation is applied."""

    def after_join(
        self,
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
        """Called after the join operation is applied."""

    def after_add(
        self,
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
        """Called after a successor is added to the worklist."""

    def after_successor_iteration(
        self,
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
        """Called after a successor is added to the worklist."""

    def after_iteration(
        self,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
        worklist: Set[ProgramPoint],
        res: AbstractEnvironment[L],
        res_variables: Mapping[str, L],
        res_instance_id: int,
    ):
        """Called after an iteration of the worklist algorithm."""

    def after_worklist_algorithm(
        self,
        variables: Set[str],
        abstract_environments: Mapping[ProgramPoint, AbstractEnvironment[L]],
    ):
        """Called after the worklist algorithm is finished."""


def worklist_algorithm(
    parameters: set[str],
    variables: set[str],
    lattice: Lattice[L],
    control_flow_function: ControlFlowFunction[L],
    condition_update_function: ConditionUpdateFunction[L, E],
    entry_point: ProgramPoint,
    program_cfg: nx.DiGraph[ProgramPoint],
    listener: WorklistListener,
):
    abstract_environments = {
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
    }

    variables = variables.union(parameters)

    listener.before_worklist_creation(
        variables,
        abstract_environments,
    )

    worklist = {entry_point}

    listener.after_worklist_creation(
        variables,
        abstract_environments,
        worklist,
    )

    while worklist:
        listener.before_iteration(
            variables,
            abstract_environments,
            worklist,
        )

        program_point = worklist.pop()

        listener.after_program_point_selection(
            program_point,
            variables,
            abstract_environments,
            worklist,
        )

        (
            res,
            res_variables,
            res_instance_id,
        ) = control_flow_function.apply_and_get_variables(
            program_point, abstract_environments[program_point]
        )

        listener.after_control_flow_function_application(
            program_point,
            variables,
            abstract_environments,
            worklist,
            res,
            res_variables,
            res_instance_id,
        )

        for successor in succ(program_cfg, program_point):
            listener.before_successor_iteration(
                program_point,
                variables,
                abstract_environments,
                worklist,
                successor,
            )

            condition: E = cond(program_cfg, program_point, successor)

            (
                res_cond,
                res_cond_variables,
                res_cond_instance_id,
            ) = condition_update_function.apply_and_get_variables(
                condition,
                res,
            )

            listener.after_condition_update_function_application(
                program_point,
                variables,
                abstract_environments,
                worklist,
                res,
                res_variables,
                res_instance_id,
                successor,
                condition,
                res_cond,
                res_cond_variables,
                res_cond_instance_id,
            )

            included = abstract_environments[successor].includes(res_cond)

            listener.after_included(
                program_point,
                variables,
                abstract_environments,
                worklist,
                res,
                res_variables,
                res_instance_id,
                successor,
                condition,
                res_cond,
                res_cond_variables,
                res_cond_instance_id,
                included,
            )

            if not included:
                listener.after_not_included(
                    program_point,
                    variables,
                    abstract_environments,
                    worklist,
                    res,
                    res_variables,
                    res_instance_id,
                    successor,
                    condition,
                    res_cond,
                    res_cond_variables,
                    res_cond_instance_id,
                )

                for variable, joined_abstract_value in abstract_environments[
                    successor
                ].join_generator(res_cond):
                    current_abstract_value = res_cond[variable]
                    successor_abstract_value = abstract_environments[successor][
                        variable
                    ]

                    abstract_environments[successor] = abstract_environments[
                        successor
                    ].set({variable: joined_abstract_value})

                    listener.while_join(
                        program_point,
                        variables,
                        abstract_environments,
                        worklist,
                        res,
                        res_variables,
                        res_instance_id,
                        successor,
                        condition,
                        res_cond,
                        res_cond_variables,
                        res_cond_instance_id,
                        variable,
                        joined_abstract_value,
                        current_abstract_value,
                        successor_abstract_value,
                    )

                listener.after_join(
                    program_point,
                    variables,
                    abstract_environments,
                    worklist,
                    res,
                    res_variables,
                    res_instance_id,
                    successor,
                    condition,
                    res_cond,
                    res_cond_variables,
                    res_cond_instance_id,
                )

                worklist.add(successor)

                listener.after_add(
                    program_point,
                    variables,
                    abstract_environments,
                    worklist,
                    res,
                    res_variables,
                    res_instance_id,
                    successor,
                    condition,
                    res_cond,
                    res_cond_variables,
                    res_cond_instance_id,
                )

            listener.after_successor_iteration(
                program_point,
                variables,
                abstract_environments,
                worklist,
                res,
                res_variables,
                res_instance_id,
                successor,
                condition,
                res_cond,
                res_cond_variables,
                res_cond_instance_id,
            )

        listener.after_iteration(
            variables,
            abstract_environments,
            worklist,
            res,
            res_variables,
            res_instance_id,
        )

    listener.after_worklist_algorithm(
        variables,
        abstract_environments,
    )
