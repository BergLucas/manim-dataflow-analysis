from manim.mobject.table import MobjectTable
from typing import Collection, Mapping, Generic, TypeVar, Set
from manim_dataflow_analysis.cfg import ProgramPoint
from manim_dataflow_analysis.abstract_environment import AbstractEnvironment
from manim.mobject.text.tex_mobject import MathTex


L = TypeVar("L")


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
            variable: MathTex(str(res[variable]) if res is not None else "?")
            for variable in variables
        }
        self.res_cond_mobjects = {
            variable: MathTex(str(res_cond[variable]) if res_cond is not None else "?")
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
