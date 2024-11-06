from __future__ import annotations

from manim.mobject.geometry.polygram import Rectangle
from manim.mobject.text.tex_mobject import MathTex, SingleStringMathTex, Tex
from manim.mobject.geometry.arc import TipableVMobject
from manim.mobject.geometry.line import Line
from manim.mobject.text.text_mobject import Text
from manim.mobject.mobject import Mobject
from manim.mobject.graph import DiGraph, LayoutName, LayoutFunction
from manim.utils.color import GREEN, RED, BLACK, WHITE
from typing import TYPE_CHECKING, Hashable
from collections import defaultdict
from dataclasses import dataclass
import networkx as nx
import numpy as np


if TYPE_CHECKING:
    from manim.mobject.geometry.tips import ArrowTip
    from typing_extensions import Self
    from manim_dataflow_analysis.ast import AstStatement
    from manim.mobject.graph import NxGraph
    from manim.typing import Point3D


class PathArrow(TipableVMobject):

    def __init__(
        self,
        *path: Point3D,
        **kwargs,
    ) -> None:
        self.path = path
        super().__init__(**kwargs)

    def generate_points(self) -> None:
        self.set_points_as_corners(np.array(self.path))

    def reset_endpoints_based_on_tip(self, tip: ArrowTip, at_start: bool) -> Self:
        return self

    def get_length(self) -> np.floating:
        length = 0
        for i in range(len(self.path) - 1):
            length += np.linalg.norm(
                np.array(self.path[i]) - np.array(self.path[i + 1])
            )
        return length


def __cfg_successors(
    graph: NxGraph,
    vertex: Hashable,
    condition_vertices: dict[Hashable, tuple[Hashable]],
) -> list[Hashable]:
    successors = list(nx.neighbors(graph, vertex))

    if len(successors) <= 1:
        return successors

    successors_condition = condition_vertices[vertex]

    if set(successors) != set(successors_condition):
        raise ValueError(
            f"The successors of the vertex {vertex} are not the same as the condition vertices"
        )

    return list(successors_condition)


def __cfg_node_depth(
    graph: NxGraph,
    vertex: Hashable,
    condition_vertices: dict[Hashable, tuple[Hashable]],
    done: set[Hashable],
    x: int = 0,
    y: int = 0,
    max_y: int = 0,
) -> tuple[int, int, int, dict[Hashable, int], dict[Hashable, tuple[int, int]]]:
    if vertex in done:
        return 1, 0, max_y, {vertex: y}, {}

    done.add(vertex)

    successors = __cfg_successors(graph, vertex, condition_vertices)

    current_max_y = max(y, max_y)

    if not successors:
        return 1, 1, current_max_y, {}, {vertex: (x, current_max_y)}

    if len(successors) == 1:
        (
            successor_width,
            successor_height,
            successor_max_y,
            successor_done_override,
            successors_coords,
        ) = __cfg_node_depth(
            graph,
            successors[0],
            condition_vertices,
            done,
            x,
            y + 1,
            current_max_y,
        )

        if successor_max_y > current_max_y:
            successors_coords[vertex] = (
                x,
                y + successor_max_y - current_max_y - successor_height,
            )
        else:
            successors_coords[vertex] = (x, y)

        return (
            successor_width,
            1 + successor_height,
            successor_max_y,
            successor_done_override,
            successors_coords,
        )

    current_width = 0
    current_height = 0

    done_override: dict[Hashable, int] = dict()
    coords: dict[Hashable, tuple[int, int]] = {}

    for successor in successors:
        (
            successor_width,
            successor_height,
            successor_max_y,
            successor_done_override,
            successors_coords,
        ) = __cfg_node_depth(
            graph,
            successor,
            condition_vertices,
            done,
            x + current_width,
            y + 1,
            current_max_y,
        )

        current_width += successor_width

        if successor_height > current_height:
            for coord_vertex, (coord_x, coord_y) in coords.items():
                coords[coord_vertex] = (
                    coord_x,
                    coord_y + successor_height - current_height,
                )
            current_height = successor_height

        if successor_max_y > current_max_y:
            for coord_vertex, (coord_x, coord_y) in coords.items():
                coords[coord_vertex] = (
                    coord_x,
                    coord_y + successor_max_y - current_max_y,
                )
            current_max_y = successor_max_y

        for vertex_done_override, y_done_override in successor_done_override.items():
            if vertex_done_override not in coords:
                done_override[vertex_done_override] = y_done_override
                continue

            _, vertex_done_override_coord_y = coords[vertex_done_override]

            if y_done_override <= vertex_done_override_coord_y:
                continue

            current_height += 1

            for coord_vertex, (coord_x, coord_y) in coords.items():
                if coord_y >= vertex_done_override_coord_y:
                    coords[coord_vertex] = (
                        coord_x,
                        y + current_height + coord_y - vertex_done_override_coord_y,
                    )

        coords.update(successors_coords)

    coords[vertex] = (x, y)

    return current_width, 1 + current_height, 0, done_override, coords


def cfg_layout(
    graph: NxGraph,
    root_vertex: Hashable,
    scale: float | tuple[float, float, float] = 1,
    condition_vertices: dict[Hashable, tuple[Hashable]] | None = None,
    vertex_spacing: tuple[float, float] = (1, 1),
) -> dict[Hashable, Point3D]:
    if condition_vertices is None:
        raise ValueError("The CFG layout requires the condition vertices to be passed")

    if isinstance(scale, float):
        scale_x = scale
        scale_y = scale
    elif isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x = scale_y = 1

    _, height, _, _, coords = __cfg_node_depth(
        graph,
        root_vertex,
        condition_vertices,
        set(),
    )

    space_x, space_y = vertex_spacing

    return {
        v: np.array([space_x * vx / scale_x, space_y * (height - 1 - vy) / scale_y, 0])
        for v, (vx, vy) in coords.items()
    }


class LabeledRectangle(Rectangle):

    def __init__(
        self,
        label: str | SingleStringMathTex | Text | Tex,
        height: float | None = None,
        width: float | None = None,
        **kwargs,
    ) -> None:
        if isinstance(label, str):
            rendered_label = MathTex(label, color=BLACK)
        else:
            rendered_label = label

        if height is None:
            height = rendered_label.height + 0.2

        if width is None:
            width = rendered_label.width + 0.2

        super().__init__(height=height, width=width, **kwargs)
        self.set_fill(color=WHITE, opacity=1)
        rendered_label.move_to(self.get_center())
        self.add(rendered_label)


@dataclass(frozen=True)
class ProgramPoint:
    point: int
    statement: AstStatement


class ControlFlowGraph(DiGraph):

    @classmethod
    def from_cfg(
        cls, entry_point: ProgramPoint, cfg: nx.DiGraph[ProgramPoint]
    ) -> ControlFlowGraph:
        labels = {pp: Text(pp.statement.header, color=BLACK) for pp in cfg}

        vertex_spacing = (
            1.25 * max(label.width for label in labels.values()),
            2.5 * max(label.height for label in labels.values()),
        )

        edge_cases: defaultdict[ProgramPoint, dict[ProgramPoint, int]] = defaultdict(
            dict
        )
        for start, end, case in cfg.edges.data("case"):
            if case is not None:
                edge_cases[start][end] = case

        condition_vertices = {
            start: tuple(sorted(ends, key=lambda end: ends[end]))
            for start, ends in edge_cases.items()
        }

        edge_config: dict[tuple[ProgramPoint, ProgramPoint], dict] = {}
        for start, ends in condition_vertices.items():
            for i, end in enumerate(ends):
                if i == 0:
                    color = RED
                elif i == 1:
                    color = GREEN
                else:
                    color = BLACK

                edge_config[(start, end)] = {"color": color}

        return cls.from_networkx(
            cfg,
            labels=labels,
            layout=cfg_layout,
            layout_config={
                "condition_vertices": condition_vertices,
                "vertex_spacing": vertex_spacing,
            },
            edge_config=edge_config,
            root_vertex=entry_point,
        )

    def __init__(
        self,
        vertices: list[Hashable],
        edges: list[tuple[Hashable, Hashable]],
        labels: bool | dict = True,
        label_fill_color: str = BLACK,
        layout: LayoutName | dict[Hashable, Point3D] | LayoutFunction = cfg_layout,
        layout_scale: float | tuple[float, float, float] = 2,
        layout_config: dict | None = None,
        vertex_type: type[Mobject] = LabeledRectangle,
        vertex_config: dict | None = None,
        vertex_mobjects: dict | None = None,
        edge_type: type[PathArrow] = PathArrow,
        partitions: list[list[Hashable]] | None = None,
        root_vertex: Hashable | None = None,
        edge_config: dict | None = None,
    ) -> None:
        super().__init__(
            vertices,
            edges,
            labels,
            label_fill_color,
            layout,
            layout_scale,
            layout_config,
            vertex_type,
            vertex_config,
            vertex_mobjects,
            edge_type,
            partitions,
            root_vertex,
            edge_config,
        )

    def _populate_edge_dict(
        self,
        edges: list[tuple[Hashable, Hashable]],
        edge_type: type[PathArrow],
    ):
        self.edges: dict[tuple[Hashable, Hashable], PathArrow] = {
            (u, v): self._create_edge_mobject(
                self[u],
                self[v],
                edge_type,
                self._edge_config[(u, v)],
            )
            for (u, v) in edges
        }

        for (u, v), edge in self.edges.items():
            edge.add_tip(**self._tip_config[(u, v)])

    def update_edges(self, graph: nx.DiGraph):
        edge: PathArrow
        for (u, v), edge in graph.edges.items():
            edge_type = type(edge)
            tip = edge.pop_tips()[0]
            new_edge = self._create_edge_mobject(
                self[u],
                self[v],
                edge_type,
                self._edge_config[(u, v)],
            )
            edge.become(new_edge)
            edge.add_tip(tip)

    def _add_edge(
        self,
        edge: tuple[Hashable, Hashable],
        edge_type: type[PathArrow] = PathArrow,
        edge_config: dict | None = None,
    ):
        # Some part of this method is copied from manim.mobject.types.graph.Graph._add_edge

        if edge_config is None:
            edge_config = self.default_edge_config.copy()
        added_mobjects = []
        for v in edge:
            if v not in self.vertices:
                added_mobjects.append(self._add_vertex(v))
        u, v = edge

        self._graph.add_edge(u, v)

        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(edge_config)
        edge_config = base_edge_config
        self._edge_config[(u, v)] = edge_config

        edge_mobject = self._create_edge_mobject(
            self[u],
            self[v],
            edge_type,
            edge_config,
        )
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def _create_edge_mobject(
        self,
        start_mobject: Mobject,
        end_mobject: Mobject,
        edge_type: type[PathArrow] = PathArrow,
        edge_config: dict | None = None,
    ):
        start_mobject_center_x, start_mobject_center_y, start_mobject_center_z = (
            start_mobject.get_center()
        )
        end_mobject_center_x, _, end_mobject_center_z = end_mobject.get_center()

        _, start_mobject_top_y, _ = start_mobject.get_top()
        _, start_mobject_bottom_y, _ = start_mobject.get_bottom()
        start_mobject_left_x, _, _ = start_mobject.get_left()
        start_mobject_right_x, _, _ = start_mobject.get_right()

        _, end_mobject_top_y, _ = end_mobject.get_top()
        _, end_mobject_bottom_y, _ = end_mobject.get_bottom()

        points: list[Point3D] = []
        end_y = end_mobject_top_y
        end_x = end_mobject_center_x
        start_y = start_mobject_bottom_y

        if end_mobject_center_x < start_mobject_left_x:
            start_x = start_mobject_center_x
            start_y = start_mobject_bottom_y
            if start_mobject_bottom_y > end_mobject_top_y:
                mid_point = start_y - self._dist_to_next_mobject_y(start_mobject) / 2
                points.append((start_x, mid_point, start_mobject_center_z))
                points.append((end_x, mid_point, end_mobject_center_z))
            elif end_mobject_bottom_y > start_mobject_top_y:
                bottom_point = start_y - self._dist_to_next_mobject_y(start_mobject) / 2
                right_point = self._same_right_block_width(start_mobject, end_mobject)
                top_point = self._dist_to_previous_mobject_y(end_mobject) / 2
                points.append((start_x, bottom_point, start_mobject_center_z))
                points.append((right_point, bottom_point, start_mobject_center_z))
                points.append((right_point, top_point, end_mobject_center_z))
                points.append((end_x, top_point, end_mobject_center_z))
        elif start_mobject_right_x < end_mobject_center_x:
            start_x = start_mobject_right_x
            start_y = start_mobject_center_y
            points.append((end_x, start_y, start_mobject_center_z))
        else:
            start_x = end_mobject_center_x

        start = (start_x, start_y, start_mobject_center_z)
        end = (end_x, end_y, end_mobject_center_z)

        return edge_type(start, *points, end, z_index=-1, **edge_config)

    def _same_right_block_width(
        self,
        start_mobject: Mobject,
        end_mobject: Mobject,
    ) -> float:
        return 1 + max(
            (
                vertex_mobject.get_right()[0]
                for vertex_mobject in self.vertices.values()
                if vertex_mobject.get_y() >= start_mobject.get_y()
                and vertex_mobject.get_y() < end_mobject.get_y()
                and vertex_mobject.get_x() >= start_mobject.get_x()
            ),
            default=start_mobject.get_right()[0],
        )

    def _next_mobject_y(self, mobject: Mobject) -> float:
        return min(
            (
                vertex_mobject.get_y()
                for vertex_mobject in self.vertices.values()
                if vertex_mobject.get_y() < mobject.get_y()
            ),
            default=1,
        )

    def _dist_to_next_mobject_y(self, mobject: Mobject) -> float:
        return abs(self._next_mobject_y(mobject) - mobject.get_y())

    def _previous_mobject_y(self, mobject: Mobject) -> float:
        return min(
            (
                vertex_mobject.get_y()
                for vertex_mobject in self.vertices.values()
                if vertex_mobject.get_y() > mobject.get_y()
            ),
            default=1,
        )

    def _dist_to_previous_mobject_y(self, mobject: Mobject) -> float:
        return abs(self._previous_mobject_y(mobject) - mobject.get_y())
