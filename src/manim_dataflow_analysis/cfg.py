from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Protocol, cast

import networkx as nx
import numpy as np
from manim.mobject.geometry.arc import TipableVMobject
from manim.mobject.text.tex_mobject import SingleStringMathTex, Tex
from manim.utils.color import BLACK, GREEN, RED

from manim_dataflow_analysis.graph import BetterDiGraph, LabeledRectangle

if TYPE_CHECKING:
    from manim.mobject.geometry.tips import ArrowTip
    from manim.mobject.graph import LayoutFunction, LayoutName, NxGraph
    from manim.mobject.mobject import Mobject
    from manim.mobject.text.text_mobject import Text
    from manim.typing import Point3D
    from typing_extensions import Self

    from manim_dataflow_analysis.ast import AstStatement


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

    def get_unpositioned_tip(
        self,
        tip_shape: type[ArrowTip] | None = None,
        tip_length: float | None = None,
        tip_width: float | None = None,
    ):
        tip = super().get_unpositioned_tip(tip_shape, tip_length, tip_width)
        tip.z_index = self.z_index
        return tip

    def reset_endpoints_based_on_tip(self, tip: ArrowTip, at_start: bool) -> Self:
        return self

    def get_length(self) -> np.floating:
        length: np.floating = np.floating(0.0)
        for i in range(len(self.path) - 1):
            length += np.linalg.norm(
                np.array(self.path[i]) - np.array(self.path[i + 1])
            )
        return length


class GraphNode(Protocol):
    def get_center(self) -> Point3D: ...

    def get_top(self) -> Point3D: ...

    def get_bottom(self) -> Point3D: ...

    def get_right(self) -> Point3D: ...

    def get_left(self) -> Point3D: ...

    def get_zenith(self) -> Point3D: ...

    def get_nadir(self) -> Point3D: ...


class EdgeLayoutFunction(Protocol):
    def __call__(
        self,
        vertices: dict[Hashable, GraphNode],
        start: Hashable,
        end: Hashable,
    ) -> list[Point3D]: ...


def default_edge_layout(
    vertices: dict[Hashable, GraphNode],
    start: Hashable,
    end: Hashable,
) -> list[Point3D]:
    return [vertices[start].get_center(), vertices[end].get_center()]


class LayoutAndEdgeLayoutFunction(Protocol):
    def __call__(
        self,
        graph: NxGraph,
        scale: float | tuple[float, float, float] = 2,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[
        dict[Hashable, Point3D],
        EdgeLayoutFunction,
    ]: ...


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
            f"The successors of the vertex {vertex} are not the same as the condition vertices"  # noqa: E501
        )

    return list(successors_condition)


VERTEX_WIDTH = 1
VERTEX_HEIGHT = 1


def __cfg_node_depth(
    graph: NxGraph,
    vertex: Hashable,
    condition_vertices: dict[Hashable, tuple[Hashable]],
    done: set[Hashable],
    x: int = 0,
    y: int = 0,
) -> tuple[
    int,
    int,
    dict[Hashable, int],
    dict[Hashable, tuple[int, int]],
    dict[Hashable, tuple[int, int]],
]:
    if vertex in done:
        return 0, 0, {vertex: y}, {}, {}

    done.add(vertex)

    all_successors_width = 0
    all_successors_height = 0
    done_override: dict[Hashable, int] = {}
    coords: dict[Hashable, tuple[int, int]] = {}
    loops: dict[Hashable, tuple[int, int]] = {}

    for successor in __cfg_successors(graph, vertex, condition_vertices):
        (
            successors_width,
            successors_height,
            successors_done_override,
            successors_coords,
            successors_loops,
        ) = __cfg_node_depth(
            graph,
            successor,
            condition_vertices,
            done,
            x + all_successors_width,
            y + VERTEX_HEIGHT,
        )

        # Decrease the y positions of the already placed vertices if we found that the
        # height of the successors would go beyond the current height
        max_successors_done_override_in_coords = max(
            (
                coord_y
                for coord_vertex, coord_y in successors_done_override.items()
                if coord_vertex in coords
            ),
            default=None,
        )

        if max_successors_done_override_in_coords is not None:
            current_height = max_successors_done_override_in_coords - y
        elif vertex in successors_done_override:
            current_height = 0
        else:
            current_height = all_successors_height

        height_difference = successors_height - current_height
        if height_difference > 0:
            for coord_vertex, (coord_x, coord_y) in coords.items():
                coords[coord_vertex] = (coord_x, coord_y + height_difference)

            for coord_vertex, coord_y in done_override.items():
                done_override[coord_vertex] = coord_y + height_difference

            all_successors_height += height_difference

        # Decrease the y positions of the already placed vertices if we found that a
        # already placed vertex would be lower if it was placed after the successors
        for vertex_done_override, y_done_override in successors_done_override.items():
            if vertex_done_override == vertex:
                continue

            if vertex_done_override not in coords:
                done_override[vertex_done_override] = y_done_override
                continue

            _, vertex_done_override_coord_y = coords[vertex_done_override]

            y_done_override_difference = y_done_override - vertex_done_override_coord_y
            if y_done_override_difference <= 0:
                continue

            for coord_vertex, (coord_x, coord_y) in coords.items():
                if coord_y < vertex_done_override_coord_y:
                    continue

                coords[coord_vertex] = (
                    coord_x,
                    coord_y + y_done_override_difference,
                )

            for coord_vertex, coord_y in done_override.items():
                if coord_y < vertex_done_override_coord_y:
                    continue

                done_override[coord_vertex] = coord_y + y_done_override_difference

            all_successors_height += y_done_override_difference

        # Decrease the x positions of the already placed vertices if there is some
        # space next to the successors
        min_coord_y = y + current_height

        max_width_next_to_successors = max(
            (
                *(
                    coord_x - (x - VERTEX_WIDTH)
                    for coord_x, coord_y in coords.values()
                    if y < coord_y < min_coord_y
                ),
                *(
                    coord_x + (loop_width - VERTEX_WIDTH) - (x - VERTEX_WIDTH)
                    for coord_x, coord_y, loop_width in (
                        (*coords[coord_vertex], loop_width)
                        for coord_vertex, (loop_width, _) in loops.items()
                    )
                    if y < coord_y < min_coord_y
                ),
            ),
            default=all_successors_width,
        )

        width_difference = all_successors_width - max_width_next_to_successors

        if width_difference > 0:
            for coord_vertex, (coord_x, coord_y) in successors_coords.items():
                if coord_y <= y or min_coord_y <= coord_y:
                    continue

                successors_coords[coord_vertex] = (
                    coord_x - width_difference,
                    coord_y,
                )

            successors_width -= width_difference

        all_successors_width += successors_width

        # Increase the height when the vertex is a loop to improve readability
        for vertex_done_override in successors_done_override:
            if vertex_done_override != vertex:
                continue

            if successors_loops:
                all_successors_width += VERTEX_WIDTH

            loop_width, loop_height = loops.get(vertex, (0, 0))

            loops[vertex] = (
                max(VERTEX_WIDTH, all_successors_width, loop_width),
                max(VERTEX_HEIGHT + successors_height, loop_height),
            )

            if not coords:
                all_successors_height += VERTEX_HEIGHT

        # Update the coords and loops with the successors coords and loops
        coords.update(successors_coords)
        loops.update(successors_loops)

        # Increase the width when we have a fork that has no successors
        if not successors_coords:
            all_successors_width += VERTEX_WIDTH

    coords[vertex] = (x, y)

    return (
        max(VERTEX_WIDTH, all_successors_width),
        VERTEX_HEIGHT + all_successors_height,
        done_override,
        coords,
        loops,
    )


def cfg_layout(
    graph: NxGraph,
    root_vertex: Hashable,
    scale: float | tuple[float, float, float] = 2,
    condition_vertices: dict[Hashable, tuple[Hashable]] | None = None,
    vertex_spacing: tuple[float, float] = (1, 1),
) -> tuple[
    dict[Hashable, Point3D],
    EdgeLayoutFunction,
]:
    if condition_vertices is None:
        raise ValueError("The CFG layout requires the condition vertices to be passed")

    if isinstance(scale, float):
        scale_x = scale
        scale_y = scale
    elif isinstance(scale, tuple):
        scale_x, scale_y, _ = scale
    else:
        scale_x = scale_y = 1

    _, height, _, coords, loops = __cfg_node_depth(
        graph,
        root_vertex,
        condition_vertices,
        set(),
    )

    inverted_y_coords = {v: (vx, height - 1 - vy) for v, (vx, vy) in coords.items()}

    space_x, space_y = vertex_spacing

    def cfg_edge_layout(
        vertices: dict[Hashable, GraphNode],
        start: Hashable,
        end: Hashable,
    ) -> list[Point3D]:
        start_x, start_y = inverted_y_coords[start]
        end_x, end_y = inverted_y_coords[end]

        node_end_x, node_end_y, node_end_z = vertices[end].get_top()

        path: list[Point3D] = []

        coords_dist_x = abs(start_x - end_x)
        coords_dist_y = abs(start_y - end_y)
        vertices_dist_x = abs(
            vertices[start].get_center()[0] - vertices[end].get_center()[0]
        )
        vertices_dist_y = abs(
            vertices[start].get_center()[1] - vertices[end].get_center()[1]
        )

        if coords_dist_x == 0:
            vertices_x_scale = 1
        else:
            vertices_x_scale = vertices_dist_x / coords_dist_x

        if coords_dist_y == 0:
            vertices_y_scale = 1
        else:
            vertices_y_scale = vertices_dist_y / coords_dist_y

        if end_x < start_x:
            node_start_x, node_start_y, node_start_z = vertices[start].get_bottom()
            path.append(np.array([node_start_x, node_start_y, node_start_z]))

            if start_y > end_y:
                elbow_y = node_end_y + 0.25 * vertices_y_scale / scale_y

                path.extend(
                    (
                        np.array([node_start_x, elbow_y, node_start_z]),
                        np.array([node_end_x, elbow_y, node_end_z]),
                    )
                )
            elif end_y > start_y:
                loop_width, loop_height = loops[end]

                bottom_elbow_y = (
                    node_end_y - (loop_height - 0.25) * vertices_y_scale / scale_y
                )
                right_elbow_x = (
                    node_end_x + (loop_width - 0.5) * vertices_x_scale / scale_x
                )
                top_elbow_y = node_end_y + 0.25 * vertices_y_scale / scale_y
                path.extend(
                    (
                        np.array([node_start_x, bottom_elbow_y, node_start_z]),
                        np.array([right_elbow_x, bottom_elbow_y, node_start_z]),
                        np.array([right_elbow_x, top_elbow_y, node_end_z]),
                        np.array([node_end_x, top_elbow_y, node_end_z]),
                    )
                )
        elif start_x < end_x:
            node_start_x, node_start_y, node_start_z = vertices[start].get_right()
            path.extend(
                (
                    np.array([node_start_x, node_start_y, node_start_z]),
                    np.array([node_end_x, node_start_y, node_end_z]),
                )
            )
        else:
            path.append(vertices[start].get_bottom())

        path.append(np.array([node_end_x, node_end_y, node_end_z]))

        return path

    return (
        {
            v: np.array([space_x * vx / scale_x, space_y * vy / scale_y, 0])
            for v, (vx, vy) in inverted_y_coords.items()
        },
        cfg_edge_layout,
    )


@dataclass(frozen=True)
class ProgramPoint:
    point: int
    statement: AstStatement


class ControlFlowGraph(BetterDiGraph):
    @classmethod
    def from_cfg(
        cls, entry_point: ProgramPoint, cfg: nx.DiGraph[ProgramPoint]
    ) -> ControlFlowGraph:
        labels = {
            pp: Tex(
                r"\fbox{%s} \texttt{%s}" % (pp.point, pp.statement.header), color=BLACK
            )
            for pp in cfg
        }

        vertex_spacing = (
            1.25 * max(label.width for label in labels.values()),
            2.5 * max(label.height for label in labels.values()),
        )

        edge_cases: defaultdict[ProgramPoint, dict[ProgramPoint, int]] = defaultdict(
            dict
        )
        case: int
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
                    edge_config[(start, end)] = {"color": RED}
                elif i == 1:
                    edge_config[(start, end)] = {"color": GREEN}

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
        labels: bool | dict[Hashable, str | SingleStringMathTex | Text | Tex] = True,
        label_fill_color: str = BLACK,
        layout: (
            LayoutName
            | dict[Hashable, Point3D]
            | LayoutFunction
            | LayoutAndEdgeLayoutFunction
        ) = cfg_layout,
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

    def _create_edge_mobject(
        self,
        start: Hashable,
        end: Hashable,
        edge_type: type[PathArrow],
        edge_config: dict,
    ):
        if edge_config is not None and "color" in edge_config:
            z_index = -1
        else:
            z_index = -2

        return edge_type(
            *self._edge_layout(self.vertices, start, end),
            z_index=z_index,
            **edge_config,
        )

    def change_layout(
        self,
        layout: (
            LayoutName
            | dict[Hashable, Point3D]
            | LayoutFunction
            | LayoutAndEdgeLayoutFunction
        ) = "spring",
        layout_scale: float | tuple[float, float, float] = 2,
        layout_config: dict[str, Any] | None = None,
        partitions: list[list[Hashable]] | None = None,
        root_vertex: Hashable | None = None,
    ) -> ControlFlowGraph:
        try:
            super().change_layout(
                layout,
                layout_scale,
                layout_config,
                partitions,
                root_vertex,
            )
            self._edge_layout = default_edge_layout
        except (TypeError, ValueError) as e:
            layout_config = {} if layout_config is None else layout_config
            if partitions is not None and "partitions" not in layout_config:
                layout_config["partitions"] = partitions
            if root_vertex is not None and "root_vertex" not in layout_config:
                layout_config["root_vertex"] = root_vertex

            try:
                self._layout, self._edge_layout = cast(
                    LayoutAndEdgeLayoutFunction, layout
                )(self._graph, scale=layout_scale, **layout_config)
            except TypeError as te:
                raise e from te

            for v in self.vertices:
                self[v].move_to(self._layout[v])

        return self


def succ(
    graph: nx.DiGraph[ProgramPoint], program_point: ProgramPoint
) -> Iterable[ProgramPoint]:
    return graph.successors(program_point)


def pred(
    graph: nx.DiGraph[ProgramPoint], program_point: ProgramPoint
) -> Iterable[ProgramPoint]:
    return graph.predecessors(program_point)


def cond(
    graph: nx.DiGraph[ProgramPoint],
    start_program_point: ProgramPoint,
    end_program_point: ProgramPoint,
) -> Any:
    return graph.get_edge_data(start_program_point, end_program_point)["condition"]
