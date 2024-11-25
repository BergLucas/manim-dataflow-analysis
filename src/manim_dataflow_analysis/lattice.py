from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Hashable,
    Protocol,
    TypeVar,
    Generic,
    Iterable,
    Callable,
    Any,
)
from manim_dataflow_analysis.graph import BetterDiGraph
from manim.mobject.geometry.arc import TipableVMobject, LabeledDot
from manim.mobject.graph import LayoutName, LayoutFunction
from manim.mobject.text.text_mobject import Text
from manim.mobject.geometry.line import Line, DashedLine
from manim.mobject.mobject import Mobject
from manim.utils.color import BLACK
from collections import defaultdict
from dataclasses import dataclass
from functools import total_ordering
import networkx as nx
import numpy as np
import math


if TYPE_CHECKING:
    from manim.mobject.graph import NxGraph
    from manim.typing import Point3D


L = TypeVar("L")


class Lattice(Protocol[L]):
    def top(self) -> L: ...

    def bottom(self) -> L: ...

    def successors(self, value: L) -> Iterable[L]: ...

    def predecessors(self, value: L) -> Iterable[L]: ...

    def includes(self, including: L, included: L) -> bool: ...

    def join(self, value1: L, value2: L) -> L: ...


class FiniteSizeLattice(Lattice[L]):

    def __init__(self, *edges: tuple[L, L]):
        self.__graph: nx.DiGraph[L] = nx.DiGraph(edges)

    def top(self) -> L:
        top = [
            node for node in self.__graph.nodes if self.__graph.out_degree(node) == 0
        ]

        if len(top) != 1:
            raise ValueError("Lattice has more than one top element")

        return top[0]

    def bottom(self) -> L:
        bottom = [
            node for node in self.__graph.nodes if self.__graph.in_degree(node) == 0
        ]

        if len(bottom) != 1:
            raise ValueError("Lattice has more than one bottom element")

        return bottom[0]

    def successors(self, value: L) -> Iterable[L]:
        return self.__graph.successors(value)

    def predecessors(self, value: L) -> Iterable[L]:
        return self.__graph.predecessors(value)

    def includes(self, including: L, included: L) -> bool:
        return including == included or nx.has_path(self.__graph, included, including)

    def join(self, value1: L, value2: L) -> L:
        joined_value: L | None = nx.lowest_common_ancestor(self.__graph, value1, value2)

        if joined_value is None:
            raise ValueError("Lattice has no join value")

        return joined_value


def __get_vertices_heights(
    graph: NxGraph, vertex: Hashable, height: int = 0
) -> dict[Hashable, int]:
    vertices_heights: dict[Hashable, int] = {vertex: height}

    for successor in graph.successors(vertex):
        for successor_vertex, successor_height in __get_vertices_heights(
            graph, successor, height + 1
        ).items():
            current_height = vertices_heights.get(successor_vertex)

            if current_height is None or current_height < successor_height:
                vertices_heights[successor_vertex] = successor_height

    return vertices_heights


def lattice_layout(
    graph: NxGraph,
    scale: float | tuple[float, float, float] = 2,
    vertex_spacing: tuple[float, float] = (2.5, 2.5),
    sorting_function: Callable[[Iterable[Hashable]], list[Hashable]] = lambda it: it,
) -> dict[Hashable, Point3D]:
    if isinstance(scale, float):
        scale_x = scale
        scale_y = scale
    elif isinstance(scale, tuple):
        scale_x, scale_y = scale
    else:
        scale_x = scale_y = 1

    vertices_heights: dict[Hashable, int] = dict()

    for vertex, degree in graph.in_degree():
        if degree != 0:
            continue

        for successor_vertex, successor_height in __get_vertices_heights(
            graph, vertex
        ).items():
            current_height = vertices_heights.get(successor_vertex)

            if current_height is None or current_height < successor_height:
                vertices_heights[successor_vertex] = successor_height

    if not vertices_heights:
        raise ValueError("There is no bottom value in the graph")

    heights_layer: dict[int, list[Hashable]] = defaultdict(list)
    for vertex, height in vertices_heights.items():
        heights_layer[height].append(vertex)

    coords: dict[Hashable, (int, int)] = {}
    for height, vertices in heights_layer.items():
        for i, vertex in enumerate(sorting_function(vertices)):
            coords[vertex] = (i - len(vertices) / 2, height)

    space_x, space_y = vertex_spacing

    return {
        v: np.array([space_x * vx / scale_x, space_y * vy / scale_y, 0])
        for v, (vx, vy) in coords.items()
    }


@total_ordering
@dataclass(frozen=True, order=False)
class LatticeNode(Generic[L]):
    parent: L
    invert_direction: bool

    def __str__(self) -> str:
        return "..."

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, LatticeNode):
            return self.parent < other.parent
        else:
            return False


class LatticeGraph(Generic[L], BetterDiGraph):

    def __init__(
        self,
        lattice: Lattice[L],
        max_horizontal_size: int = 10,
        max_vertical_size: int = 10,
        labels_size: float = 0.5,
        labels: bool | dict = True,
        label_fill_color: str = BLACK,
        layout: LayoutName | dict[Hashable, Point3D] | LayoutFunction = lattice_layout,
        layout_scale: float | tuple[float, float, float] = 2,
        layout_config: dict | None = None,
        vertex_type: type[Mobject] = LabeledDot,
        vertex_config: dict | None = None,
        vertex_mobjects: dict | None = None,
        edge_type: type[TipableVMobject] = Line,
        edge_infinity_type: type[TipableVMobject] = DashedLine,
        partitions: list[list[Hashable]] | None = None,
        root_vertex: Hashable | None = None,
        edge_config: dict | None = None,
    ) -> None:
        self._lattice = lattice
        self._max_horizontal_size = max_horizontal_size
        self._half_bottom_vertical_size = math.floor(max_vertical_size / 2)
        self._half_top_vertical_size = math.ceil(max_vertical_size / 2)
        self._edge_infinity_type = edge_infinity_type

        vertices, edges = self._build_vertices_and_edges()

        if isinstance(labels, bool) and labels:
            labels = {}
            for vertex in vertices:
                text = Text(str(vertex), fill_color=label_fill_color)
                text.scale(labels_size * layout_scale / max(text.width, text.height))
                labels[vertex] = text

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

    def _take_max_horizontal_size(self, iterable: Iterable[L]) -> tuple[list[L], bool]:
        iterator = iter(iterable)
        children: list[L] = []
        is_finished = False

        for _ in range(self._max_horizontal_size - 1):
            try:
                children.append(next(iterator))
            except StopIteration:
                is_finished = True
                break
        else:
            child = next(iterator, None)

            is_finished = child is None

            if is_finished:
                children.append(child)

        return children, is_finished

    def _build_vertices_and_edges(
        self,
    ) -> tuple[
        set[L | LatticeNode[L]], set[tuple[L | LatticeNode[L], L | LatticeNode[L]]]
    ]:
        vertices: set[L | LatticeNode[L]] = set()
        edges: set[tuple[L | LatticeNode[L], L | LatticeNode[L]]] = set()

        worklist: list[L | LatticeNode[L], bool, int] = [
            (self._lattice.top(), True, 0),
            (self._lattice.bottom(), False, 0),
        ]
        incomplete_vertices: list[LatticeNode[L]] = []

        while worklist:
            vertex, invert_direction, size = worklist.pop()

            vertices.add(vertex)

            if invert_direction:
                children, is_finished = self._take_max_horizontal_size(
                    self._lattice.predecessors(vertex)
                )
            else:
                children, is_finished = self._take_max_horizontal_size(
                    self._lattice.successors(vertex)
                )

            children_size = size + 1

            if not is_finished:
                infinite_vertex = LatticeNode(vertex, invert_direction)
                vertices.add(infinite_vertex)

                if invert_direction:
                    edges.add((infinite_vertex, vertex))
                else:
                    edges.add((vertex, infinite_vertex))

            if children:
                should_add_incomplete_vertex = 0
            else:
                should_add_incomplete_vertex = 1

            if (
                invert_direction
                and children_size + should_add_incomplete_vertex
                >= self._half_top_vertical_size - 1
            ) or (
                not invert_direction
                and children_size + should_add_incomplete_vertex
                >= self._half_bottom_vertical_size - 1
            ):
                if children:
                    incomplete_vertices.append(LatticeNode(vertex, invert_direction))
                continue

            for child in children:
                if child not in vertices:
                    worklist.append((child, invert_direction, children_size))

                if invert_direction:
                    edge = (child, vertex)
                else:
                    edge = (vertex, child)

                edges.add(edge)

        for lattice_node in incomplete_vertices:
            vertices.add(lattice_node)
            if lattice_node.invert_direction:
                edges.add((lattice_node, lattice_node.parent))
            else:
                edges.add((lattice_node.parent, lattice_node))

        lattice_vertices: list[LatticeNode[L]] = []
        start_border_vertices: list[L] = []
        end_border_vertices: list[L] = []
        for vertex in vertices:
            if isinstance(vertex, LatticeNode):
                lattice_vertices.append(vertex)

            if not any(vertex == end for _, end in edges):
                start_border_vertices.append(vertex)
            elif not any(vertex == start for start, _ in edges):
                end_border_vertices.append(vertex)

        for lattice_vertex in lattice_vertices:
            vertices.add(lattice_vertex)
            found = False
            if lattice_vertex.invert_direction:
                for end in end_border_vertices:
                    if isinstance(end, LatticeNode):
                        end_vertex = end.parent
                    else:
                        end_vertex = end

                    if (
                        self._lattice.includes(lattice_vertex.parent, end_vertex)
                        and lattice_vertex.parent != end_vertex
                    ):
                        found = True
                        edges.add((end, lattice_vertex))

                if not found:
                    edges.add((self._lattice.bottom(), lattice_vertex))
            else:
                for start in start_border_vertices:
                    if isinstance(start, LatticeNode):
                        start_vertex = start.parent
                    else:
                        start_vertex = start

                    if (
                        self._lattice.includes(start_vertex, lattice_vertex.parent)
                        and start_vertex != lattice_vertex.parent
                    ):
                        found = True
                        edges.add((lattice_vertex, start))

                if not found:
                    edges.add((lattice_vertex, self._lattice.top()))

        return vertices, edges

    def _create_edge_mobject(
        self,
        start: Hashable,
        end: Hashable,
        edge_type: type[TipableVMobject],
        edge_config: dict,
    ):
        start_mobject: Mobject = self[start]
        end_mobject: Mobject = self[end]

        if isinstance(start, LatticeNode) and isinstance(end, LatticeNode):
            edge_type = self._edge_infinity_type
            stroke_opacity = 0.5
            tip_style = dict(fill_opacity=stroke_opacity)
            z_index = -1
        else:
            stroke_opacity = 1
            tip_style = {}
            z_index = -2

        return edge_type(
            start_mobject.get_top(),
            end_mobject.get_bottom(),
            z_index=z_index,
            stroke_opacity=stroke_opacity,
            tip_style=tip_style,
            **edge_config,
        )
