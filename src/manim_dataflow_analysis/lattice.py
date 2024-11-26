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

    def nearest_descendant(self, value1: L, value2: L) -> L: ...

    def successors(self, value: L) -> Iterable[L]: ...

    def has_other_successors_than(self, value: L, *others: L) -> bool:
        for successor in self.successors(value):
            if successor not in others:
                return True

        return False

    def nearest_ancestor(self, value1: L, value2: L) -> L: ...

    def predecessors(self, value: L) -> Iterable[L]: ...

    def has_other_predecessors_than(self, value: L, *others: L) -> bool:
        for predecessor in self.predecessors(value):
            if predecessor not in others:
                return True

        return False

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

    def nearest_descendant(self, value1: L, value2: L) -> L:
        descendants1 = nx.descendants(self.__graph, value1)
        descendants2 = nx.descendants(self.__graph, value2)

        min_distance = float("inf")
        nearest_descendant = None

        for descendant in descendants1 & descendants2:
            length1 = nx.shortest_path_length(self.__graph, value1, descendant)
            length2 = nx.shortest_path_length(self.__graph, value2, descendant)
            distance = length1 + length2

            if distance < min_distance:
                min_distance = distance
                nearest_descendant = descendant

        assert nearest_descendant is not None

        return nearest_descendant

    def successors(self, value: L) -> Iterable[L]:
        return self.__graph.successors(value)

    def nearest_ancestor(self, value1: L, value2: L) -> L:
        ancestors1 = nx.ancestors(self.__graph, value1)
        ancestors2 = nx.ancestors(self.__graph, value2)

        min_distance = float("inf")
        nearest_ancestor = None

        for ancestor in ancestors1 & ancestors2:
            length1 = nx.shortest_path_length(self.__graph, value1, ancestor)
            length2 = nx.shortest_path_length(self.__graph, value2, ancestor)
            distance = length1 + length2

            if distance < min_distance:
                min_distance = distance
                nearest_ancestor = ancestor

        assert nearest_ancestor is not None

        return nearest_ancestor

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


def default_sorting_function(elements: Iterable[Hashable]) -> list[Hashable]:
    return list(elements)


def lattice_layout(
    graph: NxGraph,
    scale: float | tuple[float, float, float] = 2,
    vertex_spacing: tuple[float, float] = (2.5, 2.5),
    sorting_function: Callable[
        [Iterable[Hashable]], list[Hashable]
    ] = default_sorting_function,
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
class InfiniteNode(Generic[L]):
    base_node: L

    def __str__(self) -> str:
        return "..."

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, InfiniteNode):
            return self.base_node < other.base_node
        elif isinstance(other, IncompleteNode):
            return False
        else:
            return False


@total_ordering
@dataclass(frozen=True, order=False)
class IncompleteNode(Generic[L]):
    depth: int
    invert_direction: bool

    def __str__(self) -> str:
        return "..."

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, IncompleteNode):
            return self.invert_direction < other.invert_direction or (
                self.invert_direction == other.invert_direction
                and self.depth < other.depth
            )
        elif isinstance(other, InfiniteNode):
            return True
        else:
            return False


class LatticeGraph(Generic[L], BetterDiGraph):

    def __init__(
        self,
        lattice: Lattice[L],
        max_horizontal_size_per_vertex: int = 4,
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
        self._max_horizontal_size_per_vertex = max_horizontal_size_per_vertex
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

        for _ in range(self._max_horizontal_size_per_vertex - 1):
            try:
                children.append(next(iterator))
            except StopIteration:
                is_finished = True
                break
        else:
            is_finished = next(iterator, None) is None

        return children, is_finished

    def _build_vertices_and_edges(
        self,
    ) -> tuple[
        set[L | InfiniteNode[L] | IncompleteNode[L]],
        set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ],
    ]:
        vertices: set[L | InfiniteNode[L] | IncompleteNode[L]] = set()
        bottom_infinite_vertices: dict[InfiniteNode[L], list[L]] = {}
        top_infinite_vertices: dict[InfiniteNode[L], list[L]] = {}
        bottom_incomplete_vertices: dict[IncompleteNode, list[tuple[L, bool]]] = (
            defaultdict(list)
        )
        top_incomplete_vertices: dict[IncompleteNode, list[tuple[L, bool]]] = (
            defaultdict(list)
        )
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ] = set()

        worklist: list[tuple[L, bool, int]] = [
            (self._lattice.top(), True, 0),
            (self._lattice.bottom(), False, 0),
        ]

        while worklist:
            vertex, invert_direction, depth = worklist.pop(0)

            if invert_direction:
                vertices.add(vertex)
                children, is_finished = self._take_max_horizontal_size(
                    self._lattice.predecessors(vertex)
                )
                if self._lattice.has_other_successors_than(
                    vertex, *(end for start, end in edges if start == vertex)
                ):
                    incomplete_vertex = IncompleteNode(depth - 1, invert_direction)

                    top_max_incomplete_vertex = max(
                        top_incomplete_vertices, default=None
                    )

                    top_incomplete_vertices[incomplete_vertex].append((vertex, True))

                    if (
                        top_max_incomplete_vertex is not None
                        and top_max_incomplete_vertex != incomplete_vertex
                    ):
                        top_incomplete_vertices[incomplete_vertex].append(
                            (top_max_incomplete_vertex, False)
                        )

            else:
                vertices.add(vertex)
                children, is_finished = self._take_max_horizontal_size(
                    self._lattice.successors(vertex)
                )
                if self._lattice.has_other_predecessors_than(
                    vertex, *(start for start, end in edges if end == vertex)
                ):
                    incomplete_vertex = IncompleteNode(depth - 1, invert_direction)

                    bottom_max_incomplete_vertex = max(
                        bottom_incomplete_vertices, default=None
                    )

                    bottom_incomplete_vertices[incomplete_vertex].append((vertex, True))

                    if (
                        bottom_max_incomplete_vertex is not None
                        and bottom_max_incomplete_vertex != incomplete_vertex
                    ):
                        bottom_incomplete_vertices[incomplete_vertex].append(
                            (bottom_max_incomplete_vertex, False)
                        )

            children_depth = depth + 1

            if children:
                should_add_incomplete_vertex = 0
            else:
                should_add_incomplete_vertex = 1

            if (
                invert_direction
                and children_depth + should_add_incomplete_vertex
                >= self._half_top_vertical_size - 1
            ) or (
                not invert_direction
                and children_depth + should_add_incomplete_vertex
                >= self._half_bottom_vertical_size - 1
            ):
                if children:
                    if invert_direction:
                        for infinite_vertex in top_infinite_vertices:
                            nearest_ancestor = self._lattice.nearest_ancestor(
                                vertex, infinite_vertex.base_node
                            )

                            if nearest_ancestor == self._lattice.bottom():
                                continue

                            top_children = top_infinite_vertices.pop(infinite_vertex)
                            top_children.append(vertex)
                            top_infinite_vertices[InfiniteNode(nearest_ancestor)] = (
                                top_children
                            )
                            break
                        else:
                            top_infinite_vertices[InfiniteNode(vertex)] = [vertex]
                    else:
                        for infinite_vertex in bottom_infinite_vertices:
                            nearest_descendant = self._lattice.nearest_descendant(
                                vertex, infinite_vertex.base_node
                            )

                            if nearest_descendant == self._lattice.top():
                                continue

                            bottom_parents = bottom_infinite_vertices.pop(
                                infinite_vertex
                            )
                            bottom_parents.append(vertex)
                            bottom_infinite_vertices[
                                InfiniteNode(nearest_descendant)
                            ] = bottom_parents
                            break
                        else:
                            bottom_infinite_vertices[InfiniteNode(vertex)] = [vertex]

                continue

            for child in children:
                if child not in vertices and all(
                    work_vertex != child for work_vertex, _, _ in worklist
                ):
                    worklist.append((child, invert_direction, children_depth))

                if invert_direction:
                    edge = (child, vertex)
                else:
                    edge = (vertex, child)

                edges.add(edge)

            if not is_finished:
                incomplete_vertex = IncompleteNode(children_depth, invert_direction)

                if invert_direction:
                    top_max_incomplete_vertex = max(
                        top_incomplete_vertices, default=None
                    )

                    top_incomplete_vertices[incomplete_vertex].append((vertex, False))

                    if (
                        top_max_incomplete_vertex is not None
                        and top_max_incomplete_vertex != incomplete_vertex
                    ):
                        top_incomplete_vertices[incomplete_vertex].append(
                            (top_max_incomplete_vertex, True)
                        )
                else:
                    bottom_max_incomplete_vertex = max(
                        bottom_incomplete_vertices, default=None
                    )

                    bottom_incomplete_vertices[incomplete_vertex].append(
                        (vertex, False)
                    )

                    if (
                        bottom_max_incomplete_vertex is not None
                        and bottom_max_incomplete_vertex != incomplete_vertex
                    ):
                        bottom_incomplete_vertices[incomplete_vertex].append(
                            (bottom_max_incomplete_vertex, True)
                        )

        self._handle_infinite_vertices(
            bottom_infinite_vertices,
            bottom_incomplete_vertices,
            vertices,
            edges,
            False,
        )

        self._handle_infinite_vertices(
            top_infinite_vertices,
            top_incomplete_vertices,
            vertices,
            edges,
            True,
        )

        self._handle_incomplete_vertices(
            bottom_incomplete_vertices,
            bottom_infinite_vertices,
            vertices,
            edges,
            False,
        )

        self._handle_incomplete_vertices(
            top_incomplete_vertices,
            top_infinite_vertices,
            vertices,
            edges,
            True,
        )

        for bottom_infinite_vertex in bottom_infinite_vertices:
            for top_infinite_vertex in top_infinite_vertices:
                if self._lattice.includes(
                    top_infinite_vertex.base_node, bottom_infinite_vertex.base_node
                ):
                    edges.add((bottom_infinite_vertex, top_infinite_vertex))

        return vertices, edges

    def _handle_infinite_vertices(
        self,
        infinite_vertices: dict[InfiniteNode[L], list[L]],
        incomplete_vertices: dict[IncompleteNode[L], list[tuple[L, bool]]],
        vertices: set[L | InfiniteNode[L] | IncompleteNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ],
        invert_direction: bool,
    ) -> None:
        for infinite_vertex, connections in infinite_vertices.items():
            vertices.add(infinite_vertex)

            if invert_direction:
                incomplete_vertex = IncompleteNode(
                    self._half_top_vertical_size - 2,
                    invert_direction,
                )
            else:
                incomplete_vertex = IncompleteNode(
                    self._half_bottom_vertical_size - 2,
                    invert_direction,
                )

            max_incomplete_vertex = max(incomplete_vertices, default=None)

            if (
                max_incomplete_vertex is not None
                and max_incomplete_vertex != incomplete_vertex
            ):
                incomplete_vertices[incomplete_vertex].extend(
                    (
                        (infinite_vertex, not invert_direction),
                        (max_incomplete_vertex, invert_direction),
                    )
                )

            for connection in connections:
                if invert_direction:
                    edges.add((infinite_vertex, connection))
                else:
                    edges.add((connection, infinite_vertex))

    def _handle_incomplete_vertices(
        self,
        incomplete_vertices: dict[IncompleteNode[L], list[tuple[L, bool]]],
        infinite_vertices: dict[InfiniteNode[L], list[L]],
        vertices: set[L | InfiniteNode[L] | IncompleteNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ],
        invert_direction: bool,
    ) -> None:
        for incomplete_vertex, connections in incomplete_vertices.items():
            vertices.add(incomplete_vertex)

            if invert_direction:
                for connection_vertex, connection_invert_direction in connections:
                    if connection_invert_direction:
                        edges.add((connection_vertex, incomplete_vertex))
                    else:
                        edges.add((incomplete_vertex, connection_vertex))

                if all(end != incomplete_vertex for _, end in edges):
                    infinite_vertex = InfiniteNode(self._lattice.top())
                    vertices.add(infinite_vertex)
                    edges.add((infinite_vertex, incomplete_vertex))
                    infinite_vertices[infinite_vertex] = [self._lattice.top()]
            else:
                for connection_vertex, connection_invert_direction in connections:
                    if connection_invert_direction:
                        edges.add((incomplete_vertex, connection_vertex))
                    else:
                        edges.add((connection_vertex, incomplete_vertex))

                if all(start != incomplete_vertex for start, _ in edges):
                    infinite_vertex = InfiniteNode(self._lattice.bottom())
                    vertices.add(infinite_vertex)
                    edges.add((incomplete_vertex, infinite_vertex))
                    infinite_vertices[infinite_vertex] = [self._lattice.bottom()]

    def _create_edge_mobject(
        self,
        start: Hashable,
        end: Hashable,
        edge_type: type[TipableVMobject],
        edge_config: dict,
    ):
        start_mobject: Mobject = self[start]
        end_mobject: Mobject = self[end]

        if isinstance(start, (InfiniteNode, IncompleteNode)) or isinstance(
            end, (InfiniteNode, IncompleteNode)
        ):
            edge_type = self._edge_infinity_type
            z_index = -1
        else:
            z_index = -2

        return edge_type(
            start_mobject.get_top(),
            end_mobject.get_bottom(),
            z_index=z_index,
            **edge_config,
        )
