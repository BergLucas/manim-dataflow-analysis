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

    def is_successor(self, value: L, successor: L) -> bool:
        return any(
            successor == value_successor for value_successor in self.successors(value)
        )

    def has_other_successors_than(self, value: L, *successors: L) -> bool:
        return any(
            value_successor not in successors
            for value_successor in self.successors(value)
        )

    def nearest_ancestor(self, value1: L, value2: L) -> L: ...

    def predecessors(self, value: L) -> Iterable[L]: ...

    def is_predecessor(self, value: L, predecessor: L) -> bool:
        return any(
            predecessor == value_predecessor
            for value_predecessor in self.predecessors(value)
        )

    def has_other_predecessors_than(self, value: L, *successors: L) -> bool:
        return any(
            value_successor not in successors
            for value_successor in self.predecessors(value)
        )

    def includes(self, including: L, included: L) -> bool: ...

    def join(self, value1: L, value2: L) -> L: ...

    def path(self, start: L, end: L) -> Iterable[L]: ...

    def path_length(self, start: L, end: L) -> float:
        return sum(
            (1.0 for _ in self.path(start, end)),
            start=0.0,
        )


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

    def path(self, start: L, end: L) -> Iterable[L]:
        return nx.shortest_path(self.__graph, start, end)


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

    @classmethod
    def from_lattice(
        cls,
        lattice: Lattice[L],
        visible_vertices: set[L] | None = None,
        max_horizontal_size_per_vertex: int = 4,
        max_vertical_size: int = 10,
        labels: bool | dict = True,
        labels_size: float = 0.5,
        label_fill_color: str = BLACK,
        layout_scale: float = 2,
        **kwargs,
    ) -> LatticeGraph[L]:
        half_bottom_vertical_size = math.floor(max_vertical_size / 2)
        half_top_vertical_size = math.ceil(max_vertical_size / 2)
        if visible_vertices is None:
            visible_vertices = set()

        vertices: set[L | InfiniteNode[L] | IncompleteNode[L]] = set()
        bottom_infinite_vertices: dict[InfiniteNode[L], set[L]] = {}
        top_infinite_vertices: dict[InfiniteNode[L], set[L]] = {}
        bottom_incomplete_vertices: dict[IncompleteNode, set[tuple[L, bool]]] = (
            defaultdict(set)
        )
        top_incomplete_vertices: dict[IncompleteNode, set[tuple[L, bool]]] = (
            defaultdict(set)
        )
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ] = set()

        worklist: list[tuple[L, bool, int]] = [
            (lattice.bottom(), False, 0),
            (lattice.top(), True, 0),
        ]

        while worklist:
            vertex, invert_direction, depth = worklist.pop(0)

            vertices.add(vertex)

            if invert_direction:
                incomplete_vertices = top_incomplete_vertices
                half_vertical_size = half_top_vertical_size
                infinite_vertices = top_infinite_vertices
                children_iterable = lattice.predecessors(vertex)
                children_visible_vertices = set(
                    visible_vertex
                    for visible_vertex in visible_vertices
                    if lattice.is_predecessor(vertex, visible_vertex)
                )
            else:
                incomplete_vertices = bottom_incomplete_vertices
                half_vertical_size = half_bottom_vertical_size
                infinite_vertices = bottom_infinite_vertices
                children_iterable = lattice.successors(vertex)
                children_visible_vertices = set(
                    visible_vertex
                    for visible_vertex in visible_vertices
                    if lattice.is_successor(vertex, visible_vertex)
                )

            children, is_finished = cls._take_max_horizontal_size(
                children_iterable,
                max_horizontal_size_per_vertex,
            )

            if children_visible_vertices:
                children.update(children_visible_vertices)

                if invert_direction:
                    is_finished = not lattice.has_other_predecessors_than(
                        vertex, *children
                    )
                else:
                    is_finished = not lattice.has_other_successors_than(
                        vertex, *children
                    )

            cls._create_incomplete_vertex(
                lattice,
                incomplete_vertices,
                edges,
                vertex,
                depth,
                invert_direction,
            )

            children_depth = depth + 1

            infinite_vertex_number = 0 if children else 1

            add_children = (
                children_depth + infinite_vertex_number < half_vertical_size - 1
            )

            if not add_children:
                cls._add_children_to_worklist(
                    children_visible_vertices,
                    vertices,
                    worklist,
                    edges,
                    vertex,
                    children_depth,
                    invert_direction,
                )
                if children and children != children_visible_vertices:
                    cls._add_infinite_vertices(
                        lattice,
                        infinite_vertices,
                        vertex,
                        invert_direction,
                    )

                continue

            cls._add_children_to_worklist(
                children,
                vertices,
                worklist,
                edges,
                vertex,
                children_depth,
                invert_direction,
            )

            if not is_finished:
                cls._create_incomplete_vertices(
                    incomplete_vertices,
                    vertex,
                    children_depth,
                    invert_direction,
                )

        unprocessed_visible_vertices = set(
            vertex for vertex in visible_vertices if vertex not in vertices
        )

        cls._add_unprocessed_vertices(
            lattice,
            bottom_infinite_vertices,
            vertices,
            edges,
            unprocessed_visible_vertices,
            False,
        )

        cls._add_unprocessed_vertices(
            lattice,
            top_infinite_vertices,
            vertices,
            edges,
            unprocessed_visible_vertices,
            True,
        )

        cls._create_infinite_edges(
            bottom_infinite_vertices,
            bottom_incomplete_vertices,
            half_bottom_vertical_size,
            vertices,
            edges,
            False,
        )

        cls._create_infinite_edges(
            top_infinite_vertices,
            top_incomplete_vertices,
            half_top_vertical_size,
            vertices,
            edges,
            True,
        )

        cls._create_incomplete_edges(
            lattice,
            bottom_incomplete_vertices,
            bottom_infinite_vertices,
            vertices,
            edges,
            False,
        )

        cls._create_incomplete_edges(
            lattice,
            top_incomplete_vertices,
            top_infinite_vertices,
            vertices,
            edges,
            True,
        )

        for bottom_infinite_vertex in bottom_infinite_vertices:
            for top_infinite_vertex in top_infinite_vertices:
                if lattice.includes(
                    top_infinite_vertex.base_node, bottom_infinite_vertex.base_node
                ):
                    edges.add((bottom_infinite_vertex, top_infinite_vertex))

        if isinstance(labels, bool) and labels:
            labels = {}
            for vertex in vertices:
                text = Text(str(vertex), fill_color=label_fill_color)
                text.scale(labels_size * layout_scale / max(text.width, text.height))
                labels[vertex] = text

        return cls(
            vertices,
            edges,
            labels,
            label_fill_color=label_fill_color,
            layout_scale=layout_scale,
            **kwargs,
        )

    @classmethod
    def _take_max_horizontal_size(
        cls,
        iterable: Iterable[L],
        max_horizontal_size_per_vertex: int,
    ) -> tuple[set[L], bool]:
        iterator = iter(iterable)
        children: set[L] = set()
        is_finished = False

        for _ in range(max_horizontal_size_per_vertex - 1):
            try:
                children.add(next(iterator))
            except StopIteration:
                is_finished = True
                break
        else:
            is_finished = next(iterator, None) is None

        return children, is_finished

    @classmethod
    def _create_incomplete_vertex(
        cls,
        lattice: Lattice[L],
        incomplete_vertices: dict[IncompleteNode, set[tuple[L, bool]]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ],
        vertex: L,
        depth: int,
        invert_direction: bool,
    ) -> None:
        if invert_direction:
            incomplete = lattice.has_other_successors_than(
                vertex,
                *(end for start, end in edges if start == vertex),
            )
        else:
            incomplete = lattice.has_other_predecessors_than(
                vertex,
                *(start for start, end in edges if end == vertex),
            )

        if not incomplete:
            return

        incomplete_vertex = IncompleteNode(depth - 1, invert_direction)

        max_incomplete_vertex = max(incomplete_vertices, default=None)

        incomplete_vertices[incomplete_vertex].add((vertex, not invert_direction))

        if (
            max_incomplete_vertex is not None
            and max_incomplete_vertex != incomplete_vertex
        ):
            incomplete_vertices[incomplete_vertex].add(
                (max_incomplete_vertex, invert_direction)
            )

    @classmethod
    def _add_infinite_vertices(
        cls,
        lattice: Lattice[L],
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        vertex: L,
        invert_direction: bool,
    ) -> bool:
        for infinite_vertex in infinite_vertices:
            if invert_direction:
                nearest_connection = lattice.nearest_ancestor(
                    vertex, infinite_vertex.base_node
                )
                if nearest_connection == lattice.bottom():
                    continue
            else:
                nearest_connection = lattice.nearest_descendant(
                    vertex, infinite_vertex.base_node
                )
                if nearest_connection == lattice.top():
                    continue

            connections = infinite_vertices.pop(infinite_vertex)
            connections.add(vertex)
            infinite_vertices[InfiniteNode(nearest_connection)] = connections
            break
        else:
            infinite_vertices[InfiniteNode(vertex)] = {vertex}

    @classmethod
    def _add_children_to_worklist(
        cls,
        children: set[L],
        vertices: set[L | InfiniteNode[L] | IncompleteNode[L]],
        worklist: list[tuple[L, bool, int]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ],
        vertex: L,
        children_depth: int,
        invert_direction: bool,
    ) -> None:
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

    @classmethod
    def _create_incomplete_vertices(
        cls,
        incomplete_vertices: dict[IncompleteNode, set[tuple[L, bool]]],
        vertex: L,
        children_depth: int,
        invert_direction: bool,
    ) -> None:
        incomplete_vertex = IncompleteNode(children_depth, invert_direction)

        max_incomplete_vertex = max(incomplete_vertices, default=None)

        incomplete_vertices[incomplete_vertex].add((vertex, invert_direction))

        if (
            max_incomplete_vertex is not None
            and max_incomplete_vertex != incomplete_vertex
        ):
            incomplete_vertices[incomplete_vertex].add(
                (max_incomplete_vertex, not invert_direction)
            )

    @classmethod
    def _add_unprocessed_vertices(
        cls,
        lattice: Lattice[L],
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        vertices: set[L | InfiniteNode[L] | IncompleteNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode[L],
                L | InfiniteNode[L] | IncompleteNode[L],
            ]
        ],
        unprocessed_visible_vertices: set[L],
        invert_direction: bool,
    ) -> None:
        for infinite_vertex, connections in infinite_vertices.items():
            for visible_vertex in unprocessed_visible_vertices:
                if invert_direction:
                    included = lattice.includes(
                        visible_vertex, infinite_vertex.base_node
                    )
                else:
                    included = lattice.includes(
                        infinite_vertex.base_node, visible_vertex
                    )

                if not included or infinite_vertex.base_node == visible_vertex:
                    continue

                infinite_visible_vertex = InfiniteNode(visible_vertex)

                vertices.update(
                    (
                        visible_vertex,
                        infinite_visible_vertex,
                    )
                )

                if invert_direction:
                    edges.update(
                        (
                            (infinite_vertex, visible_vertex),
                            (visible_vertex, infinite_visible_vertex),
                        )
                    )

                    visible_vertex_connections = set(
                        connection
                        for connection in connections
                        if lattice.includes(connection, visible_vertex)
                    )
                else:
                    edges.update(
                        (
                            (infinite_visible_vertex, visible_vertex),
                            (visible_vertex, infinite_vertex),
                        )
                    )

                    visible_vertex_connections = set(
                        connection
                        for connection in connections
                        if lattice.includes(visible_vertex, connection)
                    )

                connections.difference_update(visible_vertex_connections)

                for visible_vertex_connection in visible_vertex_connections:
                    if invert_direction:
                        edges.add((infinite_visible_vertex, visible_vertex_connection))
                    else:
                        edges.add((visible_vertex_connection, infinite_visible_vertex))

    @classmethod
    def _create_infinite_edges(
        cls,
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        incomplete_vertices: dict[IncompleteNode, set[tuple[L, bool]]],
        half_vertical_size: int,
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

            incomplete_vertex = IncompleteNode(half_vertical_size - 2, invert_direction)

            max_incomplete_vertex = max(incomplete_vertices, default=None)

            if (
                max_incomplete_vertex is not None
                and max_incomplete_vertex != incomplete_vertex
            ):
                incomplete_vertices[incomplete_vertex].update(
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

    @classmethod
    def _create_incomplete_edges(
        cls,
        lattice: Lattice[L],
        incomplete_vertices: dict[IncompleteNode, set[tuple[L, bool]]],
        infinite_vertices: dict[InfiniteNode[L], set[L]],
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
                    infinite_vertex = InfiniteNode(lattice.top())
                    vertices.add(infinite_vertex)
                    edges.add((infinite_vertex, incomplete_vertex))
                    infinite_vertices[infinite_vertex] = {lattice.top()}
            else:
                for connection_vertex, connection_invert_direction in connections:
                    if connection_invert_direction:
                        edges.add((incomplete_vertex, connection_vertex))
                    else:
                        edges.add((connection_vertex, incomplete_vertex))

                if all(start != incomplete_vertex for start, _ in edges):
                    infinite_vertex = InfiniteNode(lattice.bottom())
                    vertices.add(infinite_vertex)
                    edges.add((incomplete_vertex, infinite_vertex))
                    infinite_vertices[infinite_vertex] = {lattice.bottom()}

    def __init__(
        self,
        vertices: list[Hashable],
        edges: list[tuple[Hashable, Hashable]],
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
        self._edge_infinity_type = edge_infinity_type
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
