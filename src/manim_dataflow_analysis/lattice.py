from __future__ import annotations

import math
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import total_ordering
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Protocol,
    TypeVar,
)

import networkx as nx
import numpy as np
from manim.mobject.geometry.arc import LabeledDot, TipableVMobject
from manim.mobject.geometry.line import DashedLine, Line
from manim.mobject.text.text_mobject import Text
from manim.utils.color import BLACK, GREEN, WHITE, ManimColor

from manim_dataflow_analysis.graph import BetterDiGraph

if TYPE_CHECKING:
    from manim.mobject.graph import LayoutFunction, LayoutName, NxGraph
    from manim.mobject.mobject import Mobject
    from manim.typing import Point3D


L = TypeVar("L")


class Lattice(Protocol[L]):
    @abstractmethod
    def top(self) -> L: ...

    @abstractmethod
    def bottom(self) -> L: ...

    @abstractmethod
    def successors(self, value: L) -> Iterable[L]: ...

    def is_successor(self, value: L, successor: L) -> bool:
        return any(
            successor == value_successor for value_successor in self.successors(value)
        )

    def has_other_successors_than(self, value: L, successors: set[L]) -> bool:
        return any(
            value_successor not in successors
            for value_successor in self.successors(value)
        )

    def descendants(self, value: L) -> Iterable[L]:
        for successor in self.successors(value):
            yield successor
            yield from self.descendants(successor)

    def is_descendant(self, value: L, descendant: L) -> bool:
        return any(
            descendant == value_descendant
            for value_descendant in self.descendants(value)
        )

    @abstractmethod
    def predecessors(self, value: L) -> Iterable[L]: ...

    def is_predecessor(self, value: L, predecessor: L) -> bool:
        return any(
            predecessor == value_predecessor
            for value_predecessor in self.predecessors(value)
        )

    def has_other_predecessors_than(self, value: L, predecessors: set[L]) -> bool:
        return any(
            value_predecessor not in predecessors
            for value_predecessor in self.predecessors(value)
        )

    def ancestors(self, value: L) -> Iterable[L]:
        for predecessor in self.predecessors(value):
            yield predecessor
            yield from self.ancestors(predecessor)

    def is_ancestor(self, value: L, ancestor: L) -> bool:
        return any(
            ancestor == value_ancestor for value_ancestor in self.ancestors(value)
        )

    def contains(self, containing: L, contained: L) -> bool:
        return all(
            self.is_descendant(successor, containing)
            for successor in self.successors(contained)
        )

    def reverse_contains(self, containing: L, contained: L) -> bool:
        return all(
            self.is_ancestor(predecessor, containing)
            for predecessor in self.predecessors(contained)
        )

    def includes(self, including: L, included: L) -> bool:
        return including == included or self.is_descendant(included, including)

    def reverse_includes(self, including: L, included: L) -> bool:
        return including == included or self.is_ancestor(included, including)

    def join(self, value1: L, value2: L) -> L:
        return self.__join_distance(value1, value2)[0]

    def __join_distance(self, value1: L, value2: L) -> tuple[L, float]:
        if value1 == value2:
            return value1, 0.0

        min_distance = float("inf")
        closest_joined_value = self.top()

        for successor in self.successors(value1):
            joined_value, distance = self.__join_distance(successor, value2)

            if distance < min_distance:
                min_distance = distance
                closest_joined_value = joined_value

        for successor in self.successors(value2):
            joined_value, distance = self.__join_distance(value1, successor)

            if distance < min_distance:
                min_distance = distance
                closest_joined_value = joined_value

        return closest_joined_value, min_distance + 1.0

    def meet(self, value1: L, value2: L) -> L:
        return self.__meet_distance(value1, value2)[0]

    def __meet_distance(self, value1: L, value2: L) -> tuple[L, float]:
        if value1 == value2:
            return value1, 0.0

        min_distance = float("inf")
        closest_met_value = self.bottom()

        for predecessor in self.predecessors(value1):
            met_value, distance = self.__meet_distance(predecessor, value2)

            if distance < min_distance:
                min_distance = distance
                closest_met_value = met_value

        for predecessor in self.predecessors(value2):
            met_value, distance = self.__meet_distance(value1, predecessor)

            if distance < min_distance:
                min_distance = distance
                closest_met_value = met_value

        return closest_met_value, min_distance + 1.0


class FiniteSizeLattice(Lattice[L]):
    def __init__(self, *edges: tuple[L, L]):
        self.__graph: nx.DiGraph[L] = nx.DiGraph(edges)

    def top(self) -> L:
        top = tuple(
            node for node in self.__graph.nodes if self.__graph.out_degree(node) == 0
        )

        if len(top) != 1:
            raise ValueError("Lattice has more than one top element")

        return top[0]

    def bottom(self) -> L:
        bottom = tuple(
            node for node in self.__graph.nodes if self.__graph.in_degree(node) == 0
        )

        if len(bottom) != 1:
            raise ValueError("Lattice has more than one bottom element")

        return bottom[0]

    def successors(self, value: L) -> Iterable[L]:
        return self.__graph.successors(value)

    def is_successor(self, value: L, successor: L) -> bool:
        return self.__graph.has_edge(value, successor)

    def descendants(self, value: L) -> Iterable[L]:
        return nx.descendants(self.__graph, value)

    def is_descendant(self, value: L, descendant: L) -> bool:
        return nx.has_path(self.__graph, value, descendant)

    def predecessors(self, value: L) -> Iterable[L]:
        return self.__graph.predecessors(value)

    def is_predecessor(self, value: L, predecessor: L) -> bool:
        return self.__graph.has_edge(predecessor, value)

    def ancestors(self, value: L) -> Iterable[L]:
        return nx.ancestors(self.__graph, value)

    def is_ancestor(self, value: L, ancestor: L) -> bool:
        return nx.has_path(self.__graph, ancestor, value)

    def join(self, value1: L, value2: L) -> L:
        joined_value: L | None = nx.lowest_common_ancestor(
            self.__graph.reverse(copy=False), value1, value2
        )

        if joined_value is None:
            joined_value = self.top()

        return joined_value

    def meet(self, value1: L, value2: L) -> L:
        met_value: L | None = nx.lowest_common_ancestor(self.__graph, value1, value2)

        if met_value is None:
            met_value = self.bottom()

        return met_value


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
        scale_x, scale_y, _ = scale
    else:
        scale_x = scale_y = 1

    vertices_heights: dict[Hashable, int] = {}

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

    coords: dict[Hashable, tuple[float, float]] = {}
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
        else:
            return False


@total_ordering
@dataclass(frozen=True, order=False)
class IncompleteNode:
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
        else:
            return isinstance(other, InfiniteNode)


@total_ordering
@dataclass(frozen=True, order=False)
class BridgeNode(Generic[L]):
    bottom_vertex: InfiniteNode[L]
    top_vertex: InfiniteNode[L]

    def __str__(self) -> str:
        return "..."

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, BridgeNode):
            return self.bottom_vertex < other.bottom_vertex or (
                self.bottom_vertex == other.bottom_vertex
                and self.top_vertex < other.top_vertex
            )
        else:
            return isinstance(other, (InfiniteNode, IncompleteNode))


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

        vertices: set[L | InfiniteNode[L] | IncompleteNode | BridgeNode[L]] = set()
        bottom_infinite_vertices: dict[InfiniteNode[L], set[L]] = {}
        top_infinite_vertices: dict[InfiniteNode[L], set[L]] = {}
        bottom_incomplete_vertices: dict[
            IncompleteNode, set[tuple[L | InfiniteNode[L] | IncompleteNode, bool]]
        ] = defaultdict(set)
        top_incomplete_vertices: dict[
            IncompleteNode, set[tuple[L | InfiniteNode[L] | IncompleteNode, bool]]
        ] = defaultdict(set)
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
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
                infinite_connections = cls._fix_infinite_vertices(
                    bottom_infinite_vertices,
                    edges,
                    vertex,
                    invert_direction,
                )
                all_children_done = not lattice.has_other_predecessors_than(
                    vertex,
                    infinite_connections,
                )
                incomplete_vertices = top_incomplete_vertices
                half_vertical_size = half_top_vertical_size
                infinite_vertices = top_infinite_vertices
                children_iterable = lattice.predecessors(vertex)
                children_visible_vertices = {
                    visible_vertex
                    for visible_vertex in visible_vertices
                    if lattice.is_predecessor(vertex, visible_vertex)
                }
            else:
                infinite_connections = cls._fix_infinite_vertices(
                    top_infinite_vertices,
                    edges,
                    vertex,
                    invert_direction,
                )
                all_children_done = not lattice.has_other_successors_than(
                    vertex,
                    infinite_connections,
                )
                incomplete_vertices = bottom_incomplete_vertices
                half_vertical_size = half_bottom_vertical_size
                infinite_vertices = bottom_infinite_vertices
                children_iterable = lattice.successors(vertex)
                children_visible_vertices = {
                    visible_vertex
                    for visible_vertex in visible_vertices
                    if lattice.is_successor(vertex, visible_vertex)
                }

            if all_children_done:
                continue

            children, is_finished = cls._take_max_horizontal_size(
                children_iterable,
                max_horizontal_size_per_vertex,
            )

            if children_visible_vertices:
                children.update(children_visible_vertices)

                if invert_direction:
                    is_finished = not lattice.has_other_predecessors_than(
                        vertex, children
                    )
                else:
                    is_finished = not lattice.has_other_successors_than(
                        vertex, children
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

        unprocessed_visible_vertices = {
            vertex for vertex in visible_vertices if vertex not in vertices
        }

        cls._create_unprocessed_vertices(
            lattice,
            bottom_infinite_vertices,
            vertices,
            edges,
            unprocessed_visible_vertices,
            False,
        )

        cls._create_unprocessed_vertices(
            lattice,
            top_infinite_vertices,
            vertices,
            edges,
            unprocessed_visible_vertices,
            True,
        )

        cls._create_incomplete_edges(
            bottom_infinite_vertices,
            bottom_incomplete_vertices,
            half_bottom_vertical_size,
            vertices,
            edges,
            False,
        )

        cls._create_incomplete_edges(
            top_infinite_vertices,
            top_incomplete_vertices,
            half_top_vertical_size,
            vertices,
            edges,
            True,
        )

        cls._create_infinite_edges(
            lattice,
            bottom_incomplete_vertices,
            bottom_infinite_vertices,
            vertices,
            edges,
            False,
        )

        cls._create_infinite_edges(
            lattice,
            top_incomplete_vertices,
            top_infinite_vertices,
            vertices,
            edges,
            True,
        )

        final_visible_vertices = {
            vertex for vertex in unprocessed_visible_vertices if vertex not in vertices
        }

        cls._bridge_infinite_vertices(
            lattice,
            bottom_infinite_vertices,
            top_infinite_vertices,
            vertices,
            edges,
            final_visible_vertices,
        )

        if isinstance(labels, bool) and labels:
            labels = {}
            for label_vertex in vertices:
                text = Text(str(label_vertex), fill_color=label_fill_color)
                text.scale(labels_size * layout_scale / max(text.width, text.height))
                labels[label_vertex] = text

        return cls(
            list(vertices),
            list(edges),
            labels,
            label_fill_color=label_fill_color,
            layout_scale=layout_scale,
            **kwargs,
        )

    @classmethod
    def _fix_infinite_vertices(
        cls,
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
            ]
        ],
        vertex: L,
        invert_direction: bool,
    ) -> set[L]:
        infinite_connections = infinite_vertices.pop(InfiniteNode(vertex), set())

        for infinite_connection in infinite_connections:
            if invert_direction:
                edges.add((infinite_connection, vertex))
            else:
                edges.add((vertex, infinite_connection))

        return infinite_connections

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
        incomplete_vertices: dict[
            IncompleteNode, set[tuple[L | InfiniteNode[L] | IncompleteNode, bool]]
        ],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
            ]
        ],
        vertex: L,
        depth: int,
        invert_direction: bool,
    ) -> None:
        if invert_direction:
            incomplete = lattice.has_other_successors_than(
                vertex,
                {
                    end
                    for start, end in edges
                    if start == vertex
                    and not isinstance(end, (InfiniteNode, IncompleteNode, BridgeNode))
                },
            )
        else:
            incomplete = lattice.has_other_predecessors_than(
                vertex,
                {
                    start
                    for start, end in edges
                    if end == vertex
                    and not isinstance(
                        start, (InfiniteNode, IncompleteNode, BridgeNode)
                    )
                },
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
    ) -> None:
        for infinite_vertex in infinite_vertices:
            if invert_direction:
                closest_connection = lattice.meet(vertex, infinite_vertex.base_node)
                if closest_connection == lattice.bottom():
                    continue
            else:
                closest_connection = lattice.join(vertex, infinite_vertex.base_node)
                if closest_connection == lattice.top():
                    continue

            infinite_connections = infinite_vertices.pop(infinite_vertex)
            infinite_connections.add(vertex)
            infinite_vertices[InfiniteNode(closest_connection)] = infinite_connections
            break
        else:
            infinite_vertices[InfiniteNode(vertex)] = {vertex}

    @classmethod
    def _add_children_to_worklist(
        cls,
        children: set[L],
        vertices: set[L | InfiniteNode[L] | IncompleteNode | BridgeNode[L]],
        worklist: list[tuple[L, bool, int]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
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
        incomplete_vertices: dict[
            IncompleteNode, set[tuple[L | InfiniteNode[L] | IncompleteNode, bool]]
        ],
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
    def _create_unprocessed_vertices(
        cls,
        lattice: Lattice[L],
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        vertices: set[L | InfiniteNode[L] | IncompleteNode | BridgeNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
            ]
        ],
        unprocessed_visible_vertices: set[L],
        invert_direction: bool,
    ) -> None:
        for infinite_vertex, infinite_connections in infinite_vertices.items():
            if invert_direction:
                filtered_visible_vertices = tuple(
                    visible_vertex
                    for visible_vertex in unprocessed_visible_vertices
                    if lattice.is_ancestor(visible_vertex, infinite_vertex.base_node)
                )
            else:
                filtered_visible_vertices = tuple(
                    visible_vertex
                    for visible_vertex in unprocessed_visible_vertices
                    if lattice.is_descendant(visible_vertex, infinite_vertex.base_node)
                )

            if not filtered_visible_vertices:
                continue

            vertices_connections = {
                infinite_vertex.base_node: infinite_connections,
            }

            for visible_vertex in filtered_visible_vertices:
                visible_vertex_connections = set()

                for vertex, connections in vertices_connections.items():
                    if invert_direction:
                        visible_vertex_included = lattice.is_ancestor(
                            visible_vertex, vertex
                        )
                    else:
                        visible_vertex_included = lattice.is_descendant(
                            visible_vertex, vertex
                        )

                    if not visible_vertex_included:
                        continue

                    connections.add(visible_vertex)

                    for connection in connections.copy():
                        if invert_direction:
                            connection_included = lattice.is_ancestor(
                                connection, visible_vertex
                            )
                        else:
                            connection_included = lattice.is_descendant(
                                connection, visible_vertex
                            )

                        if not connection_included:
                            continue

                        visible_vertex_connections.add(connection)

                        if invert_direction:
                            connection_contained = lattice.reverse_contains(
                                visible_vertex, connection
                            )
                        else:
                            connection_contained = lattice.contains(
                                visible_vertex, connection
                            )

                        if connection_contained:
                            connections.remove(connection)

                vertices_connections[visible_vertex] = visible_vertex_connections

            for visible_vertex in filtered_visible_vertices:
                vertices.add(visible_vertex)
                for connection in vertices_connections[visible_vertex]:
                    if invert_direction:
                        if lattice.is_predecessor(visible_vertex, connection):
                            edges.add((visible_vertex, connection))
                        else:
                            connection_infinite_vertex = InfiniteNode(connection)
                            vertices.add(connection_infinite_vertex)
                            edges.update(
                                (
                                    (visible_vertex, connection_infinite_vertex),
                                    (connection_infinite_vertex, connection),
                                )
                            )
                    elif lattice.is_successor(connection, visible_vertex):
                        edges.add((connection, visible_vertex))
                    else:
                        connection_infinite_vertex = InfiniteNode(connection)
                        vertices.add(connection_infinite_vertex)
                        edges.update(
                            (
                                (connection, connection_infinite_vertex),
                                (connection_infinite_vertex, visible_vertex),
                            )
                        )

    @classmethod
    def _create_incomplete_edges(
        cls,
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        incomplete_vertices: dict[
            IncompleteNode, set[tuple[L | InfiniteNode[L] | IncompleteNode, bool]]
        ],
        half_vertical_size: int,
        vertices: set[L | InfiniteNode[L] | IncompleteNode | BridgeNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
            ]
        ],
        invert_direction: bool,
    ) -> None:
        for infinite_vertex, infinite_connections in infinite_vertices.items():
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

            for infinite_connection in infinite_connections:
                if invert_direction:
                    edges.add((infinite_vertex, infinite_connection))
                else:
                    edges.add((infinite_connection, infinite_vertex))

    @classmethod
    def _create_infinite_edges(
        cls,
        lattice: Lattice[L],
        incomplete_vertices: dict[
            IncompleteNode, set[tuple[L | InfiniteNode[L] | IncompleteNode, bool]]
        ],
        infinite_vertices: dict[InfiniteNode[L], set[L]],
        vertices: set[L | InfiniteNode[L] | IncompleteNode | BridgeNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
            ]
        ],
        invert_direction: bool,
    ) -> None:
        for incomplete_vertex, infinite_connections in incomplete_vertices.items():
            vertices.add(incomplete_vertex)

            if invert_direction:
                for (
                    connection_vertex,
                    connection_invert_direction,
                ) in infinite_connections:
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
                for (
                    connection_vertex,
                    connection_invert_direction,
                ) in infinite_connections:
                    if connection_invert_direction:
                        edges.add((incomplete_vertex, connection_vertex))
                    else:
                        edges.add((connection_vertex, incomplete_vertex))

                if all(start != incomplete_vertex for start, _ in edges):
                    infinite_vertex = InfiniteNode(lattice.bottom())
                    vertices.add(infinite_vertex)
                    edges.add((incomplete_vertex, infinite_vertex))
                    infinite_vertices[infinite_vertex] = {lattice.bottom()}

    @classmethod
    def _bridge_infinite_vertices(
        cls,
        lattice: Lattice[L],
        bottom_infinite_vertices: dict[InfiniteNode[L], set[L]],
        top_infinite_vertices: dict[InfiniteNode[L], set[L]],
        vertices: set[L | InfiniteNode[L] | IncompleteNode | BridgeNode[L]],
        edges: set[
            tuple[
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
                L | InfiniteNode[L] | IncompleteNode | BridgeNode[L],
            ]
        ],
        final_visible_vertices: set[L],
    ) -> None:
        vertices_connections: dict[L | InfiniteNode[L], set[L | InfiniteNode[L]]] = (
            defaultdict(set)
        )

        bridge_vertices: set[tuple[InfiniteNode[L], InfiniteNode[L]]] = set()
        for bottom_infinite_vertex in bottom_infinite_vertices:
            for top_infinite_vertex in top_infinite_vertices:
                if lattice.includes(
                    top_infinite_vertex.base_node, bottom_infinite_vertex.base_node
                ):
                    vertices_connections[top_infinite_vertex].add(
                        bottom_infinite_vertex
                    )
                    bridge_vertices.add((bottom_infinite_vertex, top_infinite_vertex))

        for visible_vertex in final_visible_vertices:
            vertices.add(visible_vertex)

            visible_vertex_connections = set()

            for vertex, connections in vertices_connections.items():
                if isinstance(vertex, InfiniteNode):
                    visible_vertex_included = lattice.is_descendant(
                        visible_vertex, vertex.base_node
                    )
                else:
                    visible_vertex_included = lattice.is_descendant(
                        visible_vertex, vertex
                    )

                if not visible_vertex_included:
                    continue

                connections.add(visible_vertex)

                for connection in connections.copy():
                    if isinstance(connection, InfiniteNode):
                        connection_included = lattice.is_descendant(
                            connection.base_node, visible_vertex
                        )
                    else:
                        connection_included = lattice.is_descendant(
                            connection, visible_vertex
                        )

                    connected_to_incomplete_vertex = any(
                        isinstance(start, IncompleteNode)
                        for start, end in edges
                        if end == connection
                    )

                    if not connection_included and not connected_to_incomplete_vertex:
                        continue

                    visible_vertex_connections.add(connection)

                    if (
                        not connected_to_incomplete_vertex
                        and not isinstance(connection, InfiniteNode)
                        and lattice.contains(visible_vertex, connection)
                    ):
                        connections.remove(connection)

            vertices_connections[visible_vertex] = visible_vertex_connections

        for vertex, connections in vertices_connections.items():
            for connection in connections:
                if (
                    final_visible_vertices
                    and (connection, vertex) in bridge_vertices
                    and isinstance(vertex, InfiniteNode)
                    and isinstance(connection, InfiniteNode)
                ):
                    bridge_vertex = BridgeNode(connection, vertex)
                    vertices.add(bridge_vertex)
                    edges.update(
                        (
                            (connection, bridge_vertex),
                            (bridge_vertex, vertex),
                        )
                    )
                elif (
                    isinstance(vertex, InfiniteNode)
                    or isinstance(connection, InfiniteNode)
                    or lattice.is_successor(connection, vertex)
                ):
                    edges.add((connection, vertex))
                else:
                    infinite_vertex = InfiniteNode(connection)
                    vertices.add(infinite_vertex)
                    edges.update(
                        (
                            (connection, infinite_vertex),
                            (infinite_vertex, vertex),
                        )
                    )

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

    def color_path(
        self,
        start: Hashable,
        end: Hashable,
        color: ManimColor = GREEN,
        text_color: ManimColor = WHITE,
    ):
        path = nx.shortest_path(self._graph, start, end)

        for start_element, end_element in zip(path, path[1:]):
            mobject = self.vertices[start_element]
            edge_mobject = self.edges[(start_element, end_element)]

            mobject.color = color
            for submobject in mobject.submobjects:
                submobject.color = text_color

            edge_mobject.z_index = 1
            edge_mobject.color = color
            for submobject in edge_mobject.submobjects:
                submobject.color = color

        mobject = self.vertices[path[-1]]
        mobject.color = color
        for submobject in mobject.submobjects:
            submobject.color = text_color
