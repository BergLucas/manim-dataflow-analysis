from __future__ import annotations

from manim.mobject.geometry.polygram import Rectangle
from manim.mobject.geometry.arc import TipableVMobject, Dot
from manim.mobject.geometry.line import Line
from manim.mobject.mobject import Mobject
from manim.mobject.text.tex_mobject import MathTex, SingleStringMathTex, Tex
from manim.mobject.graph import DiGraph, LayoutName, LayoutFunction
from manim.mobject.text.text_mobject import Text
from manim.utils.color import BLACK, WHITE
from typing import TYPE_CHECKING, Hashable
import networkx as nx


if TYPE_CHECKING:
    from manim.typing import Point3D


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


class BetterDiGraph(DiGraph):

    def __init__(
        self,
        vertices: list[Hashable],
        edges: list[tuple[Hashable, Hashable]],
        labels: bool | dict = False,
        label_fill_color: str = BLACK,
        layout: LayoutName | dict[Hashable, Point3D] | LayoutFunction = "spring",
        layout_scale: float | tuple[float, float, float] = 2,
        layout_config: dict | None = None,
        vertex_type: type[Mobject] = Dot,
        vertex_config: dict | None = None,
        vertex_mobjects: dict | None = None,
        edge_type: type[TipableVMobject] = Line,
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
        self, edges: list[tuple[Hashable, Hashable]], edge_type: type[TipableVMobject]
    ):
        # Some part of this method is copied from manim.mobject.graph.GenericGraph._populate_edge_dict

        self.edges = {
            (u, v): self._create_edge_mobject(
                u, v, edge_type, self._edge_config[(u, v)]
            )
            for (u, v) in edges
        }

        for (u, v), edge in self.edges.items():
            edge.add_tip(**self._tip_config[(u, v)])

    def update_edges(self, graph: nx.DiGraph):
        # Some part of this method is copied from manim.mobject.graph.DiGraph.update_edges

        edge: TipableVMobject
        for (u, v), edge in graph.edges.items():
            edge_type = type(edge)
            tip = edge.pop_tips()[0]
            new_edge = self._create_edge_mobject(
                u, v, edge_type, self._edge_config[(u, v)]
            )
            edge.become(new_edge)
            edge.add_tip(tip)

    def _add_edge(
        self,
        edge: tuple[Hashable, Hashable],
        edge_type: type[TipableVMobject] = TipableVMobject,
        edge_config: dict | None = None,
    ):
        # Some part of this method is copied from manim.mobject.graph.GenericGraph._add_edge

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

        edge_mobject = self._create_edge_mobject(u, v, edge_type, edge_config)
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def _create_edge_mobject(
        self,
        start: Hashable,
        end: Hashable,
        edge_type: type[TipableVMobject],
        edge_config: dict,
    ):
        return edge_type(
            self[start].get_center(),
            self[end].get_center(),
            z_index=-1,
            **edge_config,
        )
