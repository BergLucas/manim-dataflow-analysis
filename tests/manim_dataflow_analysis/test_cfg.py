import unittest
import networkx as nx
from manim_dataflow_analysis.cfg import cfg_layout


class TestCfgLayout(unittest.TestCase):
    def test_straight_program(self):
        graph = nx.DiGraph()

        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(3, 4)

        layout, _ = cfg_layout(graph, root_vertex=1, condition_vertices={})

        self.assertDictEqual(
            {node: tuple(coord) for node, coord in layout.items()},
            {
                1: (0, 3, 0),
                2: (0, 2, 0),
                3: (0, 1, 0),
                4: (0, 0, 0),
            },
        )

    def test_program_with_an_if_statement(self):
        graph = nx.DiGraph()

        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        graph.add_edge(3, 5)
        graph.add_edge(4, 5)

        layout, _ = cfg_layout(graph, root_vertex=1, condition_vertices={2: [3, 4]})

        self.assertDictEqual(
            {node: tuple(coord) for node, coord in layout.items()},
            {
                1: (0, 3, 0),
                2: (0, 2, 0),
                3: (0, 1, 0),
                4: (1, 1, 0),
                5: (0, 0, 0),
            },
        )

    def test_program_with_a_nested_if_statement(self):
        graph = nx.DiGraph()

        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        graph.add_edge(4, 5)
        graph.add_edge(4, 6)
        graph.add_edge(5, 3)
        graph.add_edge(6, 3)

        layout, _ = cfg_layout(
            graph, root_vertex=1, condition_vertices={2: [3, 4], 4: [5, 6]}
        )

        self.assertDictEqual(
            {node: tuple(coord) for node, coord in layout.items()},
            {
                1: (0, 4, 0),
                2: (0, 3, 0),
                3: (0, 0, 0),
                4: (1, 2, 0),
                5: (1, 1, 0),
                6: (2, 1, 0),
            },
        )

    def test_program_with_a_while_statement(self):
        graph = nx.DiGraph()

        graph.add_edge(1, 2)
        graph.add_edge(2, 3)
        graph.add_edge(2, 4)
        graph.add_edge(4, 2)

        layout, _ = cfg_layout(graph, root_vertex=1, condition_vertices={2: [3, 4]})

        self.assertDictEqual(
            {node: tuple(coord) for node, coord in layout.items()},
            {
                1: (0, 2, 0),
                2: (0, 1, 0),
                3: (0, 0, 0),
                4: (1, 0, 0),
            },
        )
