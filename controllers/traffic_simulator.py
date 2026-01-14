"""
Closed-loop traffic flow simulator.

Simulates how routing decisions modify traffic state.
This is a SYSTEM simulator, not ML.
"""

import numpy as np


class TrafficSimulator:
    def __init__(self, adj, decay=0.85):
        """
        adj   : adjacency matrix [N, N]
        decay : how fast congestion dissipates
        """
        self.adj = adj
        self.decay = decay

    def apply_routes(self, traffic, route, load=1.0):
        """
        traffic : [N] current congestion
        route   : list of nodes
        load    : traffic volume
        """
        for node in route:
            traffic[node] += load
        return traffic

    def step(self, traffic, routes):
        """
        routes : list of (route, load)
        """
        # decay old congestion
        traffic = traffic * self.decay

        for route, load in routes:
            traffic = self.apply_routes(traffic, route, load)

        return traffic
