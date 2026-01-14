"""
Traffic Routing & Flow Control Engine (Model-2)

Consumes future traffic predictions from Model-1
and produces stable, congestion-aware route decisions.

Design goals:
- Minimal
- Deterministic
- Industry-grade
- No unnecessary learning
"""

import random
import networkx as nx
import numpy as np


# ============================================================
# 1. TEMPORAL AGGREGATION (Model-1 → Risk Signal)
# ============================================================

def aggregate_congestion(pred_traffic, mode="risk"):
    """
    pred_traffic: np.ndarray [N, H]
    returns: np.ndarray [N]

    Converts future trajectory into a single congestion risk score.
    """
    if mode == "mean":
        return pred_traffic.mean(axis=1)

    if mode == "risk":
        # Industry-style: penalize spikes more than average
        return 0.6 * pred_traffic.mean(axis=1) + 0.4 * pred_traffic.max(axis=1)

    raise ValueError(f"Unknown aggregation mode: {mode}")


# ============================================================
# 2. EDGE COST FUNCTION (Congestion-Aware)
# ============================================================

def edge_cost(distance, congestion, alpha=0.7):
    """
    distance   : base road distance
    congestion : predicted congestion risk
    alpha      : sensitivity coefficient
    """
    return distance * (1.0 + alpha * congestion)


# ============================================================
# 3. GRAPH BUILDER
# ============================================================

def build_weighted_graph(adj, dist, node_congestion):
    """
    adj  : [N, N] adjacency matrix (0/1)
    dist : [N, N] distance matrix
    node_congestion : [N] congestion score per node
    """
    G = nx.Graph()
    N = adj.shape[0]

    for i in range(N):
        for j in range(N):
            if adj[i, j] == 1:
                G.add_edge(
                    i, j,
                    weight=edge_cost(dist[i, j], node_congestion[j]),
                    distance=dist[i, j]
                )
    return G


# ============================================================
# 4. ROUTE COMPUTATION
# ============================================================

def compute_routes(G, src, dst):
    """
    Computes:
    - shortest path (distance only)
    - optimized path (distance + congestion)
    """
    optimized_route = nx.shortest_path(G, src, dst, weight="weight")
    shortest_route  = nx.shortest_path(G, src, dst, weight="distance")

    return optimized_route, shortest_route


# ============================================================
# 5. FLOW SPLITTING (Probabilistic)
# ============================================================

def assign_route(p_optimized=0.75):
    """
    Probabilistic assignment to prevent route collapse.
    """
    return "optimized" if random.random() < p_optimized else "shortest"


# ============================================================
# 6. STABILITY GUARD (ANTI-OSCILLATION)
# ============================================================

class RouteStabilityGuard:
    """
    Prevents sudden routing policy shifts that cause oscillations.
    """
    def __init__(self, initial_split=0.75, max_change=0.15):
        self.current_split = initial_split
        self.max_change = max_change

    def smooth(self, target_split):
        delta = target_split - self.current_split
        delta = np.clip(delta, -self.max_change, self.max_change)
        self.current_split += delta
        return self.current_split


# ============================================================
# 7. MAIN MODEL-2 INTERFACE
# ============================================================

class TrafficRouter:
    """
    Model-2 Controller

    Consumes Model-1 predictions and returns user routes.
    """

    def __init__(self, adj, dist, alpha=0.7):
        self.adj = adj
        self.dist = dist
        self.alpha = alpha
        self.guard = RouteStabilityGuard()

    def route(self, pred_traffic, src, dst):
        """
        pred_traffic : np.ndarray [N, H] (Model-1 output)
        src, dst     : source & destination nodes
        """

        # 1. Aggregate future congestion
        node_cong = aggregate_congestion(pred_traffic)

        # 2. Build congestion-aware graph
        G = build_weighted_graph(self.adj, self.dist, node_cong)

        # 3. Compute dual routes
        opt_route, short_route = compute_routes(G, src, dst)

        # 4. Stable flow split
        split = self.guard.smooth(0.75)

        # 5. Assign route
        choice = assign_route(split)

        return {
            "chosen_route": opt_route if choice == "optimized" else short_route,
            "optimized_route": opt_route,
            "shortest_route": short_route,
            "split_ratio": split,
            "policy": choice
        }
