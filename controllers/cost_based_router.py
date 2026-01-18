import numpy as np

class CostBasedRouter:
    def __init__(self, alpha=0.7, beta=0.3):
        """
        alpha → predicted traffic
        beta  → uncertainty penalty
        """
        self.alpha = alpha
        self.beta = beta

    def compute_node_cost(self, traffic_mean, traffic_std):
        """
        traffic_mean: [N]
        traffic_std:  [N]
        """
        return self.alpha * traffic_mean + self.beta * traffic_std

    def choose_route(self, routes, node_costs):
        """
        routes: list of routes (list of node indices)
        """
        route_costs = []
        for route in routes:
            route_costs.append(sum(node_costs[n] for n in route))

        return routes[np.argmin(route_costs)]
