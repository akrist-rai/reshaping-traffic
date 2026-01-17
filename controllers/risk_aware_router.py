import numpy as np

class RiskAwareRouter:
    def __init__(self, alpha=0.7, beta=0.3):
        """
        alpha → predicted traffic weight
        beta  → uncertainty (risk) weight
        """
        self.alpha = alpha
        self.beta = beta

    def compute_cost(self, traffic_mean, traffic_std):
        """
        traffic_mean: [N]
        traffic_std:  [N]
        """
        return self.alpha * traffic_mean + self.beta * traffic_std

    def select_route(self, routes, node_costs):
        """
        routes: list of routes (list of node indices)
        """
        route_costs = []
        for r in routes:
            cost = sum(node_costs[n] for n in r)
            route_costs.append(cost)

        return routes[np.argmin(route_costs)]
