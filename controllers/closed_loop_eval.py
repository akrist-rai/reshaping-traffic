
import numpy as np
import torch

from controllers.traffic_router import TrafficRouter
from controllers.traffic_simulator import TrafficSimulator


def closed_loop_evaluation(
    model,
    initial_traffic,
    adj,
    dist,
    src,
    dst,
    steps=10,
    device="cpu"
):
    """
    Full perception → decision → control → feedback loop
    """

    router = TrafficRouter(adj, dist)
    simulator = TrafficSimulator(adj)

    traffic = initial_traffic.copy()

    history = []

    for t in range(steps):
        # 1. Predict future traffic (Model-1)
        model.eval()
        with torch.no_grad():
            pred = model(traffic.to(device), adj)
            pred_np = pred.cpu().numpy().mean(axis=0)  # [N, H]

        # 2. Routing decision (Model-2)
        decision = router.route(pred_np, src, dst)

        # 3. Apply split
        routes = [
            (decision["optimized_route"], 0.75),
            (decision["shortest_route"], 0.25),
        ]

        # 4. Simulate traffic update
        traffic = simulator.step(traffic, routes)

        history.append({
            "step": t,
            "split": decision["split_ratio"],
            "policy": decision["policy"],
            "avg_congestion": traffic.mean()
        })

    return history
