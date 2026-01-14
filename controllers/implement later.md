
1️⃣ What “Closed-Loop Simulation” Means Here
Open-loop (what most ML projects do ❌)

Traffic → Model-1 → Prediction → (stop)

Problems:

    No proof routing helps

    No system behavior

    No control aspect

Closed-loop (what you want ✅)

Traffic state
   ↓
Model-1 (predict future)
   ↓
Model-2 (route decision)
   ↓
Traffic redistribution
   ↓
New traffic state
   ↓
(repeat)

This answers system questions, not ML questions:

    Does congestion actually reduce?

    Do bottlenecks move or disappear?

    Is routing stable over time?

That’s why this is essential even without RL.
2️⃣ What We Simulate (Minimal & Correct)

We do NOT simulate physics, cars, or signals.

We simulate only what matters:

    Traffic is a node congestion vector

    Routing adds load along routes

    Congestion decays naturally over time

This is enough to validate control logic.
3️⃣ Closed-Loop Traffic Simulator (Drop-in Code)

📁 controllers/traffic_simulator.py

import numpy as np

class TrafficSimulator:
    """
    Minimal closed-loop traffic simulator.
    """

    def __init__(self, decay=0.85):
        """
        decay: how fast congestion dissipates (0 < decay < 1)
        """
        self.decay = decay

    def step(self, traffic, routes):
        """
        traffic: np.ndarray [N]
        routes: list of (route_nodes, load)
        """
        # Natural dissipation
        traffic = traffic * self.decay

        # Apply new traffic loads
        for route, load in routes:
            for node in route:
                traffic[node] += load

        return traffic

Why this is enough:

    Decay = vehicles leave system

    Route load = users entering roads

    No unnecessary complexity

4️⃣ System-Level Metrics (What to Report Instead of RL Rewards)

These metrics are industry-meaningful.
1️⃣ Average Congestion (System Load)

def avg_congestion(traffic):
    return traffic.mean()

Meaning:

    Overall network health

    Lower = better

2️⃣ Max Congestion (Bottleneck Detection)

def max_congestion(traffic):
    return traffic.max()

Meaning:

    Worst jam in the system

    Indicates failure points

3️⃣ Route Imbalance (Stability Metric)

def route_imbalance(opt_load, short_load):
    total = opt_load + short_load
    return abs(opt_load - short_load) / (total + 1e-6)

Meaning:

    0 → perfectly balanced

    High → risk of route collapse

This replaces “RL stability rewards”.
4️⃣ Travel Time Proxy (User QoS)

We don’t simulate time — we approximate it.

def travel_time_proxy(route, traffic):
    return sum(traffic[node] for node in route)

Meaning:

    Higher congestion → slower travel

    Good enough for comparison

5️⃣ Full Closed-Loop Evaluation (Model-1 + Model-2)

📁 controllers/closed_loop_eval.py

import numpy as np
import torch

from controllers.traffic_simulator import TrafficSimulator
from controllers.traffic_router import TrafficRouter


def closed_loop_run(
    model,
    init_traffic,
    adj,
    dist,
    src,
    dst,
    steps=20,
    device="cpu"
):
    router = TrafficRouter(adj, dist)
    simulator = TrafficSimulator()

    traffic = init_traffic.copy()
    history = []

    for t in range(steps):
        # ---- Predict future traffic ----
        with torch.no_grad():
            pred = model(traffic.to(device), adj)
            pred_np = pred[0].cpu().numpy()  # [N, H]

        # ---- Route decision ----
        decision = router.route(pred_np, src, dst)

        # ---- Apply traffic split ----
        routes = [
            (decision["optimized_route"], 0.75),
            (decision["shortest_route"], 0.25)
        ]

        traffic = simulator.step(traffic, routes)

        # ---- Metrics ----
        history.append({
            "step": t,
            "avg_congestion": traffic.mean(),
            "max_congestion": traffic.max(),
            "travel_time_opt": sum(traffic[n] for n in decision["optimized_route"]),
            "travel_time_short": sum(traffic[n] for n in decision["shortest_route"]),
            "split": decision["split_ratio"]
        })

    return history

6️⃣ What This Proves (Very Important)

With this setup, you can now experimentally show:

✔ Routing reduces average congestion
✔ Dual-route prevents bottleneck spikes
✔ Stability guard prevents oscillation
✔ Control loop improves system behavior, not just prediction

That is far stronger than adding RL.
