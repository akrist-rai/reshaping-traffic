import torch

@torch.no_grad()
def mc_dropout_predict(model, X, adj, runs=20):
    """
    Monte-Carlo Dropout Inference
    Returns mean and std predictions
    """
    model.train()  # IMPORTANT: enable dropout

    preds = []
    for _ in range(runs):
        preds.append(model(X, adj))

    preds = torch.stack(preds, dim=0)
    mean = preds.mean(dim=0)
    std = preds.std(dim=0)

    model.eval()
    return mean, std
