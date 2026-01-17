from utils.uncertainty import mc_dropout_predict

model.eval()

mae = rmse = mape = 0.0
uncertainty = 0.0

with torch.no_grad():
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device).permute(0, 2, 1)

        mean, std = mc_dropout_predict(model, X, adj, runs=20)

        mae += masked_mae(mean, Y).item()
        rmse += masked_rmse(mean, Y).item()
        mape += masked_mape(mean, Y).item()
        uncertainty += std.mean().item()

print("Test MAE:", mae / len(loader))
print("Test RMSE:", rmse / len(loader))
print("Test MAPE:", mape / len(loader))
print("Avg Predictive Uncertainty:", uncertainty / len(loader))
