
# evaluate.py
model.load_state_dict(torch.load("best_st_mamba.pt"))
test_mae, test_rmse, test_mape = eval_epoch(...)
