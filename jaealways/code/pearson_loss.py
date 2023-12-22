import torch

class PearsonCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Mean-centering the predictions and the targets
        y_pred_centered = y_pred - y_pred.mean()
        y_true_centered = y_true - y_true.mean()

        # Compute the covariance between y_true and y_pred
        covariance = (y_pred_centered * y_true_centered).mean()

        # Compute the standard deviations of y_true and y_pred
        y_pred_std = y_pred_centered.std()
        y_true_std = y_true_centered.std()

        # Compute Pearson Correlation Coefficient
        correlation = covariance / (y_pred_std * y_true_std + 1e-12)  # Adding a small value to avoid division by zero

        # Since the optimizer minimizes the loss, return the negative correlation
        return -correlation