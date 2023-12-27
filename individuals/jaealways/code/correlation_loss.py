import torch
import scipy.stats as stats



class SpearmanCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(SpearmanCorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_rank = self.soft_rank(y_pred)
        y_true_rank = self.soft_rank(y_true)

        y_pred_rank_centered = y_pred_rank - y_pred_rank.mean()
        y_true_rank_centered = y_true_rank - y_true_rank.mean()

        covariance = (y_pred_rank_centered * y_true_rank_centered).mean()

        y_pred_rank_std = y_pred_rank_centered.std()
        y_true_rank_std = y_true_rank_centered.std()

        correlation = covariance / (y_pred_rank_std * y_true_rank_std + 1e-12)  # Avoid division by zero

        return -correlation

    def soft_rank(x):
        ranks = torch.argsort(torch.argsort(x, dim=0), dim=0).float()
        return ranks



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