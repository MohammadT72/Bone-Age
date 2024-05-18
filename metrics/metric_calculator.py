# Pytorch
from torch import Tensor
from torcheval.metrics import MeanSquaredError, R2Score
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

class MetricsCalculator:
    """
    A class to calculate and aggregate various metrics for model evaluation.
    Attributes:
    -----------
    metrics : list
        A list of metrics to be calculated.
    Methods:
    --------
    aggregate(y_pred, y):
        Aggregates the metrics for the given predictions and actual values.
    """
    def __init__(self, metrics: list):
        """
        Initializes the MetricsCalculator with a list of metrics.
        Parameters:
        ----------
        metrics : list
            A list of metrics to be calculated.
        """
        self.metrics = metrics
        self.metric_calculators = self._initialize_metrics()

    def _initialize_metrics(self):
        """
        Initializes the metric calculators based on the metrics list.
        Returns:
        -------
        dict
            A dictionary mapping metric names to their corresponding metric calculators.
        """
        metric_calculators = {}
        for metric in self.metrics:
            if metric == 'mse':
                metric_calculators[metric] = MeanSquaredError()
            elif metric == 'r2s':
                metric_calculators[metric] = R2Score()
            elif metric == 'mae':
                metric_calculators[metric] = MeanAbsoluteError()
            # Add more metrics as needed
            else:
                raise ValueError(f"Metric '{metric}' is not implemented.")
        return metric_calculators

    def aggregate(self, y_pred: Tensor, y: Tensor) -> dict:
        """
        Aggregates the metrics for the given predictions and actual values.
        Parameters:
        ----------
        y_pred : torch.Tensor
            The predicted values.
        y : torch.Tensor
            The actual values.
        Returns:
        -------
        dict
            A dictionary containing the aggregated metrics.
        """
        results = {}
        for metric, calculator in self.metric_calculators.items():
            calculator.update(y_pred, y)
            results[metric] = calculator.compute().item()
            calculator.reset()  # Reset the metric calculator for the next use
        return results
