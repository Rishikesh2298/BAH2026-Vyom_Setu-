"""
ML Pipeline Package
Machine Learning models and data assimilation methods
"""

__version__ = "1.0.0"

from .data_assimilation import (
    EnsembleKalmanFilter,
    VariationalAssimilation3DVar,
    HybridDataAssimilation,
    AnalysisState
)

from .climate_models import (
    LSTMClimateForecaster,
    ConvolutionalClimateModel,
    TransformerClimateForecaster,
    HybridClimateModel,
    GraphNeuralNetworkModel,
    create_model
)

from .training import (
    ClimateDataset,
    ModelTrainer,
    EnsembleTrainer,
    generate_synthetic_data
)

__all__ = [
    'EnsembleKalmanFilter',
    'VariationalAssimilation3DVar',
    'HybridDataAssimilation',
    'AnalysisState',
    'LSTMClimateForecaster',
    'ConvolutionalClimateModel',
    'TransformerClimateForecaster',
    'HybridClimateModel',
    'GraphNeuralNetworkModel',
    'create_model',
    'ClimateDataset',
    'ModelTrainer',
    'EnsembleTrainer',
    'generate_synthetic_data'
]
