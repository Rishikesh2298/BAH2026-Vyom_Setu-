"""
Data Assimilation Methods for Climate Digital Twin
Ensemble Kalman Filter (EnKF), 3D-Var, and hybrid approaches
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.linalg import solve, cholesky
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

@dataclass
class AnalysisState:
    """Analysis state from data assimilation"""
    state_vector: np.ndarray    # Analyzed state
    error_covariance: np.ndarray  # Analysis error covariance
    innovation: np.ndarray      # Observation innovations
    analysis_increment: np.ndarray  # State increment from assimilation

class EnsembleKalmanFilter:
    """
    Ensemble Kalman Filter (EnKF) for Data Assimilation
    Suitable for non-linear climate systems
    """
    
    def __init__(self, ensemble_size: int = 100, state_dimension: int = 10000):
        """
        Initialize EnKF
        
        Args:
            ensemble_size: Number of ensemble members
            state_dimension: Dimension of the state vector
        """
        self.ensemble_size = ensemble_size
        self.state_dim = state_dimension
        self.ensemble_members = np.zeros((state_dimension, ensemble_size))
        
        logger.info(f"Initialized EnKF with {ensemble_size} members and state dim {state_dimension}")
    
    def initialize_ensemble(self, background_state: np.ndarray, 
                           background_error_std: float):
        """
        Initialize ensemble from background state with perturbations
        
        Args:
            background_state: Background (prior) state estimate
            background_error_std: Standard deviation of background errors
        """
        logger.info("Initializing ensemble perturbations")
        
        for i in range(self.ensemble_size):
            perturbations = np.random.normal(0, background_error_std, self.state_dim)
            self.ensemble_members[:, i] = background_state + perturbations
    
    def forecast_step(self, forecast_model):
        """
        Advance ensemble forward in time using forecast model
        
        Args:
            forecast_model: Function that advances state in time
        """
        logger.info("Executing ensemble forecast")
        
        for i in range(self.ensemble_size):
            self.ensemble_members[:, i] = forecast_model(self.ensemble_members[:, i])
    
    def analysis_step(self, observations: np.ndarray, 
                     observation_errors: np.ndarray,
                     observation_operator) -> AnalysisState:
        """
        Update ensemble based on observations (analysis step)
        
        Args:
            observations: Observation values
            observation_errors: Observation error standard deviations
            observation_operator: Function that maps state to observation space
            
        Returns:
            AnalysisState with analyzed state and statistics
        """
        logger.info(f"Executing analysis update with {len(observations)} observations")
        
        n_obs = len(observations)
        n_ens = self.ensemble_size
        
        # Project ensemble to observation space
        H_ensemble = np.zeros((n_obs, n_ens))
        for i in range(n_ens):
            H_ensemble[:, i] = observation_operator(self.ensemble_members[:, i])
        
        # Compute ensemble mean and anomalies
        H_mean = H_ensemble.mean(axis=1)
        H_anom = H_ensemble - H_mean[:, np.newaxis]
        
        # Compute forecast error covariance in observation space
        P_f_HT = H_anom / np.sqrt(n_ens - 1)  # Forecast error in obs space
        
        # Compute Kalman gain
        obs_cov_diag = observation_errors ** 2
        HPfHT_R = np.dot(P_f_HT, P_f_HT.T) + np.diag(obs_cov_diag)
        
        K = np.dot(self.ensemble_members, P_f_HT.T) @ np.linalg.inv(HPfHT_R)
        
        # Compute innovations
        innovations = observations - H_mean
        
        # Update ensemble mean
        ensemble_mean_prior = self.ensemble_members.mean(axis=1)
        ensemble_mean_posterior = ensemble_mean_prior + K @ innovations
        
        # Update ensemble members
        for i in range(n_ens):
            pert = H_anom[:, i] / np.sqrt(n_ens - 1)
            self.ensemble_members[:, i] = self.ensemble_members[:, i] + K @ (observations - H_ensemble[:, i])
        
        # Compute analysis statistics
        analysis_state = AnalysisState(
            state_vector=self.ensemble_members.mean(axis=1),
            error_covariance=np.cov(self.ensemble_members),
            innovation=innovations,
            analysis_increment=ensemble_mean_posterior - ensemble_mean_prior
        )
        
        logger.info(f"Mean innovation: {innovations.mean():.4f}")
        return analysis_state

class VariationalAssimilation3DVar:
    """
    Three-Dimensional Variational Data Assimilation (3D-Var)
    Finds optimal analysis state that balances observations and background
    """
    
    def __init__(self, state_dimension: int = 10000, max_iterations: int = 50):
        """
        Initialize 3D-Var
        
        Args:
            state_dimension: Dimension of state vector
            max_iterations: Maximum iterations for optimization
        """
        self.state_dim = state_dimension
        self.max_iterations = max_iterations
        self.background_error_cov = None
        
        logger.info(f"Initialized 3D-Var with state dim {state_dimension}")
    
    def set_background_error_covariance(self, cov_matrix: np.ndarray):
        """
        Set background error covariance matrix B
        
        Args:
            cov_matrix: Background error covariance matrix
        """
        self.background_error_cov = cov_matrix
        logger.info(f"Set B matrix of shape {cov_matrix.shape}")
    
    def cost_function(self, state: np.ndarray, background: np.ndarray,
                     observations: np.ndarray, obs_error_cov: np.ndarray,
                     obs_operator) -> float:
        """
        Compute 3D-Var cost function J(x) = J_b + J_o
        
        Args:
            state: Current state estimate
            background: Background state
            observations: Observations
            obs_error_cov: Observation error covariance matrix R
            obs_operator: Observation operator function
            
        Returns:
            Cost function value
        """
        x_minus_xb = state - background
        
        # Background term
        J_b = 0.5 * np.dot(x_minus_xb, np.linalg.solve(self.background_error_cov, x_minus_xb))
        
        # Observation term
        y_minus_Hx = observations - obs_operator(state)
        J_o = 0.5 * np.dot(y_minus_Hx, np.linalg.solve(obs_error_cov, y_minus_Hx))
        
        return J_b + J_o
    
    def analyze(self, background: np.ndarray, observations: np.ndarray,
               obs_error_std: np.ndarray, obs_operator) -> AnalysisState:
        """
        Perform 3D-Var analysis
        
        Args:
            background: Background state
            observations: Observation vector
            obs_error_std: Observation error standard deviations
            obs_operator: Function mapping state to observation space
            
        Returns:
            AnalysisState with analyzed state
        """
        logger.info("Starting 3D-Var analysis")
        
        state = background.copy()
        obs_error_cov = np.diag(obs_error_std ** 2)
        
        # Simple gradient descent optimization
        learning_rate = 0.1
        for iteration in range(self.max_iterations):
            
            # Compute gradients
            h_x = obs_operator(state)
            innovation = observations - h_x
            
            # Approximate gradient (for simplicity)
            db_dx = np.linalg.solve(self.background_error_cov, state - background)
            
            # Update state
            state = state + learning_rate * (
                -db_dx + np.dot(np.linalg.inv(obs_error_cov), innovation)
            )
            
            if iteration % 10 == 0:
                cost = self.cost_function(state, background, observations, obs_error_cov, obs_operator)
                logger.info(f"Iteration {iteration}: Cost = {cost:.4f}")
        
        return AnalysisState(
            state_vector=state,
            error_covariance=self.background_error_cov,
            innovation=observations - obs_operator(state),
            analysis_increment=state - background
        )

class HybridDataAssimilation:
    """
    Hybrid Data Assimilation combining ensemble and variational methods
    """
    
    def __init__(self, ensemble_size: int = 50, state_dim: int = 10000):
        self.enkf = EnsembleKalmanFilter(ensemble_size, state_dim)
        self.var3d = VariationalAssimilation3DVar(state_dim)
        self.alpha = 0.5  # Weighting factor
        
        logger.info(f"Initialized Hybrid DA (alpha={self.alpha})")
    
    def analyze(self, ensemble_members: np.ndarray, observations: np.ndarray,
               obs_error_std: np.ndarray, obs_operator) -> AnalysisState:
        """
        Perform hybrid analysis
        
        Args:
            ensemble_members: Ensemble state members
            observations: Observations
            obs_error_std: Observation error standard deviations
            obs_operator: Observation operator
            
        Returns:
            Hybrid analysis state
        """
        logger.info("Performing Hybrid Data Assimilation")
        
        # EnKF analysis
        self.enkf.ensemble_members = ensemble_members
        enkf_result = self.enkf.analysis_step(observations, obs_error_std, obs_operator)
        
        # 3D-Var analysis
        background = ensemble_members.mean(axis=1)
        self.var3d.set_background_error_covariance(np.cov(ensemble_members))
        var3d_result = self.var3d.analyze(background, observations, obs_error_std, obs_operator)
        
        # Combine results
        hybrid_state = (1 - self.alpha) * enkf_result.state_vector + self.alpha * var3d_result.state_vector
        
        return AnalysisState(
            state_vector=hybrid_state,
            error_covariance=enkf_result.error_covariance,
            innovation=(enkf_result.innovation + var3d_result.innovation) / 2,
            analysis_increment=(enkf_result.analysis_increment + var3d_result.analysis_increment) / 2
        )
