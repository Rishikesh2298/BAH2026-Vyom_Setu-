"""
IMD (Indian Meteorological Department) Ground Network Data Ingestion
Handles surface and upper-air observations from meteorological stations
"""

import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class IMDStationData:
    """Container for IMD station observations"""
    station_id: str
    station_name: str
    latitude: float
    longitude: float
    elevation: float
    timestamp: datetime
    observations: Dict[str, float]  # Variables and their values

class IMDDataIngester:
    """
    IMD (Indian Meteorological Department) Data Ingestion
    Handles observations from ~2400 meteorological stations across India
    """
    
    def __init__(self):
        self.agency = "IMD"
        self.update_frequency = "daily"
        self.num_stations = 2400
        
        # Standard meteorological variables
        self.standard_variables = [
            "temperature",           # °C
            "humidity",              # %
            "wind_speed",            # m/s
            "wind_direction",        # degrees
            "rainfall",              # mm
            "sea_level_pressure",    # hPa
            "visibility",            # km
            "cloud_cover",           # oktas
            "dew_point",             # °C
            "evaporation"            # mm
        ]
        
        # Major IMD stations (placeholder)
        self.major_stations = [
            {"id": "NEW_DELHI", "name": "New Delhi", "lat": 28.57, "lon": 77.20, "elev": 216},
            {"id": "BOMBAY", "name": "Mumbai", "lat": 19.11, "lon": 72.87, "elev": 14},
            {"id": "CALCUTTA", "name": "Kolkata", "lat": 22.57, "lon": 88.37, "elev": 9},
            {"id": "MADRAS", "name": "Chennai", "lat": 13.00, "lon": 80.18, "elev": 7},
            {"id": "BANGALORE", "name": "Bangalore", "lat": 13.19, "lon": 77.57, "elev": 920},
        ]
    
    def fetch_station_data(self, station_id: str, start_date: datetime, 
                          end_date: datetime) -> Optional[List[IMDStationData]]:
        """
        Fetch IMD station data for specified period
        
        Args:
            station_id: IMD station identifier
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            List of IMDStationData objects or None if fetch fails
        """
        logger.info(f"Fetching IMD data for station {station_id} from {start_date} to {end_date}")
        
        try:
            # Find station metadata
            station_info = next((s for s in self.major_stations if s["id"] == station_id), None)
            if not station_info:
                logger.warning(f"Station {station_id} not found")
                return None
            
            # Mock data generation: one observation per day
            data_list = []
            current_date = start_date
            
            while current_date <= end_date:
                obs = {var: np.random.randn() * 10 for var in self.standard_variables}
                
                # Ensure realistic ranges
                obs["temperature"] = np.random.uniform(15, 40)
                obs["humidity"] = np.random.uniform(30, 95)
                obs["wind_speed"] = np.random.uniform(0, 15)
                obs["rainfall"] = max(0, np.random.normal(5, 10))
                obs["sea_level_pressure"] = np.random.uniform(1000, 1020)
                
                station_data = IMDStationData(
                    station_id=station_id,
                    station_name=station_info["name"],
                    latitude=station_info["lat"],
                    longitude=station_info["lon"],
                    elevation=station_info["elev"],
                    timestamp=current_date,
                    observations=obs
                )
                data_list.append(station_data)
                
                current_date += pd.Timedelta(days=1)
            
            logger.info(f"Retrieved {len(data_list)} observations for {station_id}")
            return data_list
            
        except Exception as e:
            logger.error(f"Error fetching IMD data: {str(e)}")
            return None
    
    def fetch_network_data(self, start_date: datetime, end_date: datetime) -> Dict[str, List[IMDStationData]]:
        """
        Fetch data from multiple stations across the network
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            Dictionary with station_id as key and list of observations as value
        """
        logger.info(f"Fetching IMD network data from {start_date} to {end_date}")
        
        network_data = {}
        for station in self.major_stations:
            data = self.fetch_station_data(station["id"], start_date, end_date)
            if data:
                network_data[station["id"]] = data
        
        logger.info(f"Retrieved data from {len(network_data)} stations")
        return network_data

class IMDDataValidator:
    """Validate and quality control IMD observations"""
    
    def __init__(self):
        self.variable_ranges = {
            "temperature": (-50, 50),
            "humidity": (0, 100),
            "wind_speed": (0, 100),
            "rainfall": (0, 500),
            "sea_level_pressure": (850, 1050),
            "visibility": (0, 200),
            "dew_point": (-60, 50)
        }
    
    def validate_observation(self, observation: Dict[str, float]) -> bool:
        """
        Validate a single meteorological observation
        
        Args:
            observation: Dictionary of variable names and values
            
        Returns:
            True if observation passes validation, False otherwise
        """
        for var, value in observation.items():
            if var not in self.variable_ranges:
                continue
            
            min_val, max_val = self.variable_ranges[var]
            if not (min_val <= value <= max_val):
                logger.warning(f"{var} = {value} outside range [{min_val}, {max_val}]")
                return False
        
        return True
    
    def detect_outliers(self, data_list: List[IMDStationData], 
                       variable: str, threshold: float = 3.0) -> List[int]:
        """
        Detect outliers using statistical method (Z-score)
        
        Args:
            data_list: List of station observations
            variable: Variable name to check
            threshold: Z-score threshold for outlier detection
            
        Returns:
            List of indices of detected outliers
        """
        values = [d.observations.get(variable) for d in data_list if variable in d.observations]
        if not values:
            return []
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values_array - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
        
        logger.info(f"Detected {len(outlier_indices)} outliers in {variable}")
        return outlier_indices

class IMDDataIntegrator:
    """Integrate IMD data with satellite and model data"""
    
    def __init__(self):
        self.ingester = IMDDataIngester()
        self.validator = IMDDataValidator()
    
    def create_analysis_grid(self, station_data_dict: Dict[str, List[IMDStationData]], 
                            grid_resolution: float = 0.5) -> pd.DataFrame:
        """
        Interpolate point station data to regular grid for model assimilation
        
        Args:
            station_data_dict: Dictionary of station data
            grid_resolution: Grid spacing in degrees
            
        Returns:
            DataFrame with gridded data
        """
        logger.info(f"Creating analysis grid at {grid_resolution}° resolution")
        
        # Placeholder for spatial interpolation (kriging, IDW, etc.)
        # In production, use scipy.interpolate or similar methods
        
        # Create regular grid over India
        lat = np.arange(8, 35, grid_resolution)
        lon = np.arange(68, 97, grid_resolution)
        
        grid_data = {
            'latitude': lat,
            'longitude': lon,
            'temperature': np.zeros((len(lat), len(lon))),
            'humidity': np.zeros((len(lat), len(lon)))
        }
        
        return pd.DataFrame(grid_data)
