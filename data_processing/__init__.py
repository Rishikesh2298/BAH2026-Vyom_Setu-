"""
Data Processing Package
Handles data ingestion, validation, and preprocessing
"""

__version__ = "1.0.0"

from .satellite_ingester import (
    INSATDataIngester,
    OceansatDataIngester,
    SatelliteDataProcessor,
    SatelliteData
)

from .imd_ingester import (
    IMDDataIngester,
    IMDDataValidator,
    IMDDataIntegrator,
    IMDStationData
)

__all__ = [
    'INSATDataIngester',
    'OceansatDataIngester',
    'SatelliteDataProcessor',
    'SatelliteData',
    'IMDDataIngester',
    'IMDDataValidator',
    'IMDDataIntegrator',
    'IMDStationData'
]
