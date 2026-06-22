import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './ClimateVisualization.css';

/**
 * Climate Visualization Component
 * Displays real-time climate data and forecasts
 */
const ClimateVisualization = () => {
  const [currentData, setCurrentData] = useState(null);
  const [forecastData, setForecastData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState('india_all');
  const [selectedVariable, setSelectedVariable] = useState('temperature');
  const [error, setError] = useState(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

  // Fetch current climate data
  useEffect(() => {
    const fetchCurrentData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/climate/current`, {
          params: {
            region: selectedRegion,
            variables: selectedVariable
          },
          headers: {
            'Authorization': `Bearer ${process.env.REACT_APP_API_KEY || 'dev_key'}`
          }
        });
        setCurrentData(response.data);
        setError(null);
      } catch (err) {
        setError(`Failed to fetch current data: ${err.message}`);
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCurrentData();
    const interval = setInterval(fetchCurrentData, 3600000); // Update every hour
    return () => clearInterval(interval);
  }, [selectedRegion, selectedVariable]);

  // Fetch forecast data
  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/climate/forecast`, {
          params: {
            days: 7,
            region: selectedRegion,
            ensemble: true
          },
          headers: {
            'Authorization': `Bearer ${process.env.REACT_APP_API_KEY || 'dev_key'}`
          }
        });
        setForecastData(response.data);
      } catch (err) {
        console.error('Error fetching forecast:', err);
      }
    };

    fetchForecast();
    const interval = setInterval(fetchForecast, 86400000); // Update daily
    return () => clearInterval(interval);
  }, [selectedRegion]);

  const regions = [
    { value: 'india_all', label: 'All India' },
    { value: 'monsoon_zone', label: 'Monsoon Zone' },
    { value: 'coastal', label: 'Coastal Regions' }
  ];

  const variables = [
    { value: 'temperature', label: 'Temperature (°C)' },
    { value: 'humidity', label: 'Humidity (%)' },
    { value: 'wind_speed', label: 'Wind Speed (m/s)' },
    { value: 'precipitation', label: 'Precipitation (mm)' }
  ];

  if (loading && !currentData) {
    return <div className="loading">Loading climate data...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  // Prepare forecast plot data
  const forecastPlotData = forecastData ? [
    {
      x: forecastData.forecast_dates,
      y: forecastData.forecast_data[selectedVariable],
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Mean Forecast',
      line: { color: '#1f77b4', width: 2 }
    },
    {
      x: [...forecastData.forecast_dates, ...forecastData.forecast_dates.reverse()],
      y: [
        ...Array(forecastData.forecast_dates.length).fill(0).map((_, i) => 
          forecastData.forecast_data[selectedVariable][i] + parseFloat(forecastData.uncertainty.split(' ')[0])
        ),
        ...Array(forecastData.forecast_dates.length).fill(0).reverse().map((_, i) => 
          forecastData.forecast_data[selectedVariable][forecastData.forecast_dates.length - 1 - i] - parseFloat(forecastData.uncertainty.split(' ')[0])
        )
      ],
      type: 'scatter',
      fill: 'toself',
      name: 'Uncertainty Range',
      line: { color: 'rgba(0,0,0,0)' },
      fillcolor: 'rgba(31, 119, 180, 0.2)'
    }
  ] : [];

  return (
    <div className="climate-visualization">
      <header className="header">
        <h1>🌍 India Climate Digital Twin</h1>
        <p>Real-time Climate Monitoring & Forecasting System</p>
      </header>

      <div className="controls">
        <div className="control-group">
          <label htmlFor="region">Region:</label>
          <select 
            id="region" 
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
          >
            {regions.map(r => (
              <option key={r.value} value={r.value}>{r.label}</option>
            ))}
          </select>
        </div>

        <div className="control-group">
          <label htmlFor="variable">Variable:</label>
          <select 
            id="variable"
            value={selectedVariable}
            onChange={(e) => setSelectedVariable(e.target.value)}
          >
            {variables.map(v => (
              <option key={v.value} value={v.value}>{v.label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="data-display">
        {currentData && (
          <div className="current-data">
            <h2>Current Climate State</h2>
            <div className="data-info">
              <p><strong>Timestamp:</strong> {currentData.timestamp}</p>
              <p><strong>Region:</strong> {currentData.region}</p>
              <p><strong>Data Sources:</strong> {currentData.metadata?.data_sources?.join(', ')}</p>
              <p><strong>Resolution:</strong> {currentData.metadata?.resolution}</p>
            </div>
          </div>
        )}

        {forecastPlotData.length > 0 && (
          <div className="forecast-chart">
            <h2>7-Day Forecast</h2>
            <Plot
              data={forecastPlotData}
              layout={{
                title: `${selectedVariable} Forecast`,
                xaxis: { title: 'Date' },
                yaxis: { title: variables.find(v => v.value === selectedVariable)?.label },
                hovermode: 'x unified',
                margin: { l: 60, r: 60, t: 60, b: 60 }
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '400px' }}
            />
          </div>
        )}
      </div>

      <div className="alerts-section">
        <h2>⚠️ Active Alerts</h2>
        <div className="alerts">
          <div className="alert alert-warning">
            <strong>Monsoon Status:</strong> Active across west coast
          </div>
          <div className="alert alert-info">
            <strong>Weather Forecast:</strong> Heavy rainfall expected in central India
          </div>
        </div>
      </div>

      <footer className="footer">
        <p>Bharatiya Antariksha Hackathon 2026 | AI-Powered Digital Twin of India's Climate</p>
      </footer>
    </div>
  );
};

export default ClimateVisualization;
