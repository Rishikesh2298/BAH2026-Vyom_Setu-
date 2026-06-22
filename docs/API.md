# API Documentation

## India Climate Digital Twin REST API

**Base URL**: `http://localhost:8000/api`

### Authentication

All requests require API key in header:

```
Authorization: Bearer YOUR_API_KEY
```

---

## Climate Data Endpoints

### Get Current Climate State

```
GET /api/climate/current
```

**Query Parameters**:

- `region` (optional): Regional focus (e.g., 'india_all', 'monsoon_zone', 'coastal')
- `variables` (optional): Specific variables (comma-separated)

**Response**:

```json
{
  "timestamp": "2026-06-22T10:30:00Z",
  "region": "india_all",
  "data": {
    "temperature": [[...]],
    "humidity": [[...]],
    "wind_speed": [[...]]
  },
  "metadata": {
    "resolution": "1.0°",
    "data_sources": ["INSAT", "IMD", "ERA5"]
  }
}
```

### Get Weather Forecast

```
GET /api/climate/forecast
```

**Query Parameters**:

- `days` (optional): Forecast horizon (default: 7, max: 15)
- `region` (optional): Regional focus
- `ensemble` (optional): Include ensemble members (boolean)

**Response**:

```json
{
  "forecast_dates": ["2026-06-23", "2026-06-24", ...],
  "forecast_data": {
    "temperature": [...],
    "precipitation": [...],
    "wind": [...]
  },
  "ensemble_members": 50,
  "uncertainty": "1.5°C (std)"
}
```

### Custom Scenario Simulation

```
POST /api/climate/simulate
```

**Request Body**:

```json
{
  "scenario": "monsoon_perturbation",
  "parameters": {
    "sst_anomaly": 0.5,
    "soil_moisture": -10
  },
  "duration_days": 30,
  "ensemble_size": 100
}
```

---

## Monitoring Endpoints

### Monsoon Tracking

```
GET /api/monitoring/monsoon
```

**Response**:

```json
{
  "monsoon_status": "active",
  "onset_date": "2026-06-01",
  "intensity_index": 0.85,
  "rainfall_anomaly": "+15%",
  "affected_regions": ["west_coast", "central_india"]
}
```

### Extreme Weather Alerts

```
GET /api/monitoring/extremes
```

**Query Parameters**:

- `type` (optional): 'cyclone', 'heat_wave', 'cold_wave', 'drought'
- `severity` (optional): 'low', 'moderate', 'high', 'critical'

**Response**:

```json
{
  "active_alerts": [
    {
      "id": "EX_20260622_001",
      "type": "heat_wave",
      "location": "North India",
      "severity": "high",
      "probability": 0.78,
      "expected_duration": "5 days",
      "peak_temperature": "48°C"
    }
  ],
  "update_time": "2026-06-22T10:00:00Z"
}
```

### Drought Monitoring

```
GET /api/monitoring/drought
```

**Response**:

```json
{
  "drought_index": "SPEI",
  "regions": [
    {
      "name": "Rajasthan",
      "severity": "moderate",
      "index_value": -1.2,
      "affected_area_percent": 45
    }
  ],
  "agricultural_impact": "High risk for rabi crops"
}
```

---

## Data Management Endpoints

### Get Available Data Sources

```
GET /api/data/sources
```

**Response**:

```json
{
  "data_sources": [
    {
      "name": "INSAT",
      "type": "satellite",
      "frequency": "hourly",
      "coverage": "India + Indian Ocean",
      "products": ["cloud_cover", "sst", "wind"]
    },
    {
      "name": "IMD",
      "type": "ground_network",
      "frequency": "3-hourly",
      "stations": 2400,
      "variables": ["temperature", "rainfall", "wind"]
    }
  ]
}
```

### Export Simulation Results

```
POST /api/data/export
```

**Request Body**:

```json
{
  "simulation_id": "SIM_20260622_001",
  "format": "netcdf",
  "variables": ["temperature", "precipitation"],
  "time_range": ["2026-06-22", "2026-07-22"]
}
```

---

## Configuration Endpoints

### Set Model Parameters

```
POST /api/config/model
```

**Request Body**:

```json
{
  "model_type": "lstm",
  "parameters": {
    "hidden_size": 256,
    "num_layers": 3,
    "dropout": 0.2
  }
}
```

### Get Model Configuration

```
GET /api/config/model
```

### Set Data Assimilation Parameters

```
POST /api/config/assimilation
```

**Request Body**:

```json
{
  "method": "enkf",
  "ensemble_size": 100,
  "inflation_factor": 1.1,
  "localization_radius": 500
}
```

---

## Error Responses

### 400 Bad Request

```json
{
  "error": "Invalid parameters",
  "details": "region must be one of: india_all, monsoon_zone, coastal"
}
```

### 401 Unauthorized

```json
{
  "error": "Authentication required",
  "message": "Invalid or missing API key"
}
```

### 429 Too Many Requests

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```

### 500 Internal Server Error

```json
{
  "error": "Internal server error",
  "request_id": "REQ_20260622_xyz123"
}
```

---

## Rate Limits

- **Free Tier**: 100 requests/hour
- **Standard**: 1000 requests/hour
- **Premium**: Unlimited

---

## Webhooks

Subscribe to real-time updates:

```
POST /api/webhooks/subscribe
```

**Request Body**:

```json
{
  "event": "extreme_weather_alert",
  "callback_url": "https://your-server.com/webhook",
  "regions": ["north_india", "coastal"],
  "severity_threshold": "high"
}
```

---

## Examples

### Python

```python
import requests

API_KEY = "your_api_key"
BASE_URL = "http://localhost:8000/api"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Get current climate
response = requests.get(
    f"{BASE_URL}/climate/current",
    params={"region": "monsoon_zone"},
    headers=headers
)
data = response.json()
print(data)
```

### cURL

```bash
curl -X GET "http://localhost:8000/api/climate/forecast?days=7" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### JavaScript

```javascript
const API_KEY = "your_api_key";
const BASE_URL = "http://localhost:8000/api";

fetch(`${BASE_URL}/climate/current`, {
  headers: {
    Authorization: `Bearer ${API_KEY}`,
  },
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

---

## OpenAPI Specification

Full API documentation available at: `http://localhost:8000/docs`
(Interactive Swagger UI)

ReDoc documentation: `http://localhost:8000/redoc`
