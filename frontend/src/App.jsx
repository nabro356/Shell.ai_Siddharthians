import React, { useState, useEffect } from 'react';
import './index.css';
import AlertTiers from './components/AlertTiers';
import GeospatialMap from './components/GeospatialMap';

function App() {
  const [alerts, setAlerts] = useState({
    tier1: [],
    tier2: [],
    tier3: []
  });
  const [mapData, setMapData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In actual implementation, we will fetch from FastAPI /api/alerts
    // For now we mock the data to build the UI components
    setTimeout(() => {
      setAlerts({
        tier1: [
          { id: 1, disease: 'Dengue', mandal: 'Tirupati Urban', cases: 42, startDate: '2026-04-20', severity: 'Critical Spike' }
        ],
        tier2: [
          { id: 2, disease: 'Malaria', mandal: 'Visakhapatnam', cases: 18, startDate: '2026-04-22', severity: 'Rising Trend' },
          { id: 3, disease: 'Gastroenteritis', mandal: 'Vijayawada', cases: 25, startDate: '2026-04-25', severity: 'Cluster Detected' }
        ],
        tier3: [
          { id: 4, disease: 'Typhoid', mandal: 'Guntur', cases: 8, startDate: '2026-04-26', severity: 'Early Warning' }
        ]
      });
      setMapData([
        { mandal: 'Tirupati Urban', disease: 'Dengue', cases: 42, lat: 13.6288, lng: 79.4192 },
        { mandal: 'Vijayawada', disease: 'Gastroenteritis', cases: 25, lat: 16.5062, lng: 80.6480 },
        { mandal: 'Visakhapatnam', disease: 'Malaria', cases: 18, lat: 17.6868, lng: 83.2185 }
      ]);
      setLoading(false);
    }, 1000);
  }, []);

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1>Disease Outbreak Surveillance</h1>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
            Predictive Epidemiological Intelligence
          </p>
        </div>
        <div style={{ textAlign: 'right', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Real-time Engine Active<br />
          <span style={{ color: '#22c55e' }}>● All Systems Operational</span>
        </div>
      </header>

      {loading ? (
        <div style={{ textAlign: 'center', padding: '5rem', color: 'var(--text-secondary)' }}>
          <div className="spinner"></div> {/* Handled in CSS */}
          <p>Syncing telemetry...</p>
        </div>
      ) : (
        <main>
          <div style={{ marginBottom: '3rem' }}>
            <h2 className="tier-header">📍 Geospatial Spread Prediction</h2>
            <GeospatialMap data={mapData} />
          </div>

          <AlertTiers level="1" name="Tier-I (Critical)" data={alerts.tier1} />
          <AlertTiers level="2" name="Tier-II (Warning)" data={alerts.tier2} />
          <AlertTiers level="3" name="Tier-III (Monitoring)" data={alerts.tier3} />
        </main>
      )}
    </div>
  );
}

export default App;
