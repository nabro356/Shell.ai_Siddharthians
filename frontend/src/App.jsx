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
    // Fetch live generated inputs from the python pipeline outputs via public assets
    Promise.all([
      fetch('/alerts.json').then(res => res.ok ? res.json() : null),
      fetch('/spread_map.json').then(res => res.ok ? res.json() : null)
    ])
    .then(([alertsData, mapRes]) => {
      if (alertsData) {
        setAlerts(alertsData);
      } else {
        console.warn("alerts.json not found, pipeline might not have generated alerts");
      }
      if (mapRes) {
        setMapData(mapRes);
      }
      setLoading(false);
    })
    .catch(err => {
      console.error("Error loading frontend assets:", err);
      setLoading(false);
    });
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
