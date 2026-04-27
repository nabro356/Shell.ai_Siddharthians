import React, { useState } from 'react';

function AlertCard({ alert, level }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`card tier-${level}`} onClick={() => setExpanded(!expanded)}>
      <div className="card-title">
        <span>{alert.disease}</span>
        <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          {expanded ? '▲ Close' : '▼ Expand'}
        </span>
      </div>
      
      <div className="card-meta">
        <div>📍 {alert.mandal}</div>
        <div>👥 {alert.cases} Active Cases</div>
        <div>📅 Onset: {alert.startDate}</div>
      </div>
      
      <div style={{ color: `var(--tier-${level}-border)`, fontWeight: '600', fontSize: '0.95rem' }}>
        ⚠️ {alert.severity}
      </div>

      {expanded && (
        <div className="expanded-content" onClick={(e) => e.stopPropagation()}>
          <div className="stats-grid">
            <div className="stat-box warning">
              <div className="stat-value">Poor</div>
              <div className="stat-label">Water Quality Index</div>
            </div>
            <div className="stat-box">
              <div className="stat-value">32.4°C</div>
              <div className="stat-label">Avg Temp</div>
            </div>
            <div className="stat-box" style={{ borderColor: 'rgba(59, 130, 246, 0.3)', background: 'rgba(59, 130, 246, 0.05)' }}>
              <div className="stat-value">{Math.floor(alert.cases * 0.4)}</div>
              <div className="stat-label">Last Yr Same Wk</div>
            </div>
          </div>
          
          <div className="chart-container">
            <div style={{ textAlign: 'center' }}>
              <span style={{ fontSize: '1.2rem' }}>📈 Prediction Trajectory</span>
              <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                Mandal-level forecast graph will render here using Recharts.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AlertCard;
