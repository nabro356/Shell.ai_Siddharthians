import React, { useState } from 'react';

function AlertCard({ alert, level }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      {/* Standard Card View */}
      <div className={`card tier-${level}`} onClick={() => setExpanded(true)}>
        <div className="card-title">
          <span>{alert.disease}</span>
        </div>
        
        <div className="card-meta">
          <div>📍 {alert.mandal}</div>
          <div>👥 {alert.cases} Active Cases</div>
          {alert.startDate && <div>📅 Onset: {alert.startDate}</div>}
        </div>
        
        <div style={{ color: `var(--tier-${level}-text)`, fontWeight: '600', fontSize: '0.95rem' }}>
          ⚠️ {alert.severity}
        </div>
      </div>

      {/* Expanded Modal Overlay */}
      {expanded && (
        <div className="modal-overlay" onClick={() => setExpanded(false)}>
          <div className={`modal-content tier-${level}`} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2 style={{ color: `var(--tier-${level}-text)` }}>{alert.disease} Outbreak</h2>
              <button className="close-btn" onClick={() => setExpanded(false)}>✕</button>
            </div>
            
            <div className="modal-layout">
              <div className="modal-main-stats">
                <h3>📍 {alert.mandal}</h3>
                <p><strong>Cases Detected:</strong> {alert.cases}</p>
                <p><strong>Onset Date:</strong> {alert.startDate}</p>
                <p><strong>Severity Flag:</strong> {alert.severity}</p>
              </div>

              <div className="facilities-box">
                <h4>🏥 Primary Facilities (This Week)</h4>
                <p>{alert.facilities && alert.facilities !== "" ? alert.facilities : "No isolated facility clusters identified/captured."}</p>
              </div>

              <div className="chart-container">
                <div style={{ textAlign: 'center', width: '100%' }}>
                  <span style={{ fontSize: '1.2rem', color: 'var(--text-primary)' }}>📈 Prediction Trajectory</span>
                  <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Mandal-level forecast graph and ML trends will render here dynamically.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default AlertCard;
