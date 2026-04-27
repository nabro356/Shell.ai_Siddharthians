import React from 'react';
import AlertCard from './AlertCard';

function AlertTiers({ level, name, data }) {
  if (!data || data.length === 0) return null;
  
  return (
    <section className={`tier-section tier-${level}`}>
      <div className="tier-header">
        <span className="tier-badge">Tier {level}</span>
        <span className="tier-label">{name}</span>
      </div>
      <div className="alert-grid">
        {data.map(alert => (
          <AlertCard key={alert.id} alert={alert} level={level} />
        ))}
      </div>
    </section>
  );
}

export default AlertTiers;
