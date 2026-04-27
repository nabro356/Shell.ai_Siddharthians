import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css'; 

function GeospatialMap({ data }) {
  // Center roughly on Andhra Pradesh coordinates
  const apCenter = [16.5062, 80.6480];
  
  const validData = data ? data.filter(point => point.lat != null && point.lng != null) : [];
  
  if (validData.length === 0) {
    return (
      <div style={{ color: 'var(--text-secondary)', padding: '2rem', textAlign: 'center', border: '1px dashed rgba(255,255,255,0.1)', borderRadius: '12px' }}>
        No geospatial data coordinates available for compiling the map. (Alerts are still active below)
      </div>
    );
  }

  return (
    <div style={{ height: '400px', width: '100%', borderRadius: 'var(--border-radius)', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
      <MapContainer center={apCenter} zoom={6} scrollWheelZoom={false} style={{ height: '100%', width: '100%', background: '#0f172a' }}>
        {/* CARTO Dark Matter styling for premium aesthetic */}
        <TileLayer
          attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        {validData.map((point, idx) => (
          <CircleMarker 
             key={idx}
             center={[point.lat, point.lng]}
             pathOptions={{ 
               fillColor: point.cases > 20 ? '#ef4444' : '#f59e0b',
               color: 'transparent',
               fillOpacity: 0.6
             }}
             radius={Math.max(5, Math.min(20, point.cases * 0.5))}
          >
            <Popup>
              <div style={{ color: '#000', fontSize: '0.9rem' }}>
                <strong>{point.mandal}</strong><br/>
                Disease: {point.disease}<br/>
                Cases Detected: <strong>{point.cases}</strong>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
}

export default GeospatialMap;
