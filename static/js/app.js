// Add this new helper function right after your handleFileUpload function
const resolveDigipins = async (locationsWithDigipins) => {
  const promises = locationsWithDigipins.map(async (loc) => {
      try {
          const response = await fetch('/resolve_digipin', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ digipin: loc.digipin })
          });
          if (!response.ok) return null; // Failed to resolve
          const data = await response.json();
          return {
              ...loc,
              coords: data.coords,
              name: await reverseGeocode(data.coords[0], data.coords[1])
          };
    } catch (error) {
          return null; // Network or other error
      }
  });

  const resolved = await Promise.all(promises);
  return resolved.filter(Boolean); // Filter out any nulls from failed requests
};


// REPLACE your existing handleFileUpload function with this new version
const handleFileUpload = (event) => {
  const file = (event && event.target && event.target.files) ? event.target.files[0] : null;
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = async (e) => {
      try {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          const sheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[sheetName];
          const json = XLSX.utils.sheet_to_json(worksheet);

          const locationsWithAddress = [];
          const locationsWithDigipin = [];

          // Separate locations based on whether they have a Digipin or an Address
          json.forEach(row => {
              const digipin = row['Digipin'] || row['digipin'];
              const address = row['Order Location'] || row['Location'] || row['Address'];

              if (digipin) {
                  locationsWithDigipin.push({
                      id: row['Order ID'] || Date.now() + Math.random(),
                      name: row['Order Name'] || `From Digipin ${digipin}`,
                      digipin: String(digipin).trim().toUpperCase(),
                      coords: null
                  });
              } else if (address) {
                  locationsWithAddress.push({
                      id: row['Order ID'] || Date.now() + Math.random(),
                      name: row['Order Name'] || 'Unknown',
                      address: address,
                      coords: null
                  });
              }
          });

          if (locationsWithAddress.length === 0 && locationsWithDigipin.length === 0) {
              showToast('No valid locations found in the Excel file. Check column names.', 'error');
      return;
    }

          const totalLocations = locationsWithAddress.length + locationsWithDigipin.length;
          if (totalLocations > 8) {
              showToast(`Too many locations (${totalLocations}). The maximum is 8 for this demo.`, 'error');
      return;
    }

          setIsGeocoding(true); // Show a generic "Processing..." state
          let finalLocations = [];

          // Process both lists in parallel
          const [resolvedFromDigipin, geocodedFromAddress] = await Promise.all([
              resolveDigipins(locationsWithDigipin),
              geocodeLocations(locationsWithAddress) // This function needs a small change too
          ]);
          
          finalLocations = [...resolvedFromDigipin, ...geocodedFromAddress];
          
          // Render markers for all processed points
          if (mapInstanceRef.current && window.L) {
              // Clear old markers
              markersRef.current.forEach(({ marker }) => { try { mapInstanceRef.current.removeLayer(marker); } catch(e){} });
              markersRef.current = [];
              // Add new markers
              finalLocations.forEach(loc => {
                  const popupContent = `<strong>${loc.name}</strong><br/>${loc.digipin ? `Digipin: ${loc.digipin}` : loc.address}`;
                  const m = window.L.marker(loc.coords).bindPopup(popupContent).addTo(mapInstanceRef.current);
                  markersRef.current.push({ marker: m, id: loc.id });
              });
              // Fit map to new bounds
              if (finalLocations.length > 0) {
                  try { mapInstanceRef.current.fitBounds(window.L.latLngBounds(finalLocations.map(l => l.coords)), { padding: [50, 50] }); } catch(e){}
              }
          }
          
          setSelectedLocations(finalLocations);
          showToast(`Successfully processed ${finalLocations.length} locations.`, 'success');
    } catch (error) {
          showToast('Error reading Excel file. Please check the file format.', 'error');
          console.error(error);
    } finally {
          setIsGeocoding(false);
      }
  };
  reader.readAsArrayBuffer(file);
};


// REPLACE your existing geocodeLocations function with this to make it return the results
const geocodeLocations = async (locations) => {
  if (!locations || locations.length === 0) return []; // Return empty array if no addresses
  try {
      const addresses = locations.map(l => l.address);
      const resp = await fetch('/geocode', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ addresses })
      });
      if (!resp.ok) throw new Error('Failed to geocode');
      const data = await resp.json();
      const geo = data.locations || [];
      const merged = locations.map((l, i) => {
          const gi = geo[i] || {};
          const nextName = gi && gi.name ? shortenPlaceName(gi.name) : shortenPlaceName(l.name);
          const nextCoords = gi && Array.isArray(gi.coords) ? gi.coords : null;
          return {
              ...l,
              name: nextName,
              coords: nextCoords
          };
      }).filter(l => Array.isArray(l.coords));
      return merged; // Return the geocoded locations
  } catch (e) {
      showToast('Geocoding failed for some addresses.', 'error');
      return []; // Return empty on failure
  }
};