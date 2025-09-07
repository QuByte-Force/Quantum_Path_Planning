import React, { useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, Marker, Polyline, useMapEvents, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

/**
 * Quantum Path Planning — single-file React UI
 * - Landing → Mode Select → Excel Flow / Custom Flow
 * - Light, clean design, Inter font, subtle animations
 * - Sidebar + Modal comparison (Greedy vs QAOA)
 * - Works with your Flask endpoints: /upload_excel, /resolve_digipin, /solve_tsp
 *
 * Notes:
 * - Requires Tailwind CSS (class names assume Tailwind)
 * - Requires react-leaflet & leaflet
 */

// ------------------------------
// Utilities
// ------------------------------

const cn = (...c) => c.filter(Boolean).join(" ");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Feather-ish inline icons
const Icon = ({ name, className = "w-5 h-5" }) => {
  const paths = {
    rocket: (
      <>
        <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.3.05-3.18-.65-.87-2.2-.86-3.18.05Z" />
        <path d="m12 15-3-3a9 9 0 0 1 3-7 9 9 0 0 1 7 3l-3 3" />
        <path d="m9 9 3 3" />
        <path d="m12 12 3-3" />
      </>
    ),
    upload: <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5-5 5 5M12 5v14" />, 
    mapPin: (
      <>
        <path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 0 1 16 0Z" />
        <circle cx="12" cy="10" r="3" />
      </>
    ),
    route: (
      <>
        <path d="M17 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z" />
        <path d="M12 18h.01" />
        <path d="M12 15h.01" />
        <path d="M12 12h.01" />
        <path d="M12 9h.01" />
      </>
    ),
    x: (
      <>
        <path d="M18 6 6 18" />
        <path d="m6 6 12 12" />
      </>
    ),
    car: (
      <>
        <path d="M14 16.5 17.5 13H20v-2h-1.6l-3.2-4.4A2 2 0 0 0 13.5 6H5a2 2 0 0 0-2 2v7a1 1 0 0 0 1 1h1" />
        <circle cx="7.5" cy="16.5" r="2.5" />
        <circle cx="14.5" cy="16.5" r="2.5" />
      </>
    ),
    plus: (
      <>
        <path d="M5 12h14" />
        <path d="M12 5v14" />
      </>
    ),
    loader: <path d="M21 12a9 9 0 1 1-6.219-8.56" />, 
    list: (
      <>
        <path d="M8 6h13" />
        <path d="M8 12h13" />
        <path d="M8 18h13" />
        <path d="M3 6h.01" />
        <path d="M3 12h.01" />
        <path d="M3 18h.01" />
      </>
    ),
  };
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      {paths[name]}
    </svg>
  );
};

const Spinner = ({ className = "w-4 h-4" }) => (
  <Icon name="loader" className={cn(className, "animate-spin")}/>
);

// ------------------------------
// Toast
// ------------------------------

const Toast = ({ message, type = "success", onClose }) => {
  if (!message) return null;
  const bg = type === "error" ? "bg-red-500" : type === "warn" ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className={cn("fixed bottom-5 right-5 text-white px-4 py-3 rounded-xl shadow-lg flex items-center gap-3 z-50", bg)}>
      <span className="text-sm">{message}</span>
      <button onClick={onClose} className="p-1/2 rounded hover:bg-white/25">
        <Icon name="x" />
      </button>
    </div>
  );
};

// ------------------------------
// Results Modal
// ------------------------------

const ResultsModal = ({ open, onClose, results, locations }) => {
  if (!open || !results) return null;

  const getName = (idx) => {
    if (idx === 0) return "Start / Depot";  // Backend uses 0 for depot
    return locations[idx - 1]?.name || `Location ${idx}`;  // Backend indices start from 1 for actual locations
  };

    const Card = ({ plan, title, accent }) => (
    <div className="bg-slate-800 border border-slate-600 rounded-2xl p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className={cn("w-9 h-9 rounded-full grid place-items-center", accent)}>
            <Icon name="route" />
          </span>
          <h3 className="font-semibold text-slate-100">{title}</h3>
        </div>
        <div className="text-right text-slate-100">
          <div className="text-xl font-bold">{plan.total_length_km.toFixed(2)} km</div>
          <div className="text-xs text-slate-400">
            {plan.total_travel_time ? `${plan.total_travel_time} • ` : ""}{plan.num_vehicles_used || 1} vehicle{(plan.num_vehicles_used || 1) > 1 ? 's' : ''}
          </div>
        </div>
      </div>
      <div className="space-y-3 max-h-64 overflow-auto pr-1">
        {plan.routes.map((r, i) => (
          <div key={i} className="border border-slate-600 rounded-xl p-3">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
                <Icon name="car" className="w-4 h-4" /> 
                Vehicle {r.vehicle_id || (i + 1)}
              </div>
              <div className="text-right text-xs text-slate-400">
                <div>{r.length_km.toFixed(2)} km</div>
                {r.travel_time && <div>{r.travel_time}</div>}
                {r.num_locations && <div>{r.num_locations} stops</div>}
              </div>
            </div>
            <ol className="list-decimal list-inside text-sm text-slate-300 space-y-0.5 mb-3">
              {r.tour.map((idx, k) => (
                <li key={k}>{getName(idx)}</li>
              ))}
            </ol>
            <button 
              className="w-full px-3 py-2 bg-slate-700 hover:bg-slate-600 border border-slate-500 rounded-lg text-xs text-slate-200 transition-colors"
              onClick={() => {
                // Show route details for this vehicle
                alert(`Vehicle ${r.vehicle_id || (i + 1)} Route Details:

Distance: ${r.length_km.toFixed(2)} km
Travel Time: ${r.travel_time || 'N/A'}
Stops: ${r.num_locations || r.tour.length - 1}

Route: ${r.tour.map(idx => getName(idx)).join(' → ')}`);
              }}
            >
              <Icon name="mapPin" className="w-3 h-3 inline mr-1" />
              View Route Details
            </button>
          </div>
        ))}
      </div>
    </div>
  );

  const better = results.qaoa_vrp.total_length_km < results.greedy_vrp.total_length_km;
  const pct = Math.abs(
    100 * (1 - (better
      ? results.qaoa_vrp.total_length_km / results.greedy_vrp.total_length_km
      : results.greedy_vrp.total_length_km / results.qaoa_vrp.total_length_km))
  ).toFixed(2);

  return (
    <>
      <style>
        {`
          .leaflet-container {
            display: none !important;
          }
        `}
      </style>
      <div className="fixed inset-0 bg-slate-900 grid place-items-center p-4 z-[9999]" onClick={onClose}>
        <div className="bg-slate-700 rounded-3xl shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-auto p-6 relative z-[10000]" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-full bg-emerald-500/20 text-emerald-400 grid place-items-center">
              <Icon name="rocket" className="w-6 h-6" />
            </span>
            <h2 className="text-2xl font-bold text-slate-100">Algorithm Comparison</h2>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-600 text-slate-300">
            <Icon name="x" className="w-6 h-6" />
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card plan={results.greedy_vrp} title="Greedy" accent="bg-amber-500/20 text-amber-400" />
          <Card plan={results.qaoa_vrp} title="Quantum-Inspired (QAOA)" accent="bg-emerald-500/20 text-emerald-400" />
        </div>

        <div className="mt-6 p-4 bg-slate-800 rounded-2xl border border-slate-600 text-center">
          <p className={cn("text-lg font-semibold", better ? "text-emerald-400" : "text-amber-400") }>
            {better ? `QAOA is ${pct}% shorter` : `Greedy is ${pct}% shorter`}
          </p>
          {results.greedy_vrp.num_vehicles_used > 1 && (
            <p className="text-sm text-slate-400 mt-1">
              Using {results.greedy_vrp.num_vehicles_used} vehicles for optimal delivery
            </p>
          )}
        </div>
        </div>
      </div>
    </>
  );
};

// ------------------------------
// Map
// ------------------------------

const RouteColors = {
  greedy: ["#f59e0b", "#d97706", "#b45309"], // amber-500, 600, 700
  qaoa: ["#10b981", "#059669", "#047857"], // emerald-500, 600, 700
};

const depotIcon = new L.divIcon({
  html: `<div class="text-3xl">🚚</div>`,
  className: "", // remove default background
  iconSize: [30, 42],
  iconAnchor: [15, 42],
});

const FitToMarkers = ({ markers }) => {
  const map = useMap();
  useEffect(() => {
    if (!markers?.length) return;
    const bounds = markers.map((m) => m.coords);
    // Leaflet LatLngBounds expects [lat, lng]
    if (bounds.length === 1) {
      map.setView(bounds[0], 15);
    } else {
      map.fitBounds(bounds, { padding: [32, 32] });
    }
  }, [markers, map]);
  return null;
};

const MapPick = ({
  locations,
  routes,
  onMapClick, // optional
  startDepot,
}) => {
  // Map click handler (for custom flow)
  const MapClicks = () => {
    useMapEvents({
      click: (e) => onMapClick?.([e.latlng.lat, e.latlng.lng]),
    });
    return null;
  };

  if (!startDepot?.coords) {
    return (
      <div className="h-full w-full rounded-2xl border border-slate-600 grid place-items-center bg-slate-800 text-center">
        <div>
          <Spinner className="w-8 h-8 text-slate-400 mx-auto" />
          <p className="text-sm text-slate-400 mt-2">Loading Depot Location...</p>
        </div>
      </div>
    );
  }

  return (
    <MapContainer
      center={startDepot.coords}
      zoom={13}
      className="h-full w-full rounded-2xl border border-slate-600"
      style={{ background: "#1e293b" }}
    >
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
      />

      {/* Start / Depot marker (distinct) */}
      <Marker position={startDepot.coords} icon={depotIcon}>
      </Marker>

      {/* User locations */}
      {locations.map((loc, i) => (
        <Marker key={loc.id ?? i} position={loc.coords}>
        </Marker>
      ))}

      {/* Draw routes */}
      {routes?.greedy_vrp?.routes?.map((r, i) => (
        <Polyline
          key={`g-${i}`}
          positions={r.tour.map((idx) => (idx === 0 ? startDepot.coords : locations[idx - 1]?.coords)).filter(Boolean)}
          pathOptions={{ color: RouteColors.greedy[i % RouteColors.greedy.length], weight: 4, opacity: 0.8 }}
        />
      ))}

      {routes?.qaoa_vrp?.routes?.map((r, i) => {
        const pts = r.tour
          .map((idx) => (idx === 0 ? startDepot.coords : locations[idx - 1]?.coords))
          .filter(Boolean)
          .map(([lat, lng]) => [lat + 0.00012, lng + 0.00012]);
        return (
          <Polyline
            key={`q-${i}`}
            positions={pts}
            pathOptions={{ color: RouteColors.qaoa[i % RouteColors.qaoa.length], weight: 4, opacity: 0.9, dashArray: "6 10" }}
          />
        );
      })}

      <FitToMarkers markers={[startDepot, ...locations]} />
      {onMapClick && <MapClicks />}
    </MapContainer>
  );
};

// ------------------------------
// Shared Sidebar
// ------------------------------

const Sidebar = ({
  title,
  locations,
  setLocations,
  onCompare,
  isBusy,
  extra,
  numVehicles,
  setNumVehicles,
}) => {
  const remove = (id) => setLocations((arr) => arr.filter((x) => x.id !== id));

  const maxVehicles = Math.max(1, locations.length);
  const validVehicles = Math.min(Math.max(1, numVehicles), maxVehicles);

  // Update numVehicles if it exceeds locations
  React.useEffect(() => {
    if (numVehicles > maxVehicles) {
      setNumVehicles(maxVehicles);
    }
  }, [locations.length, numVehicles, setNumVehicles, maxVehicles]);

  return (
    <aside className="w-full md:w-96 shrink-0 space-y-4">
      <div className="bg-slate-700 border border-slate-600 rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <span className="w-9 h-9 rounded-full bg-emerald-500/20 text-emerald-400 grid place-items-center">
            <Icon name="list" />
          </span>
          <h3 className="font-semibold text-slate-100">{title}</h3>
        </div>

        <div className="max-h-[32vh] overflow-auto space-y-2 pr-1">
          {locations.length === 0 ? (
            <div className="text-sm text-slate-400">No locations yet.</div>
          ) : (
            locations.map((loc, i) => (
              <div key={loc.id ?? i} className="flex items-start justify-between gap-3 border border-slate-600 rounded-xl p-3">
                <div>
                  <div className="font-medium text-slate-100">{loc.name || `Location ${i + 1}`}</div>
                  <div className="text-xs text-slate-400">{loc.digipin || "—"}</div>
                  <div className="text-[11px] text-slate-400">{loc.coords[0].toFixed(5)}, {loc.coords[1].toFixed(5)}</div>
                </div>
                <button className="text-slate-400 hover:text-red-400" onClick={() => remove(loc.id)}>
                  <Icon name="x" />
                </button>
              </div>
            ))
          )}
        </div>

        {/* Vehicle Selection */}
        {locations.length > 0 && (
          <div className="mt-4 p-3 bg-slate-800 border border-slate-600 rounded-xl">
            <div className="flex items-center gap-2 mb-2">
              <Icon name="car" className="w-4 h-4 text-emerald-400" />
              <label className="text-sm font-medium text-slate-100">Number of Vehicles</label>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setNumVehicles(Math.max(1, numVehicles - 1))}
                disabled={numVehicles <= 1}
                className="w-8 h-8 rounded-lg bg-slate-600 hover:bg-slate-500 disabled:opacity-50 disabled:cursor-not-allowed text-slate-100 font-bold"
              >
                −
              </button>
              <div className="flex-1 text-center">
                <span className="text-lg font-bold text-slate-100">{validVehicles}</span>
                <div className="text-xs text-slate-400">Max: {maxVehicles}</div>
              </div>
              <button
                onClick={() => setNumVehicles(Math.min(maxVehicles, numVehicles + 1))}
                disabled={numVehicles >= maxVehicles}
                className="w-8 h-8 rounded-lg bg-slate-600 hover:bg-slate-500 disabled:opacity-50 disabled:cursor-not-allowed text-slate-100 font-bold"
              >
                +
              </button>
            </div>
            <div className="mt-2 text-xs text-slate-400 text-center">
              {validVehicles === 1 ? "Single vehicle (TSP)" : `${validVehicles} vehicles (VRP)`}
            </div>
          </div>
        )}

        <div className="mt-4 flex gap-2">
          <button
            onClick={onCompare}
            disabled={isBusy || locations.length < 1}
            className={cn(
              "inline-flex items-center justify-center gap-2 px-4 py-2 rounded-xl font-semibold shadow-sm border",
              locations.length < 1 ? "bg-slate-600 text-slate-400 border-slate-500" : "bg-emerald-500 text-white border-emerald-500 hover:bg-emerald-600"
            )}
          >
            {isBusy ? <Spinner /> : <Icon name="route" />} Compare Algorithms
          </button>
          {extra}
        </div>
      </div>
    </aside>
  );
};

// ------------------------------
// Excel Flow
// ------------------------------

const ExcelFlow = ({ goBack, startDepot }) => {
  const [locations, setLocations] = useState([]);
  const [routes, setRoutes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [numVehicles, setNumVehicles] = useState(1);
  const [toast, setToast] = useState({ message: "", type: "success" });

  const fileInput = useRef(null);

  const showToast = (message, type = "success") => {
    setToast({ message, type });
    setTimeout(() => setToast({ message: "", type: "success" }), 3800);
  };

  const onDrop = async (f) => {
    if (!f) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", f);
      const res = await fetch("http://localhost:5000/upload_excel", { method: "POST", body: form });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Upload failed");
      const newLocs = json.data.map((item, idx) => ({
        id: item.id || `xl-${Date.now()}-${idx}`,
        name: item.name,
        coords: item.coordinates,
        digipin: item.digipin,
      }));
      setLocations(newLocs);
      setNumVehicles(1); // Reset to 1 vehicle when new file is uploaded
      showToast(json.message || "Imported successfully");
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  const handleCompare = async () => {
    if (!locations.length || !startDepot?.coords) return;
    setBusy(true);
    try {
      const res = await fetch("http://localhost:5000/solve_tsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          locations: locations.map((l) => ({ coordinates: l.coords, name: l.name })),
          num_vehicles: numVehicles,
          start_location: startDepot.coords,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to solve");
      setRoutes(json);
      setOpen(true);
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  const [open, setOpen] = useState(false);

  return (
    <div className="flex flex-col md:flex-row gap-4 h-full">
      <Sidebar
        title="Extracted Locations"
        locations={locations}
        setLocations={setLocations}
        numVehicles={numVehicles}
        setNumVehicles={setNumVehicles}
        onCompare={handleCompare}
        isBusy={busy}
        extra={
          <button onClick={goBack} className="px-4 py-2 rounded-xl font-semibold border border-slate-600 bg-slate-700 hover:bg-slate-600 text-slate-100">
            Back
          </button>
        }
      />

      <div className="flex-1 grid grid-rows-[auto_1fr] gap-4">
        {/* Upload card */}
        <div className="bg-slate-700 border border-slate-600 rounded-2xl p-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-9 h-9 rounded-full bg-emerald-500/20 text-emerald-400 grid place-items-center"><Icon name="upload"/></span>
              <h3 className="font-semibold text-slate-100">Upload Excel (.xlsx, .xls, .csv)</h3>
            </div>
            <div className="text-sm text-slate-400">Depot: <span className="font-medium text-slate-100">{startDepot.name}</span></div>
          </div>

          <div className="mt-4">
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                onDrop(f);
              }}
              className={cn(
                "border-2 border-dashed border-slate-500 rounded-2xl p-8 text-center",
                busy ? "opacity-70" : "hover:bg-slate-800"
              )}
            >
              <p className="text-sm text-slate-300">Drag & drop your file here</p>
              <div className="my-2 text-xs text-slate-500">or</div>
              <button
                onClick={() => fileInput.current?.click()}
                className="px-4 py-2 rounded-xl bg-emerald-500 text-white font-semibold hover:bg-emerald-600"
                disabled={busy}
              >
                {busy ? <span className="inline-flex items-center gap-2"><Spinner/> Processing…</span> : "Choose File"}
              </button>
              <input
                ref={fileInput}
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => onDrop(e.target.files?.[0])}
                className="hidden"
              />
            </div>
          </div>
        </div>

        {/* Map */}
        <div className="relative min-h-0">
          <MapPick locations={locations} routes={routes} startDepot={startDepot} />
        </div>
      </div>

      <Toast message={toast.message} type={toast.type} onClose={() => setToast({ message: "", type: "success" })} />
      <ResultsModal open={open} onClose={() => setOpen(false)} results={routes} locations={locations} />
    </div>
  );
};

// ------------------------------
// Custom Flow (select on map or via Digipins)
// ------------------------------

const CustomFlow = ({ goBack, startDepot }) => {
  const [locations, setLocations] = useState([]);
  const [routes, setRoutes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [numVehicles, setNumVehicles] = useState(1);
  const [digipin, setDigipin] = useState("");
  const [toast, setToast] = useState({ message: "", type: "success" });

  const showToast = (message, type = "success") => {
    setToast({ message, type });
    setTimeout(() => setToast({ message: "", type: "success" }), 3600);
  };

  const addByMap = async (coords) => {
  setBusy(true);
  try {
    // Call backend to get Digipin for clicked coordinates
    const res = await fetch("http://localhost:5000/encode_coordinates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat: coords[0], lon: coords[1] }),
    });
    const json = await res.json();

    if (!res.ok) throw new Error(json.error || "Failed to encode coordinates");

    // Add new location with Digipin
    setLocations((arr) => [
      ...arr,
      {
        id: `mp-${Date.now()}`,
        name: `Pinned @ ${coords[0].toFixed(4)}, ${coords[1].toFixed(4)}`,
        coords,
        digipin: json.digipin, // <-- now Digipin is included
      },
    ]);
  } catch (e) {
    console.error(e);
    showToast(e.message, "error");
  } finally {
    setBusy(false);
  }
};


  const addByDigipin = async (e) => {
    e.preventDefault();
    if (!digipin.trim()) return;
    setBusy(true);
    try {
      const res = await fetch("http://localhost:5000/resolve_digipin", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ digipin }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Invalid Digipin");
      setLocations((arr) => [
        ...arr,
        { id: `dp-${Date.now()}`, name: `Location ${arr.length + 1}`, coords: json.coords, digipin: json.digipin },
      ]);
      setDigipin("");
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  const handleCompare = async () => {
    if (!locations.length || !startDepot?.coords) return;
    setBusy(true);
    try {
      const res = await fetch("http://localhost:5000/solve_tsp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          locations: locations.map((l) => ({ coordinates: l.coords, name: l.name })),
          num_vehicles: numVehicles,
          start_location: startDepot.coords,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to solve");
      setRoutes(json);
      setOpen(true);
    } catch (e) {
      showToast(e.message, "error");
    } finally {
      setBusy(false);
    }
  };

  const [open, setOpen] = useState(false);

  return (
    <div className="flex flex-col md:flex-row gap-4 h-full">
      <Sidebar
        title="Selected Locations"
        locations={locations}
        setLocations={setLocations}
        numVehicles={numVehicles}
        setNumVehicles={setNumVehicles}
        onCompare={handleCompare}
        isBusy={busy}
        extra={
          <button onClick={goBack} className="px-4 py-2 rounded-xl font-semibold border border-slate-600 bg-slate-700 hover:bg-slate-600 text-slate-100">
            Back
          </button>
        }
      />

      <div className="flex-1 grid grid-rows-[auto_auto_1fr] gap-4">
        <div className="bg-slate-700 border border-slate-600 rounded-2xl p-5">
          <div className="flex items-center gap-2">
            <span className="w-9 h-9 rounded-full bg-amber-500/20 text-amber-400 grid place-items-center"><Icon name="mapPin"/></span>
            <h3 className="font-semibold text-slate-100">Add by clicking on map</h3>
          </div>
          <p className="text-sm text-slate-400 mt-1">Click anywhere on the map to drop a location.</p>
        </div>

        <div className="bg-slate-700 border border-slate-600 rounded-2xl p-5">
          <form onSubmit={addByDigipin} className="flex items-center gap-2">
            <input
              value={digipin}
              onChange={(e) => setDigipin(e.target.value)}
              placeholder="Enter Digipin (e.g., 47T-886-93P7)"
              className="flex-1 px-3 py-2 border border-slate-600 bg-slate-800 text-slate-100 rounded-xl focus:ring-2 focus:ring-emerald-500 outline-none"
            />
            <button
              type="submit"
              disabled={busy || !digipin.trim()}
              className={cn(
                "px-4 py-2 rounded-xl font-semibold",
                !digipin.trim() ? "bg-slate-600 text-slate-400" : "bg-emerald-500 text-white hover:bg-emerald-600"
              )}
            >
              {busy ? <span className="inline-flex items-center gap-2"><Spinner/> Adding…</span> : <span className="inline-flex items-center gap-2"><Icon name="plus"/> Add</span>}
            </button>
          </form>
        </div>

        <div className="relative min-h-0">
          <MapPick locations={locations} routes={routes} onMapClick={addByMap} startDepot={startDepot} />
        </div>
      </div>

      <Toast message={toast.message} type={toast.type} onClose={() => setToast({ message: "", type: "success" })} />
      <ResultsModal open={open} onClose={() => setOpen(false)} results={routes} locations={locations} />
    </div>
  );
};

// ------------------------------
// Mode Select
// ------------------------------

const ModeSelect = ({ onPick, onBack }) => {
  return (
    <div className="max-w-5xl mx-auto px-4 py-10 text-slate-100">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <span className="w-10 h-10 rounded-2xl bg-emerald-500/20 text-emerald-400 grid place-items-center"><Icon name="rocket" className="w-6 h-6"/></span>
          <h2 className="text-2xl font-bold text-slate-100">Choose a mode</h2>
        </div>
        <button onClick={onBack} className="px-4 py-2 rounded-xl font-semibold border border-slate-600 bg-slate-700 hover:bg-slate-600 text-slate-100">Back</button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <button
          onClick={() => onPick("excel")}
          className="group bg-slate-700 border border-slate-600 rounded-2xl p-6 text-left hover:border-slate-500 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-full bg-emerald-500/20 text-emerald-400 grid place-items-center"><Icon name="upload"/></span>
            <h3 className="text-lg font-semibold text-slate-100">Import from Excel</h3>
          </div>
          <p className="mt-2 text-sm text-slate-300">Automatically extract Digipins & locations, preview on map, and compare algorithms.</p>
          <div className="mt-4 text-emerald-400 font-medium group-hover:underline">Open flow →</div>
        </button>

        <button
          onClick={() => onPick("custom")}
          className="group bg-slate-700 border border-slate-600 rounded-2xl p-6 text-left hover:border-slate-500 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-full bg-amber-500/20 text-amber-400 grid place-items-center"><Icon name="mapPin"/></span>
            <h3 className="text-lg font-semibold text-slate-100">Custom Location Selection</h3>
          </div>
          <p className="mt-2 text-sm text-slate-300">Click on the map or enter Digipins manually. Perfect for ad‑hoc trips.</p>
          <div className="mt-4 text-emerald-400 font-medium group-hover:underline">Open flow →</div>
        </button>
      </div>
    </div>
  );
};

// ------------------------------
// Landing
// ------------------------------

const Landing = ({ onStart }) => {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-slate-800">
      <div className="text-center px-6">
        <span className="px-4 py-1 text-sm rounded-full bg-emerald-500/20 text-emerald-400">
          Quantum • VRP / TSP • Greedy vs QAOA
        </span>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mt-6 text-slate-100">
          Quantum Path Planning
        </h1>
        <p className="mt-4 text-lg md:text-xl text-slate-300 max-w-2xl mx-auto">
          Smarter travel with quantum-inspired optimization. Import from Excel or select locations
          on a live map, then compare routes instantly.
        </p>
        <button
          onClick={onStart}
          className="mt-8 px-8 py-3 bg-emerald-500 text-white rounded-xl shadow-lg hover:bg-emerald-600 transition"
        >
          Start
        </button>
      </div>
    </div>
  );
};

// ------------------------------
// App Shell
// ------------------------------

export default function App() {
  const [page, setPage] = useState("landing"); // landing | pick | excel | custom
  const [startDepot, setStartDepot] = useState({
    name: "SRKR pakka delivery",
    digipin: "47T-886-93P7",
    coords: null, // Coords will be fetched from the backend
  });

  // --- Fetch accurate depot location on load ---
  useEffect(() => {
    const fetchDepotCoords = async () => {
      try {
        const res = await fetch(`http://localhost:5000/resolve_digipin`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ digipin: "47T-886-93P7" }),
        });
        const json = await res.json();
        if (res.ok) {
          setStartDepot((prev) => ({ ...prev, coords: json.coords }));
        } else {
          throw new Error(json.error || "Failed to resolve depot location");
        }
      } catch (error) {
        console.error("Failed to fetch depot coordinates:", error);
        // Fallback to last known good coordinates on error
        setStartDepot((prev) => ({ ...prev, coords: [16.544, 81.521] }));
      }
    };
    fetchDepotCoords();
  }, []);

  // --- Routing ---
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace("#", "");
      if (["pick", "excel", "custom"].includes(hash)) {
        setPage(hash);
      } else {
        setPage("landing");
      }
    };

    // Set initial page from hash
    handleHashChange();

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  const navigate = (target) => {
    window.location.hash = target;
  };

  const goBack = () => {
    window.history.back();
  };
  
  // Inject Inter font (best-effort) — optional
  useEffect(() => {
    const link = document.createElement("link");
    link.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap";
    link.rel = "stylesheet";
    document.head.appendChild(link);
    document.body.style.fontFamily = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial";
    return () => { try { document.head.removeChild(link); } catch {} };
  }, []);

  // Definitive fix for external container styles that cause layout issues.
  // This injects a style tag to override problematic global CSS.
  useEffect(() => {
    const style = document.createElement('style');
    style.innerHTML = `
      html, body, #root {
        margin: 0 !important;
        padding: 0 !important;
        max-width: none !important;
        width: 100vw !important;
        overflow-x: hidden !important;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  const MainContent = () => (
    <>
      {page === "pick" && (
        <ModeSelect
          onPick={(m) => navigate(m)}
          onBack={goBack}
        />
      )}
      {page === "excel" && <ExcelFlow goBack={goBack} startDepot={startDepot} />}
      {page === "custom" && <CustomFlow goBack={goBack} startDepot={startDepot} />}
    </>
  );

  return (
    <div className="h-screen bg-slate-800 text-slate-100 flex flex-col">
      <header className="sticky top-0 z-40 bg-slate-800/80 backdrop-blur border-b border-slate-600">
        <div className="h-20 flex items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <span className="w-10 h-10 rounded-2xl bg-emerald-500 text-white grid place-items-center shadow-md">
              <Icon name="rocket" className="w-6 h-6" />
            </span>
            <div>
              <div className="font-extrabold leading-tight text-slate-100">Quantum Path Planning</div>
              <div className="text-xs text-slate-400 -mt-0.5">SRKR depot • Compare Greedy vs QAOA</div>
            </div>
          </div>
        </div>
      </header>

      {page === "landing" ? (
         <main className="flex-1">
            <Landing onStart={() => navigate("pick")} />
         </main>
      ) : (
        <main className="p-4 sm:p-6 flex-1 overflow-y-auto">
          <div className="bg-slate-700 w-full h-full rounded-2xl shadow-sm border border-slate-600 p-4">
             <MainContent />
          </div>
        </main>
      )}

      <footer className="w-full border-t border-slate-600 bg-slate-800">
        <div className="px-6 py-4 flex flex-col md:flex-row items-center justify-between text-sm text-slate-400 space-y-2 md:space-y-0">
          {/* Depot Info */}
          <div>
            Depot: <span className="font-medium text-slate-200">{startDepot.name}</span> •{" "}
            <span className="font-mono">{startDepot.digipin}</span>
          </div>

          {/* Copyright */}
          <div>© {new Date().getFullYear()} Quantum Path Planning</div>
        </div>
      </footer>
    </div>
  );
}