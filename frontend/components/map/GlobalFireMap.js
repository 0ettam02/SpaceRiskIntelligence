"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";
import maplibregl from "maplibre-gl";
import { cellsToPointFeatureCollection, cellsToPolygonFeatureCollection, getMetricValue } from "@/data/mock-map";
import { RISK_LEVELS } from "@/lib/risk-utils";
import { formatCoordinate, formatNumber, formatPercent } from "@/lib/formatters";

// Stile completamente autonomo (nessuna sorgente esterna): evita dipendenze
// di rete/CORS verso servizi di tile pubblici, così la mappa resta
// utilizzabile offline e in ambienti con restrizioni di rete. Il riferimento
// geografico è dato da un graticolo generato localmente e dai confini
// nazionali (public/data/countries.geojson, serviti dalla stessa origine).
const BASE_STYLE = {
  version: 8,
  sources: {},
  layers: [{ id: "background", type: "background", paint: { "background-color": "#0a1622" } }],
};
const DEFAULT_VIEW = { center: [12, 14], zoom: 1.5 };
const RISK_COLORS = RISK_LEVELS.map((level) => level.color);

function buildGraticule() {
  const features = [];
  for (let lon = -180; lon <= 180; lon += 30) {
    features.push({ type: "Feature", properties: { emphasis: lon === 0 }, geometry: { type: "LineString", coordinates: [[lon, -85], [lon, 85]] } });
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    features.push({ type: "Feature", properties: { emphasis: lat === 0 }, geometry: { type: "LineString", coordinates: [[-180, lat], [180, lat]] } });
  }
  return { type: "FeatureCollection", features };
}

function buildColorSteps(breaks) {
  // Espressione "step": < breaks[0] → colore 1, [breaks[0], breaks[1]) → colore 2, ecc.
  return ["step", ["get", "metricValue"], RISK_COLORS[0], breaks[0], RISK_COLORS[1], breaks[1], RISK_COLORS[2], breaks[2], RISK_COLORS[3]];
}

function getMetricBreaks(cells, metric, intervalDays) {
  if (metric === "probability") return [0.25, 0.5, 0.75];
  const values = cells.map((cell) => getMetricValue(cell, metric, intervalDays)).sort((a, b) => a - b);
  const quantile = (p) => values[Math.min(values.length - 1, Math.floor(p * values.length))] ?? 0;
  return [quantile(0.25), quantile(0.5), quantile(0.75)];
}

function formatMetricValue(value, metric) {
  if (metric === "probability") return formatPercent(value);
  return formatNumber(Math.round(value));
}

// `command` è un oggetto { action: "reset" | "flyTo", lat?, lon?, nonce }:
// incrementare `nonce` a parità di action/lat/lon fa rieseguire il comando
// (usato da FilterPanel/ricerca coordinate senza dover esporre un ref
// imperativo, che non attraversa in modo affidabile un import dinamico).
export function GlobalFireMap({ cells, metric, viewMode, intervalDays = 7, selectedCellId, onCellSelect, command, compact = false }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const popupRef = useRef(null);
  const stateRef = useRef({ cells, metric, viewMode, intervalDays, onCellSelect });
  stateRef.current = { cells, metric, viewMode, intervalDays, onCellSelect };

  useEffect(() => {
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASE_STYLE,
      center: DEFAULT_VIEW.center,
      zoom: DEFAULT_VIEW.zoom,
      attributionControl: !compact,
      interactive: true,
    });
    mapRef.current = map;

    if (!compact) {
      map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    }

    popupRef.current = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12 });

    map.on("load", () => {
      const { cells, metric, intervalDays } = stateRef.current;
      const breaks = getMetricBreaks(cells, metric, intervalDays);

      map.addSource("graticule", { type: "geojson", data: buildGraticule() });
      map.addLayer({
        id: "graticule-lines",
        type: "line",
        source: "graticule",
        paint: {
          "line-color": "#27506b",
          "line-width": ["case", ["get", "emphasis"], 1.2, 0.5],
          "line-opacity": ["case", ["get", "emphasis"], 0.6, 0.35],
        },
      });

      // Confini politici (Natural Earth 110m, servito da /public: nessuna
      // chiamata di rete esterna) inseriti sotto il graticolo così i confini
      // restano leggibili senza coprire cluster/celle disegnati dopo.
      fetch("/data/countries.geojson")
        .then((response) => response.json())
        .then((geojson) => {
          if (!map.getStyle() || map.getSource("countries")) return;
          map.addSource("countries", { type: "geojson", data: geojson });
          map.addLayer(
            {
              id: "countries-fill",
              type: "fill",
              source: "countries",
              paint: { "fill-color": "#16283a", "fill-opacity": 0.9 },
            },
            "graticule-lines"
          );
          map.addLayer(
            {
              id: "countries-line",
              type: "line",
              source: "countries",
              paint: {
                "line-color": "rgba(148, 163, 184, 0.55)",
                "line-width": ["interpolate", ["linear"], ["zoom"], 1, 0.4, 4, 0.8, 10, 1.6],
              },
            },
            "graticule-lines"
          );
        })
        .catch(() => {});

      map.addSource("cells-point", {
        type: "geojson",
        data: cellsToPointFeatureCollection(cells, metric, intervalDays),
        cluster: true,
        clusterMaxZoom: 6,
        clusterRadius: 44,
      });
      map.addSource("cells-polygon", {
        type: "geojson",
        data: cellsToPolygonFeatureCollection(cells, metric, intervalDays),
      });

      map.addLayer({
        id: "clusters",
        type: "circle",
        source: "cells-point",
        filter: ["has", "point_count"],
        paint: {
          "circle-color": "#199e70",
          "circle-opacity": 0.85,
          "circle-stroke-width": 2,
          "circle-stroke-color": "#0c1b27",
          "circle-radius": ["step", ["get", "point_count"], 16, 15, 22, 40, 28],
        },
      });
      map.addLayer({
        id: "cluster-count",
        type: "symbol",
        source: "cells-point",
        filter: ["has", "point_count"],
        layout: { "text-field": ["get", "point_count_abbreviated"], "text-size": 12, "text-font": ["Noto Sans Bold"] },
        paint: { "text-color": "#ffffff" },
      });
      map.addLayer({
        id: "unclustered-point",
        type: "circle",
        source: "cells-point",
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-color": buildColorSteps(breaks),
          "circle-radius": 6,
          "circle-stroke-width": 1.5,
          "circle-stroke-color": "#0c1b27",
        },
      });
      map.addLayer({
        id: "cell-fill",
        type: "fill",
        source: "cells-polygon",
        layout: { visibility: "none" },
        paint: { "fill-color": buildColorSteps(breaks), "fill-opacity": 0.85 },
      });
      map.addLayer({
        id: "cell-outline",
        type: "line",
        source: "cells-polygon",
        layout: { visibility: "none" },
        paint: {
          "line-color": "rgba(226, 232, 240, 0.65)",
          "line-width": ["interpolate", ["linear"], ["zoom"], 1, 0.6, 6, 1.2, 10, 2],
        },
      });
      map.addLayer({
        id: "selected-outline",
        type: "line",
        source: "cells-polygon",
        filter: ["==", ["get", "id"], "__none__"],
        paint: { "line-color": "#ffffff", "line-width": 2.5 },
      });

      applyViewMode(map, stateRef.current.viewMode);

      const showPopup = (event, metricKey) => {
        const feature = event.features?.[0];
        if (!feature || compact) return;
        map.getCanvas().style.cursor = "pointer";
        const { lat, lon } = feature.properties;
        popupRef.current
          .setLngLat(event.lngLat)
          .setHTML(
            `<div style="font:12px system-ui;color:#0b0b0b;">${formatCoordinate(Number(lat), Number(lon))}<br/><strong>${formatMetricValue(
              feature.properties.metricValue,
              metricKey
            )}</strong></div>`
          )
          .addTo(map);
      };
      const hidePopup = () => {
        map.getCanvas().style.cursor = "";
        popupRef.current.remove();
      };

      ["unclustered-point", "cell-fill"].forEach((layerId) => {
        map.on("mousemove", layerId, (event) => showPopup(event, stateRef.current.metric));
        map.on("mouseleave", layerId, hidePopup);
        map.on("click", layerId, (event) => {
          const feature = event.features?.[0];
          if (feature) stateRef.current.onCellSelect?.(feature.properties.id);
        });
      });

      map.on("click", "clusters", (event) => {
        const feature = event.features?.[0];
        const clusterId = feature.properties.cluster_id;
        map.getSource("cells-point").getClusterExpansionZoom(clusterId, (err, zoom) => {
          if (err) return;
          map.easeTo({ center: feature.geometry.coordinates, zoom, duration: 500 });
        });
      });
      map.on("mouseenter", "clusters", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "clusters", () => {
        map.getCanvas().style.cursor = "";
      });
    });

    return () => {
      popupRef.current?.remove();
      map.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compact]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const update = () => {
      const pointSource = map.getSource("cells-point");
      const polygonSource = map.getSource("cells-polygon");
      if (!pointSource || !polygonSource) return;
      const breaks = getMetricBreaks(cells, metric, intervalDays);
      pointSource.setData(cellsToPointFeatureCollection(cells, metric, intervalDays));
      polygonSource.setData(cellsToPolygonFeatureCollection(cells, metric, intervalDays));
      map.setPaintProperty("unclustered-point", "circle-color", buildColorSteps(breaks));
      map.setPaintProperty("cell-fill", "fill-color", buildColorSteps(breaks));
    };
    if (map.isStyleLoaded()) update();
    else map.once("load", update);
  }, [cells, metric, intervalDays]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const apply = () => applyViewMode(map, viewMode);
    if (map.isStyleLoaded()) apply();
    else map.once("load", apply);
  }, [viewMode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.getLayer("selected-outline")) return;
    map.setFilter("selected-outline", ["==", ["get", "id"], selectedCellId || "__none__"]);
  }, [selectedCellId]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !command) return;
    if (command.action === "reset") {
      map.easeTo({ center: DEFAULT_VIEW.center, zoom: DEFAULT_VIEW.zoom, duration: 500 });
    } else if (command.action === "flyTo" && typeof command.lat === "number" && typeof command.lon === "number") {
      map.flyTo({ center: [command.lon, command.lat], zoom: 6, duration: 700 });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [command?.nonce]);

  return (
    <div
      ref={containerRef}
      role="application"
      aria-label="Mappa geospaziale interattiva dei rilevamenti satellitari"
      className="h-full w-full"
    />
  );
}

function applyViewMode(map, viewMode) {
  const clusterVisible = viewMode !== "cells";
  const cellVisible = viewMode === "cells";
  ["clusters", "cluster-count", "unclustered-point"].forEach((layerId) => {
    if (map.getLayer(layerId)) map.setLayoutProperty(layerId, "visibility", clusterVisible ? "visible" : "none");
  });
  ["cell-fill", "cell-outline"].forEach((layerId) => {
    if (map.getLayer(layerId)) map.setLayoutProperty(layerId, "visibility", cellVisible ? "visible" : "none");
  });
}
