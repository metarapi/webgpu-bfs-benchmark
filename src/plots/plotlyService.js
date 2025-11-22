import Plotly from 'plotly.js-dist';

export function newPlot(containerId, data, layout, config) {
  const el = document.getElementById(containerId);
  if (!el) throw new Error(`Container ${containerId} not found`);
  return Plotly.newPlot(el, data, layout, config);
}

export function resize(containerId) {
  const el = document.getElementById(containerId);
  if (el) Plotly.Plots.resize(el);
}

export function resizeAll(ids = []) {
  ids.forEach(id => resize(id));
}
