import { newPlot, resizeAll } from './plotlyService.js';

// Tailwind dark mode colorscale for terrain/cost visualization
export const terrainColorscale = [
  [0, '#1f2937'],    // gray-800 - impassable
  [0.25, '#374151'], // gray-700
  [0.5, '#4b5563'],  // gray-600
  [0.75, '#9ca3af'], // gray-400
  [1, '#d1d5db']     // gray-300 - low cost
];

// Distance field colorscale with black for impassable
export const distanceColorscale = [
  [0, '#000000'],    // black - impassable (value = -1)
  [0.001, '#1e3a8a'], // blue-909 - not reached yet (value = 0)
  [0.2, '#1e40af'],  // blue-800
  [0.4, '#2563eb'],  // blue-700
  [0.6, '#3b82f6'],  // blue-600
  [0.8, '#60a5fa'],  // blue-500
  [1, '#93c5fd']     // blue-400
];

// Create layout for heatmaps with adjusted margins for horizontal colorbar
export const createLayout = (title) => ({
  title: {
    text: title,
    font: { color: '#f9fafb', size: 14 }
  },
  autosize: true,
  margin: { t: 40, r: 20, b: 80, l: 50 }, // More bottom margin for horizontal colorbar
  paper_bgcolor: '#171A24',
  plot_bgcolor: '#171A24',
  font: {
    color: '#e5e7eb'
  },
  xaxis: {
    gridcolor: '#374151',
    zerolinecolor: '#4b5563',
    constrain: 'domain',
    scaleanchor: 'y',
    scaleratio: 1
  },
  yaxis: {
    gridcolor: '#374151',
    zerolinecolor: '#4b5563',
    constrain: 'domain'
  }
});

const plotConfig = {
  responsive: true,
  displayModeBar: false
};

/**
 * Render input (obstacle/cost) heatmap
 */
export function renderInputHeatmap(data, gridSize) {
  // Reshape flat array to 2D
  const z = [];
  for (let y = 0; y < gridSize; y++) {
    const row = [];
    for (let x = 0; x < gridSize; x++) {
      row.push(data[y * gridSize + x]);
    }
    z.push(row);
  }

  const heatmapData = [{
    z,
    type: 'heatmap',
    colorscale: terrainColorscale,
    colorbar: {
      orientation: 'h',           // Horizontal colorbar
      y: -0.15,                   // Position below plot
      yanchor: 'top',
      x: 0.5,                     // Center horizontally
      xanchor: 'center',
      len: 1.0,                   // Full width to match square heatmap
      lenmode: 'fraction',        // Use fraction of plot area
      thickness: 15,              // Thinner for horizontal
      tickfont: { 
        color: '#e5e7eb',
        size: 10
      },
      outlinecolor: '#4b5563',
      outlinewidth: 1,
      exponentformat: 'e',        // Scientific notation (e.g., 1e+2)
      showexponent: 'all'         // Always show exponent
    }
  }];

  newPlot('heatmap-input', heatmapData, createLayout(''), plotConfig);
}

/**
 * Render output (distance field) heatmap
 * Marks impassable cells (terrain = 0) as -1 for black color
 */
export function renderOutputHeatmap(distanceField, terrainData, gridSize) {
  // Reshape flat array to 2D and mark impassable cells
  const z = [];
  for (let y = 0; y < gridSize; y++) {
    const row = [];
    for (let x = 0; x < gridSize; x++) {
      const idx = y * gridSize + x;
      const distance = distanceField[idx];
      const terrain = terrainData[idx];
      
      // If terrain is impassable (0), mark as -1 for black color
      // Otherwise use the distance value
      if (terrain === 0) {
        row.push(-1);
      } else {
        row.push(distance);
      }
    }
    z.push(row);
  }

  const heatmapData = [{
    z,
    type: 'heatmap',
    colorscale: distanceColorscale,
    zmin: -1,  // Force min to include black for impassable
    colorbar: {
      orientation: 'h',           // Horizontal colorbar
      y: -0.15,                   // Position below plot
      yanchor: 'top',
      x: 0.5,                     // Center horizontally
      xanchor: 'center',
      len: 1.0,                   // Full width to match square heatmap
      lenmode: 'fraction',        // Use fraction of plot area
      thickness: 15,              // Thinner for horizontal
      tickfont: { 
        color: '#e5e7eb',
        size: 10
      },
      outlinecolor: '#4b5563',
      outlinewidth: 1,
      exponentformat: 'e',        // Scientific notation (e.g., 1e+2)
      showexponent: 'all'         // Always show exponent
    }
  }];

  newPlot('heatmap-output', heatmapData, createLayout(''), plotConfig);
}

/**
 * Handle window resize
 */
export function handleResize() {
  resizeAll(['heatmap-input', 'heatmap-output']);
}
