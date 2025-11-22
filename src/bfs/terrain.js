/**
 * Terrain generation and loading from CSV files
 */

/**
 * Load CSV terrain map from public folder
 */
export async function loadTerrainFromCSV(filename) {
  try {
    const response = await fetch(`/obstacle-maps/${filename}`);
    if (!response.ok) {
      throw new Error(`Failed to load ${filename}: ${response.statusText}`);
    }
    
    const text = await response.text();
    return parseCSV(text);
  } catch (err) {
    console.error(`Error loading terrain: ${err.message}`);
    return null;
  }
}

/**
 * Simple CSV parser for flat numeric data
 */
function parseCSV(csvText) {
  const lines = csvText.trim().split('\n');
  const values = [];
  
  for (const line of lines) {
    const row = line.split(',').map(v => parseFloat(v.trim()));
    values.push(...row);
  }
  
  return new Float32Array(values);
}

/**
 * Terrain type definitions with uniform and varying cost variants
 */
export const terrainTypes = {
  'dla-maze': {
    label: 'DLA Maze',
    uniformPrefix: 'uniform-cost/dla-morphological-filter-maze',
    varyingPrefix: 'varying-cost/dla-morphological-filter-maze',
    varyingSuffix: '-cost-modified'
  },
  'maze-rotated': {
    label: 'Maze Rotated',
    uniformPrefix: 'uniform-cost/maze-rotated',
    varyingPrefix: 'varying-cost/maze-rotated',
    varyingSuffix: '-cost-modified'
  },
  'organic': {
    label: 'Organic',
    uniformPrefix: 'uniform-cost/organic',
    varyingPrefix: 'varying-cost/organic',
    varyingSuffix: '-cost-modified'
  },
  'no-obstacle': {
    label: 'No Obstacles',
    uniformPrefix: 'uniform-cost/uniformCost-noObstacle',
    varyingPrefix: 'varying-cost/nonuniformCost-noObstacle',
    varyingSuffix: ''
  }
};

/**
 * Get terrain filename for given type, cost mode, and size
 */
function getTerrainFilename(obstacleType, uniformCost, gridSize) {
  const terrain = terrainTypes[obstacleType];
  if (!terrain) return null;
  
  const prefix = uniformCost ? terrain.uniformPrefix : terrain.varyingPrefix;
  const suffix = uniformCost ? '' : terrain.varyingSuffix;
  
  return `${prefix}-${gridSize}x${gridSize}${suffix}.csv`;
}

/**
 * Get terrain data for given configuration
 */
export async function getTerrain(obstacleType, uniformCost, gridSize, goalX, goalY) {
  const filename = getTerrainFilename(obstacleType, uniformCost, gridSize);
  
  if (!filename) {
    console.error(`Unknown terrain type: ${obstacleType}`);
    return buildOpenGrid(gridSize);
  }
  
  const data = await loadTerrainFromCSV(filename);
  
  if (!data) {
    console.error(`Failed to load terrain from ${filename}, falling back to open grid`);
    return buildOpenGrid(gridSize);
  }
  
  // Verify size matches
  if (data.length !== gridSize * gridSize) {
    console.error(`Terrain size mismatch: expected ${gridSize * gridSize}, got ${data.length}`);
    return buildOpenGrid(gridSize);
  }
  
  return data;
}

/**
 * Build open grid (all passable) as fallback
 */
export function buildOpenGrid(size) {
  return new Float32Array(size * size).fill(1.0);
}

/**
 * Get list of available terrain types
 */
export function getTerrainTypesList() {
  return Object.entries(terrainTypes).map(([key, info]) => ({
    key,
    label: info.label
  }));
}
