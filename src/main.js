import { createLogger } from './ui/logger.js';
import { initControls } from './ui/controls.js';
import { initWebGPU } from './webgpu/device.js';
import { config, onConfigChange } from './config.js';
import { getTerrain } from './bfs/terrain.js';
import { renderInputHeatmap, renderOutputHeatmap, handleResize } from './plots/heatmaps.js';
import { runAllBFS } from './bfs/runner.js';

// Initialize logger
const logger = createLogger('console-output');
window.logger = logger;

// Initialize controls
initControls();

// Resize handler
window.addEventListener('resize', handleResize);

// WebGPU context and results storage
let gpuContext = null;
let benchmarkResults = [];
let currentTerrain = null;

// Track previous config to detect what changed
let prevConfig = { ...config };

// Initialize WebGPU and check features
(async () => {
  try {
    logger.info('Initializing WebGPU...');
    gpuContext = await initWebGPU();
    
    if (!gpuContext) {
      logger.error('WebGPU is not supported in this browser');
      return;
    }
    
    logger.success(`WebGPU initialized successfully`);
    logger.info(`Timestamp queries: ${gpuContext.canTimestamp ? 'Supported' : 'Not supported'}`);
    
    // Load and display initial terrain
    await updateInputHeatmap();
    
  } catch (err) {
    logger.error(`WebGPU initialization failed: ${err.message}`);
  }
})();

// Update input heatmap when config changes
async function updateInputHeatmap() {
  try {
    const terrain = await getTerrain(
      config.obstacleType,
      config.uniformCost,
      config.gridSize,
      config.goalX,
      config.goalY
    );
    
    if (!terrain) {
      logger.error('Failed to load terrain');
      return;
    }
    
    currentTerrain = terrain;
    renderInputHeatmap(terrain, config.gridSize);
    
  } catch (err) {
    logger.error(`Failed to update input heatmap: ${err.message}`);
  }
}

// Update output heatmap when selected shader changes
function updateOutputHeatmap() {
  if (benchmarkResults.length === 0) return;
  
  const result = benchmarkResults.find(r => r.key === config.selectedShader);
  if (result && result.distanceField) {
    renderOutputHeatmap(result.distanceField, currentTerrain, config.gridSize);
  }
}

// Listen for config changes - only update what changed
onConfigChange(() => {
  // Check if input-related settings changed
  const inputChanged = 
    prevConfig.obstacleType !== config.obstacleType ||
    prevConfig.uniformCost !== config.uniformCost ||
    prevConfig.gridSize !== config.gridSize;
  
  // Check if shader selection changed
  const shaderChanged = prevConfig.selectedShader !== config.selectedShader;
  
  // Update only what's needed
  if (inputChanged) {
    updateInputHeatmap();
  }
  
  if (shaderChanged) {
    updateOutputHeatmap();
  }
  
  // Update previous config for next comparison
  prevConfig = { ...config };
});

// Run BFS button handler
document.getElementById('runBFS')?.addEventListener('click', async () => {
  if (!gpuContext) {
    logger.error('WebGPU not initialized');
    return;
  }
  
  try {
    benchmarkResults = await runAllBFS(gpuContext, config, logger);
    
    // Display the selected shader result
    updateOutputHeatmap();
    
  } catch (err) {
    logger.error(`BFS benchmark failed: ${err.message}`);
  }
});
