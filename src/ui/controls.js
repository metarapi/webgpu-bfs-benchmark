import { config, updateGoal, notifyConfigChange } from '../config.js';
import { getAllShaderOptions } from '../bfs/shaderRegistry.js';

export function initControls() {
  // Populate shader dropdown initially
  populateShaderDropdown();

  // Map size
  document.getElementById('mapSize').addEventListener('change', (e) => {
    config.gridSize = parseInt(e.target.value);
    updateGoal();
    notifyConfigChange();
  });

  // Storage type toggle
  document.getElementById('useTextures').addEventListener('change', (e) => {
    config.useTextures = e.target.checked;
    populateShaderDropdown(); // Refresh shader list only
    // No notify here: user must re-run benchmark for buffer vs texture comparisons
  });

  // Early out toggle
  document.getElementById('earlyOut').addEventListener('change', (e) => {
    config.earlyOut = e.target.checked;
    notifyConfigChange();
  });

  // BFS shader select
  document.getElementById('bfsShader').addEventListener('change', (e) => {
    config.selectedShader = e.target.value;
    notifyConfigChange();
  });

  // Obstacle map radios
  document.querySelectorAll('input[name="obstacleMap"]').forEach((radio) => {
    radio.addEventListener('change', (e) => {
      config.obstacleType = e.target.value;
      notifyConfigChange();
    });
  });

  // Uniform cost toggle
  document.getElementById('uniformCost').addEventListener('change', (e) => {
    config.uniformCost = e.target.checked;
    notifyConfigChange();
  });

  // Global iterations input
  document.getElementById('globalIterations').addEventListener('change', (e) => {
    config.globalIterations = parseInt(e.target.value);
    notifyConfigChange();
  });

  // Inner iterations input
  document.getElementById('innerIterations').addEventListener('change', (e) => {
    config.innerIterations = parseInt(e.target.value);
    notifyConfigChange();
  });
}

// Populate shader dropdown based on storage type
function populateShaderDropdown() {
  const select = document.getElementById('bfsShader');
  const options = getAllShaderOptions(config.useTextures);

  const currentSelection = config.selectedShader;

  select.innerHTML = '';

  options.forEach(({ key, label }) => {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = label;
    select.appendChild(option);
  });

  const matchingOption = options.find((opt) => opt.key === currentSelection);
  if (matchingOption) {
    config.selectedShader = matchingOption.key;
    select.value = matchingOption.key;
  } else {
    config.selectedShader = options[0]?.key || '';
    select.value = config.selectedShader;
  }
}
