/**
 * Global configuration state linked to UI controls
 */
export const config = {
  gridSize: 512,
  earlyOut: true,
  globalIterations: 16,
  innerIterationsWorkgroup: 4,
  innerIterationsSubgroup: 4,
  goalX: 1,  // Changed from gridSize-3 to (1,1)
  goalY: 1,
  selectedShader: 'BF naive',
  useTextures: false,
  obstacleType: 'dla-maze',
  uniformCost: true,
};

// Goal is now fixed at (1,1) - no need to update it
export function updateGoal() {
  config.goalX = 1;
  config.goalY = 1;
}

updateGoal();

// Observable pattern for config changes
const listeners = [];
export function onConfigChange(callback) {
  listeners.push(callback);
}

export function notifyConfigChange() {
  listeners.forEach(cb => cb(config));
}
