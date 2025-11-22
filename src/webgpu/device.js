/**
 * WebGPU device initialization with timestamp query support and higher limits
 */
export async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error('WebGPU not supported in this browser');
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: 'high-performance'
  });

  if (!adapter) {
    throw new Error('No WebGPU adapter available');
  }

  // Gather features
  const features = [];
  if (adapter.features.has('timestamp-query')) {
    features.push('timestamp-query');
  }

  // Request higher limits for large tiled workgroups
  const requiredLimits = {
    maxComputeInvocationsPerWorkgroup: Math.min(
      adapter.limits.maxComputeInvocationsPerWorkgroup,
      1024
    ),
    maxComputeWorkgroupSizeX: Math.min(
      adapter.limits.maxComputeWorkgroupSizeX,
      32
    ),
    maxComputeWorkgroupSizeY: Math.min(
      adapter.limits.maxComputeWorkgroupSizeY,
      32
    ),
    maxComputeWorkgroupStorageSize: Math.min(
      adapter.limits.maxComputeWorkgroupStorageSize,
      32768
    )
  };

  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: requiredLimits
  });

  const canTimestamp = device.features.has('timestamp-query');

  return { adapter, device, canTimestamp };
}
