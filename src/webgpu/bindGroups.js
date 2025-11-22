/**
 * Bind group creation matching Python profiler patterns
 */

/**
 * Create bind group layout for storage buffer-based BFS
 */
export function createBFSBindGroupLayout(device) {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // terrain
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // distance_in
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // distance_out
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }            // uniforms
    ]
  });
}

/**
 * Create bind group layout for texture-based BFS
 */
export function createBFSTexBindGroupLayout(device) {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-only', format: 'r32float' } },   // terrain
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-only', format: 'r32float' } },   // distance_in
      { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },  // distance_out
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }                                     // uniforms
    ]
  });
}

/**
 * Create bind group for buffer-based BFS (ping or pong)
 */
export function createBFSBindGroup(device, layout, terrainBuffer, distanceIn, distanceOut, uniformBuffer) {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: terrainBuffer } },
      { binding: 1, resource: { buffer: distanceIn } },
      { binding: 2, resource: { buffer: distanceOut } },
      { binding: 3, resource: { buffer: uniformBuffer } }
    ]
  });
}

/**
 * Create bind group for texture-based BFS (ping or pong)
 */
export function createBFSTexBindGroup(device, layout, terrainView, distanceInView, distanceOutView, uniformBuffer) {
  return device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: terrainView },
      { binding: 1, resource: distanceInView },
      { binding: 2, resource: distanceOutView },
      { binding: 3, resource: { buffer: uniformBuffer } }
    ]
  });
}
