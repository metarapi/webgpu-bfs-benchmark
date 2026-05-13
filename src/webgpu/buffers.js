/**
 * Buffer creation utilities matching Python profiler.py patterns
 */

/**
 * Create uniform buffer with grid parameters
 */
export function createUniformBuffer(device, config, uniformProfile = 'default') {
  const sqrt2 = Math.sqrt(2);
  
  // Both uniform profiles are packed into 32 bytes for 16-byte alignment.
  const uniformData = new ArrayBuffer(32);
  const view = new DataView(uniformData);
  
  view.setUint32(0, config.gridSize, true);           // gridSizeX
  view.setUint32(4, config.gridSize, true);           // gridSizeY

  if (uniformProfile === 'subgroup') {
    view.setUint32(8, config.innerIterationsSubgroup, true);       // maxSubgroupIterations
    view.setUint32(12, config.innerIterationsWorkgroup, true);     // maxWorkgroupIterations
    view.setFloat32(16, sqrt2, true);                               // sqrt2
    view.setFloat32(20, 1e-6, true);                                // epsilon
    view.setFloat32(24, 0.0, true);                                 // _pad0.x
    view.setFloat32(28, 0.0, true);                                 // _pad0.y
  } else {
    view.setFloat32(8, sqrt2, true);                                // sqrt2
    view.setFloat32(12, 1.0 / sqrt2, true);                         // sqrt2inv
    view.setUint32(16, config.innerIterationsWorkgroup, true);      // iterations (inner WG)
    view.setUint32(20, config.earlyOut ? 1 : 0, true);              // earlyOut
    view.setUint32(24, 0, true);                                    // _pad0
    view.setUint32(28, 0, true);                                    // _pad1 for alignment
  }

  const buffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true
  });

  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(uniformData));
  buffer.unmap();
  return buffer;
}

/**
 * Create storage buffer with initial data
 */
export function createStorageBuffer(device, data, usage = GPUBufferUsage.STORAGE) {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true
  });

  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

/**
 * Create ping-pong distance buffers seeded at goal (1,1)
 */
export function createDistanceBuffers(device, gridSize) {
  const numCells = gridSize * gridSize;
  const distanceSeed = new Float32Array(numCells);
  
  // Seed at (1, 1) - goalIndex = 1 * gridSize + 1
  const goalIndex = 1 * gridSize + 1;
  distanceSeed[goalIndex] = 1.0;

  const ping = createStorageBuffer(device, distanceSeed);
  const pong = createStorageBuffer(device, distanceSeed);

  return { ping, pong, seed: distanceSeed };
}

/**
 * Readback buffer for GPU -> CPU transfer
 */
export function createReadbackBuffer(device, size) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
}
