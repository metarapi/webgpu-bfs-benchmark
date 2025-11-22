/**
 * Texture creation utilities for storage texture variants
 */

/**
 * Create r32float storage texture
 */
export function createStorageTexture(device, size, usage = GPUTextureUsage.STORAGE_BINDING) {
  return device.createTexture({
    size: [size, size, 1],
    usage: usage | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    dimension: '2d',
    format: 'r32float'
  });
}

/**
 * Upload Float32Array data to texture
 */
export async function uploadTextureData(device, texture, data, size) {
  const bytesPerRow = size * 4;
  const stagingBuffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true
  });

  new Float32Array(stagingBuffer.getMappedRange()).set(data);
  stagingBuffer.unmap();

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToTexture(
    { buffer: stagingBuffer, bytesPerRow },
    { texture },
    [size, size, 1]
  );

  device.queue.submit([encoder.finish()]);
}

/**
 * Create distance texture seeded at goal (1,1)
 */
export async function createDistanceTextures(device, gridSize) {
  const numCells = gridSize * gridSize;
  const seed = new Float32Array(numCells);
  
  // Seed at (1, 1)
  const goalIndex = 1 * gridSize + 1;
  seed[goalIndex] = 1.0;

  const ping = createStorageTexture(device, gridSize);
  const pong = createStorageTexture(device, gridSize);

  // Upload seed data
  await uploadTextureData(device, ping, seed, gridSize);
  await uploadTextureData(device, pong, seed, gridSize);

  return { ping, pong, seed };
}
