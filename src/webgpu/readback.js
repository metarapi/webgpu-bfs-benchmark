/**
 * GPU to CPU readback utilities
 */

/**
 * Read buffer data back to CPU
 */
export async function readBuffer(device, sourceBuffer, size) {
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, size);
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(stagingBuffer.getMappedRange()).slice();
  stagingBuffer.unmap();

  return data;
}

/**
 * Read texture data back to CPU
 */
export async function readTexture(device, texture, gridSize) {
  const size = gridSize * gridSize * 4;
  const buffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer(
    { texture },
    { buffer, bytesPerRow: gridSize * 4 },
    [gridSize, gridSize, 1]
  );
  device.queue.submit([encoder.finish()]);

  await buffer.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(buffer.getMappedRange()).slice();
  buffer.unmap();

  return data;
}

/**
 * Read timestamp query results and convert to milliseconds
 */
export async function readTimestamps(device, timestampResources) {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(
    timestampResources.resolveBuffer,
    0,
    timestampResources.readBuffer,
    0,
    timestampResources.readBuffer.size
  );
  device.queue.submit([encoder.finish()]);

  await timestampResources.readBuffer.mapAsync(GPUMapMode.READ);
  const raw = new Uint8Array(timestampResources.readBuffer.getMappedRange());
  
  // Extract timestamps (one u64 per 256-byte slot)
  const times = [];
  for (let i = 0; i < timestampResources.queryCount; i++) {
    const offset = i * timestampResources.slotStride;
    const view = new DataView(raw.buffer, offset, 8);
    const low = view.getUint32(0, true);
    const high = view.getUint32(4, true);
    times.push(BigInt(high) << 32n | BigInt(low));
  }
  
  timestampResources.readBuffer.unmap();

  // Calculate per-iteration deltas in milliseconds
  const deltas = [];
  for (let i = 0; i < times.length / 2; i++) {
    const start = times[2 * i];
    const end = times[2 * i + 1];
    const deltaNs = Number(end - start);
    deltas.push(deltaNs / 1_000_000); // ns -> ms
  }

  return deltas;
}
