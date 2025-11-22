/**
 * Command encoding utilities with timestamp support
 */

/**
 * Create timestamp query resources
 */
export function createTimestampResources(device, iterations) {
  const queryCount = 2 * iterations;
  const slotStride = 256; // 256-byte alignment
  const resolveBufferSize = queryCount * slotStride;

  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: queryCount
  });

  const resolveBuffer = device.createBuffer({
    size: resolveBufferSize,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
  });

  const readBuffer = device.createBuffer({
    size: resolveBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  return { querySet, resolveBuffer, readBuffer, slotStride, queryCount };
}

/**
 * Encode BFS compute passes with ping-pong and timestamps
 */
export function encodeBFSPasses(encoder, pipeline, bindGroupPing, bindGroupPong, dispatchX, dispatchY, iterations, timestampResources = null) {
  let readFromPing = true;

  for (let i = 0; i < iterations; i++) {
    const passDescriptor = {};
    
    if (timestampResources) {
      passDescriptor.timestampWrites = {
        querySet: timestampResources.querySet,
        beginningOfPassWriteIndex: 2 * i,
        endOfPassWriteIndex: 2 * i + 1
      };
    }

    const pass = encoder.beginComputePass(passDescriptor);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, readFromPing ? bindGroupPing : bindGroupPong);
    pass.dispatchWorkgroups(dispatchX, dispatchY);
    pass.end();

    readFromPing = !readFromPing;
  }

  // Resolve timestamps
  if (timestampResources) {
    for (let i = 0; i < timestampResources.queryCount; i++) {
      encoder.resolveQuerySet(
        timestampResources.querySet,
        i,
        1,
        timestampResources.resolveBuffer,
        i * timestampResources.slotStride
      );
    }
  }

  return readFromPing; // indicates which buffer has final result
}
