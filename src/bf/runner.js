/**
 * BF benchmark runner - orchestrates shader execution with timing
 */

import { getTerrain } from './terrain.js';
import { getAllShaderOptions, getShader } from './shaderRegistry.js';
import { createUniformBuffer, createStorageBuffer, createDistanceBuffers } from '../webgpu/buffers.js';
import { createStorageTexture, createDistanceTextures, uploadTextureData } from '../webgpu/textures.js';
import { createBFBindGroupLayout, createBFTexBindGroupLayout, createBFBindGroup, createBFTexBindGroup } from '../webgpu/bindGroups.js';
import { createComputePipeline } from '../webgpu/pipelines.js';
import { createTimestampResources, encodeBFPasses } from '../webgpu/commands.js';
import { readBuffer, readTexture, readTimestamps } from '../webgpu/readback.js';

/**
 * Run all BF shaders matching the current storage type
 */
export async function runAllBF(gpuContext, config, logger) {
  const { device, canTimestamp } = gpuContext;
  
  logger.info('═'.repeat(50));
  logger.info('Starting BF Benchmark Suite');
  logger.info('═'.repeat(50));
  
  // Load terrain
  logger.info(`Loading terrain: ${config.obstacleType} (${config.uniformCost ? 'uniform' : 'varying'})`);
  const terrainData = await getTerrain(
    config.obstacleType,
    config.uniformCost,
    config.gridSize,
    config.goalX,
    config.goalY
  );
  
  if (!terrainData) {
    throw new Error('Failed to load terrain data');
  }
  
  // Get all shaders and filter out unsupported feature requirements.
  const allShaderOptions = getAllShaderOptions(config.useTextures);
  const shaderOptions = getAllShaderOptions(config.useTextures, device.features);

  const supportedShaderKeys = new Set(shaderOptions.map(({ key }) => key));
  const skippedShaders = allShaderOptions.filter(({ key }) => !supportedShaderKeys.has(key));
  
  logger.info(`Running ${shaderOptions.length} shaders (${config.useTextures ? 'Texture' : 'Buffer'} mode)`);
  logger.info(`Global iterations: ${config.globalIterations}, Inner iterations (WG): ${config.innerIterationsWorkgroup}, Inner iterations (SG): ${config.innerIterationsSubgroup}`);
  logger.info(`Grid size: ${config.gridSize}×${config.gridSize}, Early out: ${config.earlyOut ? 'ON' : 'OFF'}`);
  if (skippedShaders.length > 0) {
    logger.warning(`Skipping unsupported shaders: ${skippedShaders.map(({ label }) => label).join(', ')}`);
  }
  logger.info('');
  
  const results = [];
  
  // Run each shader
  for (const { key, label } of shaderOptions) {
    const shaderInfo = getShader(key, config.useTextures);
    
    try {
      const result = await runSingleBF(
        device,
        canTimestamp,
        shaderInfo,
        label,
        terrainData,
        config,
        logger
      );
      
      results.push({ key, label, ...result });
    } catch (err) {
      logger.error(`Failed to run ${label}: ${err.message}`);
    }
  }
  
  // Summary
  logger.info('');
  logger.info('═'.repeat(50));
  logger.info('Benchmark Summary');
  logger.info('═'.repeat(50));
  
  for (const result of results) {
    if (result.avgTimeMs) {
      logger.info(`${result.label.padEnd(30)} ${result.avgTimeMs.toFixed(4)} ms/iter (${result.totalTimeMs.toFixed(2)} ms total)`);
    }
  }

  logger.info('═'.repeat(50));
  logger.info('Throughput');
  logger.info('═'.repeat(50));
  for (const result of results) {
    if (result.filled && result.totalTimeMs) {
      const throughput = result.filled / result.totalTimeMs;
      logger.info(`${result.label.padEnd(30)} ${Math.round(throughput)} cells/ms`);
    }
  }

  return results;
}

/**
 * Run a single BF shader variant
 */
async function runSingleBF(device, canTimestamp, shaderInfo, label, terrainData, config, logger) {
  logger.info(`Running: ${label}`);

  const gridSize = config.gridSize;
  const uniformProfile = shaderInfo.uniformProfile || 'default';
  const uniformBuffer = createUniformBuffer(device, config, uniformProfile);
  
  // Calculate dispatch dimensions based on workgroup size
  const workgroupSize = shaderInfo.workgroupSize || [8, 8];
  const dispatchX = Math.ceil(gridSize / workgroupSize[0]);
  const dispatchY = Math.ceil(gridSize / workgroupSize[1]);
  try {
    if (config.useTextures) {
      return await runTextureVariant(
        device,
        canTimestamp,
        shaderInfo,
        label,
        terrainData,
        uniformBuffer,
        config,
        dispatchX,
        dispatchY,
        logger
      );
    } else {
      return await runBufferVariant(
        device,
        canTimestamp,
        shaderInfo,
        label,
        terrainData,
        uniformBuffer,
        config,
        dispatchX,
        dispatchY,
        logger
      );
    }
  } finally {
    uniformBuffer.destroy();
  }
}

/**
 * Run buffer-based shader variant
 */
async function runBufferVariant(device, canTimestamp, shaderInfo, label, terrainData, uniformBuffer, config, dispatchX, dispatchY, logger) {
  // Create buffers
  const terrainBuffer = createStorageBuffer(device, terrainData);
  const { ping, pong } = createDistanceBuffers(device, config.gridSize);
  
  // Create bind group layout and pipeline
  const bindGroupLayout = createBFBindGroupLayout(device);
  const pipeline = createComputePipeline(device, shaderInfo.code, bindGroupLayout, label);
  
  // Create bind groups for ping-pong
  const bindGroupPing = createBFBindGroup(device, bindGroupLayout, terrainBuffer, ping, pong, uniformBuffer);
  const bindGroupPong = createBFBindGroup(device, bindGroupLayout, terrainBuffer, pong, ping, uniformBuffer);
  
  // Timestamp resources
  const timestampResources = canTimestamp 
    ? createTimestampResources(device, config.globalIterations)
    : null;
  
  // Encode all passes
  const encoder = device.createCommandEncoder();
  const lastIsPing = encodeBFPasses(
    encoder,
    pipeline,
    bindGroupPing,
    bindGroupPong,
    dispatchX,
    dispatchY,
    config.globalIterations,
    timestampResources
  );
  
  // Submit
  device.queue.submit([encoder.finish()]);
  
  // Read timestamps
  let avgTimeMs = null;
  let totalTimeMs = null;
  
  if (canTimestamp) {
    const timings = await readTimestamps(device, timestampResources);
    const nonZero = timings.filter(t => t > 0);
    
    if (nonZero.length > 0) {
      avgTimeMs = nonZero.reduce((a, b) => a + b, 0) / nonZero.length;
      totalTimeMs = timings.reduce((a, b) => a + b, 0);
      logger.success(`  Average: ${avgTimeMs.toFixed(4)} ms/iter, Total: ${totalTimeMs.toFixed(2)} ms`);
    } else {
      logger.warning(`  No timing data available`);
    }
  } else {
    logger.warning(`  Timestamps not supported`);
  }
  
  // Read back final distance field
  const lastBuffer = lastIsPing ? ping : pong;
  const distanceField = await readBuffer(device, lastBuffer, terrainData.byteLength);
  
  const filled = distanceField.filter(v => v > 0).length;
  logger.info(`  Filled cells: ${filled}/${config.gridSize * config.gridSize}`);
  
  return {
    distanceField,
    avgTimeMs,
    totalTimeMs,
    filled
  };
}

/**
 * Run texture-based shader variant
 */
async function runTextureVariant(device, canTimestamp, shaderInfo, label, terrainData, uniformBuffer, config, dispatchX, dispatchY, logger) {
  // Create textures
  const terrainTex = createStorageTexture(device, config.gridSize);
  await uploadTextureData(device, terrainTex, terrainData, config.gridSize);
  const terrainView = terrainTex.createView();
  
  const { ping, pong } = await createDistanceTextures(device, config.gridSize);
  const pingView = ping.createView();
  const pongView = pong.createView();
  
  // Create bind group layout and pipeline
  const bindGroupLayout = createBFTexBindGroupLayout(device);
  const pipeline = createComputePipeline(device, shaderInfo.code, bindGroupLayout, label);
  
  // Create bind groups for ping-pong
  const bindGroupPing = createBFTexBindGroup(device, bindGroupLayout, terrainView, pingView, pongView, uniformBuffer);
  const bindGroupPong = createBFTexBindGroup(device, bindGroupLayout, terrainView, pongView, pingView, uniformBuffer);
  
  // Timestamp resources
  const timestampResources = canTimestamp 
    ? createTimestampResources(device, config.globalIterations)
    : null;
  
  // Encode all passes
  const encoder = device.createCommandEncoder();
  const lastIsPing = encodeBFPasses(
    encoder,
    pipeline,
    bindGroupPing,
    bindGroupPong,
    dispatchX,
    dispatchY,
    config.globalIterations,
    timestampResources
  );
  
  // Submit
  device.queue.submit([encoder.finish()]);
  
  // Read timestamps
  let avgTimeMs = null;
  let totalTimeMs = null;
  
  if (canTimestamp) {
    const timings = await readTimestamps(device, timestampResources);
    const nonZero = timings.filter(t => t > 0);
    
    if (nonZero.length > 0) {
      avgTimeMs = nonZero.reduce((a, b) => a + b, 0) / nonZero.length;
      totalTimeMs = timings.reduce((a, b) => a + b, 0);
      logger.success(`  Average: ${avgTimeMs.toFixed(4)} ms/iter, Total: ${totalTimeMs.toFixed(2)} ms`);
    } else {
      logger.warning(`  No timing data available`);
    }
  } else {
    logger.warning(`  Timestamps not supported`);
  }
  
  // Read back final distance field
  const lastTex = lastIsPing ? ping : pong;
  const distanceField = await readTexture(device, lastTex, config.gridSize);
  
  const filled = distanceField.filter(v => v > 0).length;
  logger.info(`  Filled cells: ${filled}/${config.gridSize * config.gridSize}`);
  
  return {
    distanceField,
    avgTimeMs,
    totalTimeMs,
    filled
  };
}
