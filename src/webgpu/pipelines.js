/**
 * Compute pipeline creation
 */

/**
 * Create compute pipeline from WGSL shader code
 */
export function createComputePipeline(device, shaderCode, bindGroupLayout, label = 'compute') {
  const shaderModule = device.createShaderModule({
    label: `${label}_shader`,
    code: shaderCode
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });

  return device.createComputePipeline({
    label: `${label}_pipeline`,
    layout: pipelineLayout,
    compute: {
      module: shaderModule,
      entryPoint: 'main'
    }
  });
}
