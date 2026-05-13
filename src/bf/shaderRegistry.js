/**
 * Shader path registry - import with ?raw suffix for strings
 */

// Storage buffer variants
import BF_naive from '../shaders/BF.wgsl?raw';
import BF_8x8 from '../shaders/8x8/StorageBufferBased/BF-tiled-8x8.wgsl?raw';
import BF_8x8_lib from '../shaders/8x8/StorageBufferBased/BF-tiled-library-8x8.wgsl?raw';
import BF_16x16 from '../shaders/16x16/StorageBufferBased/BF-tiled-16x16.wgsl?raw';
import BF_16x16_lib from '../shaders/16x16/StorageBufferBased/BF-tiled-library-16x16.wgsl?raw';
import BF_16x16_subgroup from '../shaders/16x16/SubgroupBased/BF-subgroup-16x16.wgsl?raw';
import BF_32x32 from '../shaders/32x32/StorageBufferBased/BF-tiled-32x32.wgsl?raw';
import BF_32x32_lib from '../shaders/32x32/StorageBufferBased/BF-tiled-library-32x32.wgsl?raw';

// Storage texture variants
import BF_tex from '../shaders/BF-tex.wgsl?raw';
import BF_tex_8x8 from '../shaders/8x8/StorageTextureBased/BF-tex-tiled-8x8.wgsl?raw';
import BF_tex_8x8_lib from '../shaders/8x8/StorageTextureBased/BF-tex-tiled-library-8x8.wgsl?raw';
import BF_tex_16x16 from '../shaders/16x16/StorageTextureBased/BF-tex-tiled-16x16.wgsl?raw';
import BF_tex_16x16_lib from '../shaders/16x16/StorageTextureBased/BF-tex-tiled-library-16x16.wgsl?raw';
import BF_tex_16x16_subgroup from '../shaders/16x16/SubgroupBased/BF-tex-subgroup-16x16.wgsl?raw';
import BF_tex_32x32 from '../shaders/32x32/StorageTextureBased/BF-tex-tiled-32x32.wgsl?raw';
import BF_tex_32x32_lib from '../shaders/32x32/StorageTextureBased/BF-tex-tiled-library-32x32.wgsl?raw';

// Organized by storage type for easier filtering
export const bufferShaders = {
  'BF naive': { code: BF_naive, label: 'Naive BF', workgroupSize: [8, 8] },
  'BF 8x8': { code: BF_8x8, label: 'Tile 8×8', workgroupSize: [8, 8] },
  'BF 8x8 lib': { code: BF_8x8_lib, label: 'Tile 8×8 Library', workgroupSize: [8, 8] },
  'BF 16x16': { code: BF_16x16, label: 'Tile 16×16', workgroupSize: [16, 16] },
  'BF 16x16 lib': { code: BF_16x16_lib, label: 'Tile 16×16 Library', workgroupSize: [16, 16] },
  'BF 16x16 subgroup': {
    code: BF_16x16_subgroup,
    label: 'Tile 16×16 Subgroup',
    workgroupSize: [16, 16],
    requiredFeatures: ['subgroups'],
    uniformProfile: 'subgroup'
  },
  'BF 32x32': { code: BF_32x32, label: 'Tile 32×32', workgroupSize: [32, 32] },
  'BF 32x32 lib': { code: BF_32x32_lib, label: 'Tile 32×32 Library', workgroupSize: [32, 32] },
};

export const textureShaders = {
  'BF tex': { code: BF_tex, label: 'Naive BF (Tex)', workgroupSize: [8, 8] },
  'BF tex 8x8': { code: BF_tex_8x8, label: 'Tile 8×8 (Tex)', workgroupSize: [8, 8] },
  'BF tex 8x8 lib': { code: BF_tex_8x8_lib, label: 'Tile 8×8 Library (Tex)', workgroupSize: [8, 8] },
  'BF tex 16x16': { code: BF_tex_16x16, label: 'Tile 16×16 (Tex)', workgroupSize: [16, 16] },
  'BF tex 16x16 lib': { code: BF_tex_16x16_lib, label: 'Tile 16×16 Library (Tex)', workgroupSize: [16, 16] },
  'BF tex 16x16 subgroup': {
    code: BF_tex_16x16_subgroup,
    label: 'Tile 16×16 Subgroup (Tex)',
    workgroupSize: [16, 16],
    requiredFeatures: ['subgroups'],
    uniformProfile: 'subgroup'
  },
  'BF tex 32x32': { code: BF_tex_32x32, label: 'Tile 32×32 (Tex)', workgroupSize: [32, 32] },
  'BF tex 32x32 lib': { code: BF_tex_32x32_lib, label: 'Tile 32×32 Library (Tex)', workgroupSize: [32, 32] },
};

// Get all shader keys for the current storage type
function supportsShaderFeatures(shaderInfo, supportedFeatures) {
  if (!shaderInfo.requiredFeatures || shaderInfo.requiredFeatures.length === 0) {
    return true;
  }

  if (!supportedFeatures || typeof supportedFeatures.has !== 'function') {
    return true;
  }

  return shaderInfo.requiredFeatures.every((feature) => supportedFeatures.has(feature));
}

export function getShaderKeys(useTextures, supportedFeatures = null) {
  const registry = useTextures ? textureShaders : bufferShaders;
  return Object.entries(registry)
    .filter(([, info]) => supportsShaderFeatures(info, supportedFeatures))
    .map(([key]) => key);
}

// Get shader info by key
export function getShader(key, useTextures) {
  const registry = useTextures ? textureShaders : bufferShaders;
  return registry[key];
}

// Get all shaders with display labels
export function getAllShaderOptions(useTextures, supportedFeatures = null) {
  const registry = useTextures ? textureShaders : bufferShaders;
  return Object.entries(registry)
    .filter(([, info]) => supportsShaderFeatures(info, supportedFeatures))
    .map(([key, info]) => ({
      key,
      label: info.label
    }));
}
