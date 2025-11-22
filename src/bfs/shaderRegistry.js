/**
 * Shader path registry - import with ?raw suffix for strings
 */

// Storage buffer variants
import BFS_naive from '../shaders/BFS.wgsl?raw';
import BFS_8x8 from '../shaders/8x8/StorageBufferBased/BFS-tiled-8x8.wgsl?raw';
import BFS_8x8_lib from '../shaders/8x8/StorageBufferBased/BFS-tiled-library-8x8.wgsl?raw';
import BFS_16x16 from '../shaders/16x16/StorageBufferBased/BFS-tiled-16x16.wgsl?raw';
import BFS_16x16_lib from '../shaders/16x16/StorageBufferBased/BFS-tiled-library-16x16.wgsl?raw';
import BFS_32x32 from '../shaders/32x32/StorageBufferBased/BFS-tiled-32x32.wgsl?raw';
import BFS_32x32_lib from '../shaders/32x32/StorageBufferBased/BFS-tiled-library-32x32.wgsl?raw';

// Storage texture variants
import BFS_tex from '../shaders/BFS-tex.wgsl?raw';
import BFS_tex_8x8 from '../shaders/8x8/StorageTextureBased/BFS-tex-tiled-8x8.wgsl?raw';
import BFS_tex_8x8_lib from '../shaders/8x8/StorageTextureBased/BFS-tex-tiled-library-8x8.wgsl?raw';
import BFS_tex_16x16 from '../shaders/16x16/StorageTextureBased/BFS-tex-tiled-16x16.wgsl?raw';
import BFS_tex_16x16_lib from '../shaders/16x16/StorageTextureBased/BFS-tex-tiled-library-16x16.wgsl?raw';
import BFS_tex_32x32 from '../shaders/32x32/StorageTextureBased/BFS-tex-tiled-32x32.wgsl?raw';
import BFS_tex_32x32_lib from '../shaders/32x32/StorageTextureBased/BFS-tex-tiled-library-32x32.wgsl?raw';

// Organized by storage type for easier filtering
export const bufferShaders = {
  'BFS naive': { code: BFS_naive, label: 'Naive BFS', workgroupSize: [8, 8] },
  'BFS 8x8': { code: BFS_8x8, label: 'Tile 8×8', workgroupSize: [8, 8] },
  'BFS 8x8 lib': { code: BFS_8x8_lib, label: 'Tile 8×8 Library', workgroupSize: [8, 8] },
  'BFS 16x16': { code: BFS_16x16, label: 'Tile 16×16', workgroupSize: [16, 16] },
  'BFS 16x16 lib': { code: BFS_16x16_lib, label: 'Tile 16×16 Library', workgroupSize: [16, 16] },
  'BFS 32x32': { code: BFS_32x32, label: 'Tile 32×32', workgroupSize: [32, 32] },
  'BFS 32x32 lib': { code: BFS_32x32_lib, label: 'Tile 32×32 Library', workgroupSize: [32, 32] },
};

export const textureShaders = {
  'BFS tex': { code: BFS_tex, label: 'Naive BFS (Tex)', workgroupSize: [8, 8] },
  'BFS tex 8x8': { code: BFS_tex_8x8, label: 'Tile 8×8 (Tex)', workgroupSize: [8, 8] },
  'BFS tex 8x8 lib': { code: BFS_tex_8x8_lib, label: 'Tile 8×8 Library (Tex)', workgroupSize: [8, 8] },
  'BFS tex 16x16': { code: BFS_tex_16x16, label: 'Tile 16×16 (Tex)', workgroupSize: [16, 16] },
  'BFS tex 16x16 lib': { code: BFS_tex_16x16_lib, label: 'Tile 16×16 Library (Tex)', workgroupSize: [16, 16] },
  'BFS tex 32x32': { code: BFS_tex_32x32, label: 'Tile 32×32 (Tex)', workgroupSize: [32, 32] },
  'BFS tex 32x32 lib': { code: BFS_tex_32x32_lib, label: 'Tile 32×32 Library (Tex)', workgroupSize: [32, 32] },
};

// Get all shader keys for the current storage type
export function getShaderKeys(useTextures) {
  return Object.keys(useTextures ? textureShaders : bufferShaders);
}

// Get shader info by key
export function getShader(key, useTextures) {
  const registry = useTextures ? textureShaders : bufferShaders;
  return registry[key];
}

// Get all shaders with display labels
export function getAllShaderOptions(useTextures) {
  const registry = useTextures ? textureShaders : bufferShaders;
  return Object.entries(registry).map(([key, info]) => ({
    key,
    label: info.label
  }));
}
