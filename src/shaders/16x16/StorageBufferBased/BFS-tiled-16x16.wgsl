// Uniforms and constants
struct Uniforms {
  gridSizeX: u32,
  gridSizeY: u32,
  sqrt2: f32,
  sqrt2inv: f32,
  iterations: u32,
  earlyOut: u32,
  _pad0: u32,
};

// Values to try:
// 8x8:   12
// 16x16: 23
// 32x32: 46

const WORKGROUP_SIZE : u32 = 16u;
// Number of inner iterations is now set by uniforms.iterations

// Derived tile constants
const W : u32 = WORKGROUP_SIZE;            // interior width/height
const S : u32 = WORKGROUP_SIZE + 2u;       // haloed side (1-cell halo on each side)
const T : u32 = S * S;                     // total haloed tile elements
const N : u32 = W * W;                     // threads per workgroup (matches interior)

// Bindings
@group(0) @binding(0) var<storage, read>        gridBuffer: array<f32>;
@group(0) @binding(1) var<storage, read>        distanceBufferIn: array<f32>;
@group(0) @binding(2) var<storage, read_write>  distanceBufferOut: array<f32>;
@group(0) @binding(3) var<uniform>              uniforms: Uniforms;

// Indexing helper (row-major)
fn idx(x: u32, y: u32) -> u32 {
  return y * uniforms.gridSizeX + x;
}

// Neighbor offsets (8-connected)
const OFFS8 : array<vec2<i32>, 8> = array<vec2<i32>, 8>(
  vec2<i32>( 1,  0), vec2<i32>( 0,  1), vec2<i32>(-1,  0), vec2<i32>( 0, -1),
  vec2<i32>( 1,  1), vec2<i32>(-1,  1), vec2<i32>(-1, -1), vec2<i32>( 1, -1)
);

// Shared tiles
var<workgroup> wgTerrainA : array<f32, T>; // read-only snapshot for terrain
var<workgroup> wgDistA    : array<f32, T>; // distances ping-pong A/B
var<workgroup> wgDistB    : array<f32, T>;
var<workgroup> wgIterChanged : atomic<u32>; // early-out flag for tile convergence
var<workgroup> wgIterChangedUniform : u32;  // broadcast of atomic load

// Convert interior (tx,ty) in [1..W]x[1..W] to linear haloed index
fn tIndex(tx: u32, ty: u32) -> u32 {
  return ty * S + tx;
}

// Cooperative load: fill haloed tile for terrain and initial distance
fn loadTile(
  tile_origin_x: u32,
  tile_origin_y: u32,
  local_index: u32
) {
  var t = local_index;
  loop {
    if (t >= T) { break; }

    // Map linear t into haloed tile (tx,ty) in [0..S-1]
    let tx = t % S;
    let ty = t / S;

    // Map haloed tile to global coords (offset -1 for halo)
    let gx = tile_origin_x + tx - 1u;
    let gy = tile_origin_y + ty - 1u;
    let ti = t;

    var terrain : f32 = 0.0;
    var dist     : f32 = 1e20;

    if (gx < uniforms.gridSizeX && gy < uniforms.gridSizeY) {
      let gi = idx(gx, gy);
      terrain = gridBuffer[gi];
      dist    = distanceBufferIn[gi];
    }

    wgTerrainA[ti] = terrain;
    wgDistA[ti]    = dist;
    wgDistB[ti]    = dist; // mirror halo into B before ping-ponging

    t += N; // stride by the number of threads to cover all T entries
  }

  workgroupBarrier(); // make shared tile visible before sweeps
}

// One relaxation sweep using shared A/B selected by parity
fn relax_one_parity(
  tx: u32, ty: u32,
  parity: u32   // 0: read A write B, 1: read B write A
) {
  let ti      = tIndex(tx, ty);
  let terrain = wgTerrainA[ti];

  // Freeze impassable cells
  if (terrain == 0.0) {
    if (parity == 0u) {
      wgDistB[ti] = 0.0;
    } else {
      wgDistA[ti] = 0.0;
    }
    return;
  }

  // Select src/dst shared arrays by parity
  let curRaw = select(wgDistA[ti], wgDistB[ti], parity == 1u);
  var best   = select(1e20, curRaw, curRaw > 0.0);
  var found  = curRaw > 0.0;

  for (var n: u32 = 0u; n < 8u; n = n + 1u) {
    let o   = OFFS8[n];
    let ntx = i32(tx) + o.x;
    let nty = i32(ty) + o.y;
    let nti = tIndex(u32(ntx), u32(nty));

    let nTerrain = wgTerrainA[nti];
    let nDist    = select(wgDistA[nti], wgDistB[nti], parity == 1u);

    if (nDist > 0.0 && nTerrain != 0.0) {
      if (abs(o.x) + abs(o.y) == 2) {
        let gate1 = tIndex(u32(i32(tx) + o.x), ty);
        let gate2 = tIndex(tx, u32(i32(ty) + o.y));
        if (wgTerrainA[gate1] == 0.0 || wgTerrainA[gate2] == 0.0) { continue; }
      }

      var cost = select(1.0, uniforms.sqrt2, n >= 4u);
      if (terrain < 0.0) { cost = cost * abs(terrain); }
      else if (terrain > 0.0) { cost = cost / terrain; }

      let cand = nDist + cost;
      if (cand < best) {
        best = cand;
        found = true;
      }
    }
  }

  // Write to the destination buffer selected by parity
  let outVal = select(curRaw, best, found);

  // Detect per-cell change and set tile flag if changed enough
  if (found) {
    let delta = abs(outVal - curRaw);
    if (delta > 1e-6) {
      atomicStore(&wgIterChanged, 1u);
    }
  }

  if (parity == 0u) {
    wgDistB[ti] = outVal;
  } else {
    wgDistA[ti] = outVal;
  }
}

// Fixed-count shared-memory relaxation (no early-out, no atomics)
// Non-early-out variant (original behaviour)
fn relax_tile_fixed_no_early(r: u32, local_id: vec3<u32>) -> u32 {
  let tx = local_id.x + 1u;
  let ty = local_id.y + 1u;

  var parity : u32 = 0u;

  for (var it: u32 = 0u; it < r; it = it + 1u) {
    relax_one_parity(tx, ty, parity);
    workgroupBarrier();          // ensure writes visible to next iteration
    parity = 1u - parity;        // swap A/B roles
  }

  return 1u - parity;            // after loop, last writes are in the opposite of next parity
}

// Early-out variant using a per-workgroup atomic change flag
fn relax_tile_fixed_early(r: u32, local_id: vec3<u32>) -> u32 {
  let tx = local_id.x + 1u;
  let ty = local_id.y + 1u;

  var parity : u32 = 0u;

  for (var it: u32 = 0u; it < r; it = it + 1u) {
    // Reset change flag at the beginning of this sweep
    if (local_id.x == 0u && local_id.y == 0u) {
      atomicStore(&wgIterChanged, 0u);
    }
    workgroupBarrier();

    // Do one sweep
    relax_one_parity(tx, ty, parity);
    workgroupBarrier(); // ensure writes visible

    // Check if any cell changed in this sweep (must stay uniform)
    if (local_id.x == 0u && local_id.y == 0u) {
      wgIterChangedUniform = atomicLoad(&wgIterChanged);
    }
    workgroupBarrier();
    let changed = workgroupUniformLoad(&wgIterChangedUniform);
    if (changed == 0u) {
      // No changes in this iteration: tile converged, break out early
      break;
    }

    // Swap A/B roles
    parity = 1u - parity;
  }

  return 1u - parity;            // after loop, last writes are in opposite of next parity
}

// Dispatcher: choose early-out or non-early-out implementation based on uniform
fn relax_tile_fixed(r: u32, local_id: vec3<u32>) -> u32 {
  if (uniforms.earlyOut != 0u) {
    return relax_tile_fixed_early(r, local_id);
  } else {
    return relax_tile_fixed_no_early(r, local_id);
  }
}

// Cooperative write-back of the WxW interior from shared to global
fn storeTile(
  tile_origin_x: u32,
  tile_origin_y: u32,
  finalParity: u32,
  local_index: u32
) {
  const interiorCount : u32 = W * W;

  var t = local_index; // 0..N-1
  loop {
    if (t >= interiorCount) { break; }

    // Map linear interior index to interior coords [1..W]x[1..W]
    let itx = (t % W) + 1u;
    let ity = (t / W) + 1u;

    // Tile-space to global coords
    let gx = tile_origin_x + itx - 1u;
    let gy = tile_origin_y + ity - 1u;

    // Bounds check (for right/bottom edges)
    let inBounds = gx < uniforms.gridSizeX && gy < uniforms.gridSizeY;

    // Read from the correct final shared buffer
    let ti = ity * S + itx;
    let valA = wgDistA[ti];
    let valB = wgDistB[ti];
    let val  = select(valA, valB, finalParity == 0u); // if finalParity==0, B holds latest

    if (inBounds) {
      let gi = idx(gx, gy);
      distanceBufferOut[gi] = val;
    }

    t += N;
  }
}

// Entry point: orchestrate tile load, fixed-count sweeps, and write-back
@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main(
  @builtin(workgroup_id)           wg_id : vec3<u32>,
  @builtin(local_invocation_id)    lid   : vec3<u32>,
  @builtin(local_invocation_index) lidx  : u32
) {
  // Derive this workgroup's tile origin in global coordinates
  let tile_origin_x = wg_id.x * W;
  let tile_origin_y = wg_id.y * W;

  // 1) Cooperative load of haloed tile snapshot into shared memory
  loadTile(tile_origin_x, tile_origin_y, lidx);

  // 2) r inner sweeps entirely in shared memory (fixed count, no early-out)
  //    Now set by uniforms.iterations
  let r : u32 = uniforms.iterations;
  let finalParity = relax_tile_fixed(r, lid);

  // 3) Cooperative write-back of the WxW interior from final shared buffer
  storeTile(tile_origin_x, tile_origin_y, finalParity, lidx);
}
