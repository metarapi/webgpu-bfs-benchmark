enable subgroups;

// Prototype subgroup-microtile relaxer with both:
// 1) subgroup-local early-out inside each microtile, and
// 2) workgroup-level early-out across outer publish/sync phases.
//
// Assumptions:
// - Workgroup size is fixed at 16x16.
// - Haloed LDS tile is 18x18.
// - One logical center cell per invocation.
// - Microtile mapping is specialized for 16x16 workgroups.

struct Uniforms {
    gridSizeX: u32,
    gridSizeY: u32,
    maxSubgroupIterations: u32,
    maxWorkgroupIterations: u32,
    sqrt2: f32,
    epsilon: f32,
    _pad0: vec2<f32>,
};

struct SubgroupMicrotile {
    width: u32,
    height: u32,
    tilesPerRow: u32,
    tilesPerCol: u32,
};

const WG_X : u32 = 16u;
const WG_Y : u32 = 16u;
const LDS_X : u32 = WG_X + 2u;
const LDS_Y : u32 = WG_Y + 2u;
const LDS_COUNT : u32 = LDS_X * LDS_Y;
const INVOCATIONS_PER_WG : u32 = WG_X * WG_Y;
const INF : f32 = 1e20;

@group(0) @binding(0) var<storage, read> gridBuffer : array<f32>;
@group(0) @binding(1) var<storage, read> distBufferIn : array<f32>;
@group(0) @binding(2) var<storage, read_write> distBufferOut : array<f32>;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

var<workgroup> wgTerrain : array<f32, LDS_COUNT>;
var<workgroup> wgDistA   : array<f32, LDS_COUNT>;
var<workgroup> wgDistB   : array<f32, LDS_COUNT>;
var<workgroup> wgIterChanged : atomic<u32>;
var<workgroup> wgIterChangedUniform : u32;

fn subgroupMicrotileForSize(subgroupSize: u32) -> SubgroupMicrotile {
    switch subgroupSize {
        case 4u:  { return SubgroupMicrotile(2u, 2u, 8u, 8u); }
        case 8u:  { return SubgroupMicrotile(4u, 2u, 4u, 8u); }
        case 16u: { return SubgroupMicrotile(4u, 4u, 4u, 4u); }
        case 32u: { return SubgroupMicrotile(8u, 4u, 2u, 4u); }
        case 64u: { return SubgroupMicrotile(8u, 8u, 2u, 2u); }
        default:  { return SubgroupMicrotile(subgroupSize, 1u, 1u, 1u); }
    }
}

fn subgroupLaneCoord(lane: u32, tile: SubgroupMicrotile) -> vec2<u32> {
    return vec2<u32>(lane % tile.width, lane / tile.width);
}

fn subgroupOrigin(subgroupId: u32, tile: SubgroupMicrotile) -> vec2<u32> {
    let tx = subgroupId % tile.tilesPerRow;
    let ty = subgroupId / tile.tilesPerRow;
    return vec2<u32>(tx * tile.width, ty * tile.height);
}

fn subgroupWorkgroupCoord(subgroupId: u32, lane: u32, tile: SubgroupMicrotile) -> vec2<u32> {
    return subgroupOrigin(subgroupId, tile) + subgroupLaneCoord(lane, tile);
}

fn ldsIndex(x: u32, y: u32) -> u32 {
    return y * LDS_X + x;
}

fn gridIndex(x: u32, y: u32) -> u32 {
    return y * uniforms.gridSizeX + x;
}

fn inGrid(g: vec2<u32>) -> bool {
    return g.x < uniforms.gridSizeX && g.y < uniforms.gridSizeY;
}

fn loadSharedTile(tileOrigin: vec2<u32>, localIndex: u32) {
    var t = localIndex;
    loop {
        if (t >= LDS_COUNT) { break; }

        let lx = t % LDS_X;
        let ly = t / LDS_X;

        var terrain = 0.0;
        var dist = INF;

        if (tileOrigin.x + lx >= 1u && tileOrigin.y + ly >= 1u) {
            let gx = tileOrigin.x + lx - 1u;
            let gy = tileOrigin.y + ly - 1u;
            if (gx < uniforms.gridSizeX && gy < uniforms.gridSizeY) {
                let gi = gridIndex(gx, gy);
                terrain = gridBuffer[gi];
                dist = distBufferIn[gi];
            }
        }

        wgTerrain[t] = terrain;
        wgDistA[t] = dist;
        wgDistB[t] = dist;
        t += INVOCATIONS_PER_WG;
    }

    workgroupBarrier();
}

fn relaxedCost(centerTerrain: f32, diagonal: bool) -> f32 {
    var cost = select(1.0, uniforms.sqrt2, diagonal);
    if (centerTerrain < 0.0) {
        cost = cost * abs(centerTerrain);
    } else if (centerTerrain > 0.0) {
        cost = cost / centerTerrain;
    }
    return cost;
}

fn readBoundaryOrHalo(
    subgroupId: u32,
    tile: SubgroupMicrotile,
    ncoord: vec2<i32>,
    readFromB: bool
) -> f32 {
    let base = subgroupOrigin(subgroupId, tile);
    let wx = i32(base.x) + ncoord.x;
    let wy = i32(base.y) + ncoord.y;
    let ldsX = u32(wx + 1);
    let ldsY = u32(wy + 1);
    let idx = ldsIndex(ldsX, ldsY);
    return select(wgDistA[idx], wgDistB[idx], readFromB);
}

fn shuffleOrHalo(value: f32, srcLane: u32, useShuffle: bool, haloValue: f32) -> f32 {
    let safeSrcLane = select(0u, srcLane, useShuffle);
    let shuffled = subgroupShuffle(value, safeSrcLane);
    return select(haloValue, shuffled, useShuffle);
}

@compute @workgroup_size(WG_X, WG_Y)
fn main(
    @builtin(workgroup_id)           wgid : vec3<u32>,
    @builtin(local_invocation_index) lidx : u32,
    @builtin(subgroup_id)            sgid : u32,
    @builtin(subgroup_size)          sgsz : u32,
    @builtin(subgroup_invocation_id) lane : u32,
) {
    let tileOrigin = vec2<u32>(wgid.xy) * vec2<u32>(WG_X, WG_Y);
    loadSharedTile(tileOrigin, lidx);

    let micro = subgroupMicrotileForSize(sgsz);
    let wgCoord = subgroupWorkgroupCoord(sgid, lane, micro);
    let laneCoord = subgroupLaneCoord(lane, micro);
    let ownsCell = wgCoord.x < WG_X && wgCoord.y < WG_Y;

    let globalCoord = tileOrigin + wgCoord;
    let inBounds = ownsCell && inGrid(globalCoord);

    let ldsX = wgCoord.x + 1u;
    let ldsY = wgCoord.y + 1u;
    let centerIdx = ldsIndex(ldsX, ldsY);
    let terrainCenter = select(0.0, wgTerrain[centerIdx], ownsCell);

    var current = select(0.0, wgDistA[centerIdx], ownsCell);
    var readFromB = false;

    for (var workgroupIt: u32 = 0u; workgroupIt < uniforms.maxWorkgroupIterations; workgroupIt = workgroupIt + 1u) {
        if (lidx == 0u) {
            atomicStore(&wgIterChanged, 0u);
        }
        workgroupBarrier();

        if (ownsCell) {
            current = select(wgDistA[centerIdx], wgDistB[centerIdx], readFromB);
        }

        let writeToB = !readFromB;

        var subgroupActive = ownsCell;
        var oldOuterValue = current;

        for (var subgroupIt: u32 = 0u; subgroupIt < uniforms.maxSubgroupIterations; subgroupIt = subgroupIt + 1u) {
            let oldCenter = current;

            let hasL = laneCoord.x > 0u;
            let hasR = laneCoord.x + 1u < micro.width;
            let hasU = laneCoord.y > 0u;
            let hasD = laneCoord.y + 1u < micro.height;

            let lHalo  = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x) - 1, i32(laneCoord.y)), readFromB);
            let rHalo  = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x) + 1, i32(laneCoord.y)), readFromB);
            let uHalo  = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x), i32(laneCoord.y) - 1), readFromB);
            let dHalo  = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x), i32(laneCoord.y) + 1), readFromB);
            let ulHalo = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x) - 1, i32(laneCoord.y) - 1), readFromB);
            let urHalo = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x) + 1, i32(laneCoord.y) - 1), readFromB);
            let dlHalo = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x) - 1, i32(laneCoord.y) + 1), readFromB);
            let drHalo = readBoundaryOrHalo(sgid, micro, vec2<i32>(i32(laneCoord.x) + 1, i32(laneCoord.y) + 1), readFromB);

            let left      = shuffleOrHalo(current, lane - 1u,                hasL,           lHalo);
            let right     = shuffleOrHalo(current, lane + 1u,                hasR,           rHalo);
            let up        = shuffleOrHalo(current, lane - micro.width,       hasU,           uHalo);
            let down      = shuffleOrHalo(current, lane + micro.width,       hasD,           dHalo);
            let upLeft    = shuffleOrHalo(current, lane - micro.width - 1u,  hasU && hasL,   ulHalo);
            let upRight   = shuffleOrHalo(current, lane - micro.width + 1u,  hasU && hasR,   urHalo);
            let downLeft  = shuffleOrHalo(current, lane + micro.width - 1u,  hasD && hasL,   dlHalo);
            let downRight = shuffleOrHalo(current, lane + micro.width + 1u,  hasD && hasR,   drHalo);

            let terrainL  = wgTerrain[ldsIndex(ldsX - 1u, ldsY)];
            let terrainR  = wgTerrain[ldsIndex(ldsX + 1u, ldsY)];
            let terrainU  = wgTerrain[ldsIndex(ldsX, ldsY - 1u)];
            let terrainD  = wgTerrain[ldsIndex(ldsX, ldsY + 1u)];
            let terrainUL = wgTerrain[ldsIndex(ldsX - 1u, ldsY - 1u)];
            let terrainUR = wgTerrain[ldsIndex(ldsX + 1u, ldsY - 1u)];
            let terrainDL = wgTerrain[ldsIndex(ldsX - 1u, ldsY + 1u)];
            let terrainDR = wgTerrain[ldsIndex(ldsX + 1u, ldsY + 1u)];

            var best = select(INF, oldCenter, oldCenter > 0.0);
            var found = oldCenter > 0.0;

            if (left > 0.0 && terrainL != 0.0) {
                let cand = left + relaxedCost(terrainCenter, false);
                if (cand < best) { best = cand; found = true; }
            }
            if (right > 0.0 && terrainR != 0.0) {
                let cand = right + relaxedCost(terrainCenter, false);
                if (cand < best) { best = cand; found = true; }
            }
            if (up > 0.0 && terrainU != 0.0) {
                let cand = up + relaxedCost(terrainCenter, false);
                if (cand < best) { best = cand; found = true; }
            }
            if (down > 0.0 && terrainD != 0.0) {
                let cand = down + relaxedCost(terrainCenter, false);
                if (cand < best) { best = cand; found = true; }
            }
            if (upLeft > 0.0 && terrainUL != 0.0 && terrainL != 0.0 && terrainU != 0.0) {
                let cand = upLeft + relaxedCost(terrainCenter, true);
                if (cand < best) { best = cand; found = true; }
            }
            if (upRight > 0.0 && terrainUR != 0.0 && terrainR != 0.0 && terrainU != 0.0) {
                let cand = upRight + relaxedCost(terrainCenter, true);
                if (cand < best) { best = cand; found = true; }
            }
            if (downLeft > 0.0 && terrainDL != 0.0 && terrainL != 0.0 && terrainD != 0.0) {
                let cand = downLeft + relaxedCost(terrainCenter, true);
                if (cand < best) { best = cand; found = true; }
            }
            if (downRight > 0.0 && terrainDR != 0.0 && terrainR != 0.0 && terrainD != 0.0) {
                let cand = downRight + relaxedCost(terrainCenter, true);
                if (cand < best) { best = cand; found = true; }
            }

            let relaxedValue = select(oldCenter, best, found);
            let terrainValue = select(relaxedValue, 0.0, terrainCenter == 0.0);
            current = select(oldCenter, terrainValue, subgroupActive);

            let changedInner = subgroupActive && abs(current - oldCenter) > uniforms.epsilon;
            let subgroupHasChange = subgroupAny(changedInner);
            subgroupActive = subgroupActive && subgroupHasChange;
        }

        if (ownsCell) {
            if (abs(current - oldOuterValue) > uniforms.epsilon) {
                atomicStore(&wgIterChanged, 1u);
            }
            if (writeToB) {
                wgDistB[centerIdx] = current;
            } else {
                wgDistA[centerIdx] = current;
            }
        }

        workgroupBarrier();

        if (lidx == 0u) {
            wgIterChangedUniform = atomicLoad(&wgIterChanged);
        }
        workgroupBarrier();

        let workgroupHasChange = workgroupUniformLoad(&wgIterChangedUniform);
        readFromB = writeToB;
        if (workgroupHasChange == 0u) {
            break;
        }
    }

    let finalValue = select(wgDistA[centerIdx], wgDistB[centerIdx], readFromB);
    if (inBounds) {
        distBufferOut[gridIndex(globalCoord.x, globalCoord.y)] = finalValue;
    }
}