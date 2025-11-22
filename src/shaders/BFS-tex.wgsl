struct Uniforms {
    gridSizeX: u32,
    gridSizeY: u32,
    sqrt2: f32,
    sqrt2inv: f32,
};

@group(0) @binding(0) var terrainTex  : texture_storage_2d<r32float, read>;
@group(0) @binding(1) var distTexIn   : texture_storage_2d<r32float, read>;
@group(0) @binding(2) var distTexOut  : texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> uniforms : Uniforms;

// 8-connected neighbor offsets
const OFFS8 : array<vec2<i32>, 8> = array<vec2<i32>, 8>(
    vec2<i32>( 1,  0), // right
    vec2<i32>( 0,  1), // down
    vec2<i32>(-1,  0), // left
    vec2<i32>( 0, -1), // up
    vec2<i32>( 1,  1), // down-right
    vec2<i32>(-1,  1), // down-left
    vec2<i32>(-1, -1), // up-left
    vec2<i32>( 1, -1)  // up-right
);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= uniforms.gridSizeX || y >= uniforms.gridSizeY) {
        return;
    }

    let p        = vec2<i32>(i32(x), i32(y));
    let terrain  = textureLoad(terrainTex, p).x;
    let curDist  = textureLoad(distTexIn,  p).x;

    // Impassable stays 0
    if (terrain == 0.0) {
        textureStore(distTexOut, p, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // Start with current distance (or +inf if not set)
    var best  = select(1e20, curDist, curDist > 0.0);
    var found = curDist > 0.0;

    // Base move costs
    let costAx = 1.0;
    let costDg = uniforms.sqrt2;

    // Terrain scaling based on current cell
    var scale : f32 = 1.0;
    if (terrain < 0.0) {
        scale = abs(terrain);
    } else if (terrain > 0.0) {
        scale = 1.0 / terrain;
    }

    for (var n: u32 = 0u; n < 8u; n = n + 1u) {
        let o  = OFFS8[n];
        let np = p + o;

        // Bounds
        if (np.x < 0 || np.y < 0 ||
            np.x >= i32(uniforms.gridSizeX) ||
            np.y >= i32(uniforms.gridSizeY)) {
            continue;
        }

        let nTerrain = textureLoad(terrainTex, np).x;
        if (nTerrain == 0.0) {
            continue;
        }

        // Diagonal: no corner cutting
        if (n >= 4u) {
            let gate1 = vec2<i32>(p.x + o.x, p.y);
            let gate2 = vec2<i32>(p.x,       p.y + o.y);
            let g1 = textureLoad(terrainTex, gate1).x;
            let g2 = textureLoad(terrainTex, gate2).x;
            if (g1 == 0.0 || g2 == 0.0) {
                continue;
            }
        }

        let nDist = textureLoad(distTexIn, np).x;
        if (nDist <= 0.0) {
            continue;
        }

        let step = select(costAx, costDg, n >= 4u) * scale;
        let cand = nDist + step;
        if (cand < best) {
            best  = cand;
            found = true;
        }
    }

    let outVal = select(curDist, best, found);
    textureStore(distTexOut, p, vec4<f32>(outVal, 0.0, 0.0, 0.0));
}
