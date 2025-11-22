import wgpu
import numpy as np
from pathlib import Path

# ----------------------
# Common simulation setup
# ----------------------

GRID_SIZE = 2048
INNER_ITERATIONS = 1000
EARLY_OUT = 1
GOAL = (GRID_SIZE - 3, GRID_SIZE - 3)
NUM_CELLS = GRID_SIZE * GRID_SIZE
GOAL_INDEX = GOAL[1] * GRID_SIZE + GOAL[0]

# Matches the demo grid logic from the notebooks
def build_demo_grid(size: int, goal: tuple[int, int]) -> np.ndarray:
    grid = np.ones((size, size), dtype=np.float32)

    # Horizontal wall with a gap
    wall_y = size // 2
    grid[wall_y, size // 4: size - size // 6] = 0.0
    grid[wall_y, goal[0]] = 1.0  # keep the goal open

    # Vertical wall that almost splits the map
    wall_x = size // 3
    grid[size // 5: size - size // 5, wall_x] = 0.0
    grid[goal[1], wall_x] = 1.0

    # Make some areas more costly
    grid[:wall_y, wall_x:] *= 2.0
    grid[:wall_y, :wall_x] *= 4.0

    grid[goal[1], goal[0]] = 1.0  # goal must stay passable
    return grid

terrain_grid = build_demo_grid(GRID_SIZE, GOAL)
terrain_flat = terrain_grid.ravel().astype(np.float32)

# Dispatch grid is the same formula you used in all three notebooks
DISPATCH_X = (GRID_SIZE + 7) // 8
DISPATCH_Y = (GRID_SIZE + 7) // 8

# ----------------------
# WebGPU device + queries
# ----------------------

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
if adapter is None:
    raise RuntimeError("No WebGPU adapter available.")

features = adapter.features
can_timestamp = "timestamp-query" in features

required_features = []
if can_timestamp:
    required_features.append("timestamp-query")
    required_features.append("texture-adapter-specific-format-features")

device = adapter.request_device_sync(required_features=required_features)
print("Adapter:", adapter)
print("Device:", device)
print("Can timestamp:", can_timestamp)

# ----------------------
# Common buffers (uniform + terrain)
# ----------------------

uniform_dtype = np.dtype([
    ("gridSizeX", np.uint32),
    ("gridSizeY", np.uint32),
    ("sqrt2", np.float32),
    ("sqrt2inv", np.float32),
    ("iterations", np.uint32),
    ("earlyOut", np.uint32),
    ("_pad0", np.uint32),
], align=True)

uniform_data = np.array([
    (
        GRID_SIZE,
        GRID_SIZE,
        np.float32(np.sqrt(2.0)),
        np.float32(1.0 / np.sqrt(2.0)),
        INNER_ITERATIONS,
        EARLY_OUT,
        0,
    )
], dtype=uniform_dtype)

uniform_buffer = device.create_buffer_with_data(
    data=uniform_data,
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
)

grid_buffer = device.create_buffer_with_data(
    data=terrain_flat,
    usage=wgpu.BufferUsage.STORAGE
    | wgpu.BufferUsage.COPY_DST
    | wgpu.BufferUsage.COPY_SRC,
)


# ----------------------
# Storage textures for BFS-tex.wgsl
# ----------------------

# Terrain as r32float storage texture
terrain_tex = device.create_texture(
    size=(GRID_SIZE, GRID_SIZE, 1),
    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.STORAGE_BINDING,
    dimension=wgpu.TextureDimension.d2,
    format=wgpu.TextureFormat.r32float,
)
terrain_view = terrain_tex.create_view()

# Upload terrain_flat into terrain_tex
terrain_bytes = terrain_flat.tobytes()
terrain_staging = device.create_buffer_with_data(
    data=terrain_bytes,
    usage=wgpu.BufferUsage.COPY_SRC,
)
enc = device.create_command_encoder()
enc.copy_buffer_to_texture(
    {
        "buffer": terrain_staging,
        "offset": 0,
        "bytes_per_row": GRID_SIZE * 4,
        "rows_per_image": GRID_SIZE,
    },
    {
        "texture": terrain_tex,
        "mip_level": 0,
        "origin": (0, 0, 0),
    },
    (GRID_SIZE, GRID_SIZE, 1),
)
device.queue.submit([enc.finish()])

def make_distance_texture_seed():
    """Create a distance storage texture seeded at GOAL."""
    distance_seed = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    distance_seed[GOAL[1], GOAL[0]] = 1.0
    seed_bytes = distance_seed.tobytes()

    tex = device.create_texture(
        size=(GRID_SIZE, GRID_SIZE, 1),
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
        dimension=wgpu.TextureDimension.d2,
        format=wgpu.TextureFormat.r32float,
    )
    view = tex.create_view()

    staging = device.create_buffer_with_data(
        data=seed_bytes,
        usage=wgpu.BufferUsage.COPY_SRC,
    )
    e = device.create_command_encoder()
    e.copy_buffer_to_texture(
        {
            "buffer": staging,
            "offset": 0,
            "bytes_per_row": GRID_SIZE * 4,
            "rows_per_image": GRID_SIZE,
        },
        {
            "texture": tex,
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        (GRID_SIZE, GRID_SIZE, 1),
    )
    device.queue.submit([e.finish()])
    return tex, view

def read_distance_texture(tex: wgpu.GPUTexture) -> np.ndarray:
    """Read back an r32f storage texture into (GRID_SIZE, GRID_SIZE)."""
    out_buffer = device.create_buffer(
        size=terrain_flat.nbytes,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    e = device.create_command_encoder()
    e.copy_texture_to_buffer(
        {
            "texture": tex,
            "mip_level": 0,
            "origin": (0, 0, 0),
        },
        {
            "buffer": out_buffer,
            "offset": 0,
            "bytes_per_row": GRID_SIZE * 4,
            "rows_per_image": GRID_SIZE,
        },
        (GRID_SIZE, GRID_SIZE, 1),
    )
    device.queue.submit([e.finish()])

    out_buffer.map_sync(mode=wgpu.MapMode.READ)
    data = out_buffer.read_mapped()
    arr = np.frombuffer(data, dtype=np.float32).copy()
    out_buffer.unmap()
    return arr.reshape((GRID_SIZE, GRID_SIZE))


def make_distance_buffers():
    """Create fresh ping/pong distance buffers seeded at the GOAL."""
    distance_seed = np.zeros(NUM_CELLS, dtype=np.float32)
    distance_seed[GOAL_INDEX] = 1.0

    distance_buffer_ping = device.create_buffer_with_data(
        data=distance_seed,
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )

    distance_buffer_pong = device.create_buffer_with_data(
        data=distance_seed,
        usage=wgpu.BufferUsage.STORAGE
        | wgpu.BufferUsage.COPY_SRC
        | wgpu.BufferUsage.COPY_DST,
    )

    return distance_seed, distance_buffer_ping, distance_buffer_pong


def read_distance_buffer(buffer: wgpu.GPUBuffer) -> np.ndarray:
    """Read back a distance field buffer into a (GRID_SIZE, GRID_SIZE) array."""
    staging_buffer = device.create_buffer(
        size=terrain_flat.nbytes,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(
        buffer, 0, staging_buffer, 0, terrain_flat.nbytes
    )
    device.queue.submit([command_encoder.finish()])

    staging_buffer.map_sync(mode=wgpu.MapMode.READ)
    data = staging_buffer.read_mapped()
    result = np.frombuffer(data, dtype=np.float32).copy()
    staging_buffer.unmap()
    return result.reshape((GRID_SIZE, GRID_SIZE))


def make_bfs_pipeline(shader_path: str):
    """Create pipeline + bind-group layout for one BFS shader."""
    p = Path(shader_path)
    if not p.exists():
        # Resolve relative to this script's directory (profiler.py),
        # which is the expected location for the tmp/Production-shaders paths.
        p = Path(__file__).parent.joinpath(shader_path)
    shader_source = p.read_text()
    shader_module = device.create_shader_module(code=shader_source)

    bfs_bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.storage},
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    bfs_pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bfs_bind_group_layout]
    )

    bfs_pipeline = device.create_compute_pipeline(
        layout=bfs_pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )

    return bfs_pipeline, bfs_bind_group_layout

# ----------------------
# Pipeline builder for BFS-tex.wgsl
# ----------------------
def make_bfs_tex_pipeline(shader_path: str):
    """Create pipeline + bind-group layout for BFS-tex.wgsl (storage textures)."""
    p = Path(shader_path)
    if not p.exists():
        p = Path(__file__).parent.joinpath(shader_path)
    shader_source = p.read_text()
    shader_module = device.create_shader_module(code=shader_source)

    bfs_tex_bgl = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.read_only,
                    "format": wgpu.TextureFormat.r32float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.read_only,
                    "format": wgpu.TextureFormat.r32float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": wgpu.StorageTextureAccess.write_only,
                    "format": wgpu.TextureFormat.r32float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]
    )

    bfs_tex_pl = device.create_pipeline_layout(bind_group_layouts=[bfs_tex_bgl])

    bfs_tex_pipeline = device.create_compute_pipeline(
        layout=bfs_tex_pl,
        compute={"module": shader_module, "entry_point": "main"},
    )

    return bfs_tex_pipeline, bfs_tex_bgl


def run_bfs_variant(name: str, shader_path: str, iterations: int):
    print(f"\n=== {name} ===")
    print(f"Shader: {shader_path}, iterations: {iterations}")

    bfs_pipeline, bfs_bgl = make_bfs_pipeline(shader_path)
    distance_seed, distance_buffer_ping, distance_buffer_pong = make_distance_buffers()

    bind_group_ping = device.create_bind_group(
        layout=bfs_bgl,
        entries=[
            {"binding": 0, "resource": {"buffer": grid_buffer, "offset": 0, "size": terrain_flat.nbytes}},
            {"binding": 1, "resource": {"buffer": distance_buffer_ping, "offset": 0, "size": distance_seed.nbytes}},
            {"binding": 2, "resource": {"buffer": distance_buffer_pong, "offset": 0, "size": distance_seed.nbytes}},
            {"binding": 3, "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_data.nbytes}},
        ],
    )
    bind_group_pong = device.create_bind_group(
        layout=bfs_bgl,
        entries=[
            {"binding": 0, "resource": {"buffer": grid_buffer, "offset": 0, "size": terrain_flat.nbytes}},
            {"binding": 1, "resource": {"buffer": distance_buffer_pong, "offset": 0, "size": distance_seed.nbytes}},
            {"binding": 2, "resource": {"buffer": distance_buffer_ping, "offset": 0, "size": distance_seed.nbytes}},
            {"binding": 3, "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_data.nbytes}},
        ],
    )

    # Timestamp setup
    if can_timestamp:
        query_count = 2 * iterations
        # Resolve offset must be 256-byte aligned; write each 64-bit result at multiples of 256
        slot_stride = 256
        resolve_buffer_size = query_count * slot_stride
        query_set = device.create_query_set(type="timestamp", count=query_count)

        resolve_buffer = device.create_buffer(
            size=resolve_buffer_size,
            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
        )
        timestamp_read_buffer = device.create_buffer(
            size=resolve_buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        timestamp_ticks = None
    else:
        query_set = None
        resolve_buffer = None
        timestamp_read_buffer = None
        timestamp_ticks = None

    # Build ONE command encoder for all iterations
    encoder = device.create_command_encoder()

    read_from_ping = True
    last_output = distance_buffer_ping

    for step in range(iterations):
        if can_timestamp:
            q_start = 2 * step
            q_end = q_start + 1
            pass_desc = {
                "timestamp_writes": {
                    "query_set": query_set,
                    "beginning_of_pass_write_index": q_start,
                    "end_of_pass_write_index": q_end,
                }
            }
            compute_pass = encoder.begin_compute_pass(**pass_desc)
        else:
            compute_pass = encoder.begin_compute_pass()

        compute_pass.set_pipeline(bfs_pipeline)
        compute_pass.set_bind_group(0, bind_group_ping if read_from_ping else bind_group_pong)
        compute_pass.dispatch_workgroups(DISPATCH_X, DISPATCH_Y)
        compute_pass.end()

        last_output = distance_buffer_pong if read_from_ping else distance_buffer_ping
        read_from_ping = not read_from_ping

    # Resolve all timestamps once at the end of the encoder
    if can_timestamp:
        # Write each query into its own 256-byte slot
        for qi in range(2 * iterations):
            encoder.resolve_query_set(
                query_set,
                qi,
                1,
                resolve_buffer,
                qi * 256,
            )

    # Submit the whole batch at once
    device.queue.submit([encoder.finish()])
    device.queue.on_submitted_work_done()

    # Read timestamps back
    if can_timestamp:
        copy_encoder = device.create_command_encoder()
        copy_encoder.copy_buffer_to_buffer(resolve_buffer, 0, timestamp_read_buffer, 0, resolve_buffer_size)
        device.queue.submit([copy_encoder.finish()])
        device.queue.on_submitted_work_done()

        timestamp_read_buffer.map_sync(mode=wgpu.MapMode.READ)
        raw = timestamp_read_buffer.read_mapped()
        # One u64 at the start of each 256-byte slot
        all_u64 = np.frombuffer(raw, dtype=np.uint64)
        times = all_u64[::32].copy()  # 256 bytes / 8 bytes = 32 u64 per slot
        timestamp_read_buffer.unmap()

        # per-iteration diffs
        timestamp_ticks = times[1::2] - times[0:2 * iterations:2]

        # Convert to milliseconds.
        # WebGPU timestamp query values are reported as 64-bit integers representing
        # nanoseconds relative to an implementation-defined epoch. Some backends may
        # expose a timestamp period but the raw resolves are already in nanoseconds
        # on many implementations. We therefore treat the resolved u64 as nanoseconds
        # and convert to ms by multiplying with 1e-6. If a driver provides a different
        # period via get_timestamp_period(), apply it as ns-per-tick.
        try:
            # If available, get the adapter/queue-provided period (ns per tick).
            period_ns = device.queue.get_timestamp_period()
        except Exception:
            # Fall back to 1.0 (i.e., resolved values are already nanoseconds).
            period_ns = 1.0

        # period_ns is nanoseconds per tick; convert ticks -> ms: ticks * period_ns / 1e6
        per_iter_ms = timestamp_ticks.astype(np.float64) * (period_ns / 1e6)
        nz = per_iter_ms[per_iter_ms > 0.0]
        if nz.size:
            print(f"Average per iteration: {nz.mean():.4f} ms")
        else:
            print("Average per iteration: 0 ms")
        print(f"Total GPU time: {per_iter_ms.sum():.4f} ms")
    else:
        print("Timestamp queries not supported on this adapter.")

    # Optional: read back for correctness
    distance_field = read_distance_buffer(last_output)
    positive = distance_field[distance_field > 0.0]
    print(f"Finished {iterations} iterations. Filled cells: {positive.size}/{NUM_CELLS}.")

    return distance_field, timestamp_ticks



# ----------------------
# Run BFS-tex.wgsl variant (storage textures)
# ----------------------
def run_bfs_tex_variant(name: str, shader_path: str, iterations: int):
    print(f"\n=== {name} ===")
    print(f"Shader: {shader_path}, iterations: {iterations}")

    bfs_pipeline, bfs_bgl = make_bfs_tex_pipeline(shader_path)

    # Per-variant distance textures (ping/pong)
    dist_tex_ping, dist_view_ping = make_distance_texture_seed()
    dist_tex_pong, dist_view_pong = make_distance_texture_seed()

    bind_group_ping = device.create_bind_group(
        layout=bfs_bgl,
        entries=[
            {"binding": 0, "resource": terrain_view},
            {"binding": 1, "resource": dist_view_ping},
            {"binding": 2, "resource": dist_view_pong},
            {"binding": 3, "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_data.nbytes}},
        ],
    )
    bind_group_pong = device.create_bind_group(
        layout=bfs_bgl,
        entries=[
            {"binding": 0, "resource": terrain_view},
            {"binding": 1, "resource": dist_view_pong},
            {"binding": 2, "resource": dist_view_ping},
            {"binding": 3, "resource": {"buffer": uniform_buffer, "offset": 0, "size": uniform_data.nbytes}},
        ],
    )

    # Timestamp setup (identical to run_bfs_variant)
    if can_timestamp:
        query_count = 2 * iterations
        slot_stride = 256
        resolve_buffer_size = query_count * slot_stride
        query_set = device.create_query_set(type="timestamp", count=query_count)

        resolve_buffer = device.create_buffer(
            size=resolve_buffer_size,
            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
        )
        timestamp_read_buffer = device.create_buffer(
            size=resolve_buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        timestamp_ticks = None
    else:
        query_set = None
        resolve_buffer = None
        timestamp_read_buffer = None
        timestamp_ticks = None

    encoder = device.create_command_encoder()

    read_from_ping = True

    for step in range(iterations):
        if can_timestamp:
            q_start = 2 * step
            q_end = q_start + 1
            pass_desc = {
                "timestamp_writes": {
                    "query_set": query_set,
                    "beginning_of_pass_write_index": q_start,
                    "end_of_pass_write_index": q_end,
                }
            }
            compute_pass = encoder.begin_compute_pass(**pass_desc)
        else:
            compute_pass = encoder.begin_compute_pass()

        compute_pass.set_pipeline(bfs_pipeline)
        compute_pass.set_bind_group(0, bind_group_ping if read_from_ping else bind_group_pong)
        compute_pass.dispatch_workgroups(DISPATCH_X, DISPATCH_Y)
        compute_pass.end()

        read_from_ping = not read_from_ping

    if can_timestamp:
        for qi in range(2 * iterations):
            encoder.resolve_query_set(
                query_set,
                qi,
                1,
                resolve_buffer,
                qi * 256,
            )

    device.queue.submit([encoder.finish()])
    device.queue.on_submitted_work_done()

    if can_timestamp:
        copy_encoder = device.create_command_encoder()
        copy_encoder.copy_buffer_to_buffer(resolve_buffer, 0, timestamp_read_buffer, 0, resolve_buffer_size)
        device.queue.submit([copy_encoder.finish()])
        device.queue.on_submitted_work_done()

        timestamp_read_buffer.map_sync(mode=wgpu.MapMode.READ)
        raw = timestamp_read_buffer.read_mapped()
        all_u64 = np.frombuffer(raw, dtype=np.uint64)
        times = all_u64[::32].copy()
        timestamp_read_buffer.unmap()

        timestamp_ticks = times[1::2] - times[0:2 * iterations:2]

        try:
            period_ns = device.queue.get_timestamp_period()
        except Exception:
            period_ns = 1.0
        per_iter_ms = timestamp_ticks.astype(np.float64) * (period_ns / 1e6)
        nz = per_iter_ms[per_iter_ms > 0.0]
        print(f"Average per iteration: {nz.mean():.4f} ms" if nz.size else "Average per iteration: 0 ms")
        print(f"Total GPU time: {per_iter_ms.sum():.4f} ms")
    else:
        print("Timestamp queries not supported on this adapter.")

    # Last write target:
    last_tex = dist_tex_pong if not read_from_ping else dist_tex_ping
    distance_field = read_distance_texture(last_tex)
    positive = distance_field[distance_field > 0.0]
    print(f"Finished {iterations} iterations. Filled cells: {positive.size}/{NUM_CELLS}.")

    return distance_field, timestamp_ticks

# ----------------------
# Run all three variants
# ----------------------

# Use whatever global iteration counts you consider "apples to apples".
# These default to the notebook values you had:
#   - BFS.wgsl: 1024 iterations
#   - BFS-tiled.wgsl: 32 iterations
#   - BFS-tiled-library.wgsl: 128 iterations
# Point variants to shader files exported into tmp/Production-shaders.
# profiler.py lives in tmp/Streamlined, so the relative path to Production-shaders is ../Production-shaders
# All now use early out

# variants = [
#     # Storage buffer based variants:
#     ("BFS naive",               "../Production-shaders/BFS.wgsl", 50),
#     ("BFS tiled 8x8",               "../Production-shaders/8x8/StorageBufferBased/BFS-tiled-8x8.wgsl", 50),
#     ("BFS tiled + library 8x8",     "../Production-shaders/8x8/StorageBufferBased/BFS-tiled-library-8x8.wgsl", 50),
#     ("BFS tiled 16x16",               "../Production-shaders/16x16/StorageBufferBased/BFS-tiled-16x16.wgsl", 50),
#     ("BFS tiled + library 16x16",     "../Production-shaders/16x16/StorageBufferBased/BFS-tiled-library-16x16.wgsl", 50),
#     ("BFS tiled 32x32",               "../Production-shaders/32x32/StorageBufferBased/BFS-tiled-32x32.wgsl", 50),
#     ("BFS tiled + library 32x32",     "../Production-shaders/32x32/StorageBufferBased/BFS-tiled-library-32x32.wgsl", 50),
#     # Storage texture based variants:
#     ("BFS tex naive",           "../Production-shaders/BFS-tex.wgsl", 50),
#     ("BFS tex tiled 8x8",           "../Production-shaders/8x8/StorageTextureBased/BFS-tex-tiled-8x8.wgsl", 50),
#     ("BFS tex tiled + library 8x8", "../Production-shaders/8x8/StorageTextureBased/BFS-tex-tiled-library-8x8.wgsl", 50),
#     ("BFS tex tiled 16x16",           "../Production-shaders/16x16/StorageTextureBased/BFS-tex-tiled-16x16.wgsl", 50),
#     ("BFS tex tiled + library 16x16", "../Production-shaders/16x16/StorageTextureBased/BFS-tex-tiled-library-16x16.wgsl", 50),
#     ("BFS tex tiled 32x32",           "../Production-shaders/32x32/StorageTextureBased/BFS-tex-tiled-32x32.wgsl", 50),
#     ("BFS tex tiled + library 32x32", "../Production-shaders/32x32/StorageTextureBased/BFS-tex-tiled-library-32x32.wgsl", 50),
# ]

# Reduction vs atomics
# variants = [
#     # Storage buffer based variants:
#     ("BFS naive",                   "../Production-shaders/BFS.wgsl", 50),
#     ("BFS tiled 16x16",             "../Production-shaders/16x16/StorageBufferBased/BFS-tiled-16x16.wgsl", 100),
#     ("BFS tiled reduction 16x16",   "../Production-shaders/16x16/StorageBufferBased/BFS-tiled-16x16-reduction.wgsl", 100),
#     ("BFS tiled 32x32",             "../Production-shaders/32x32/StorageBufferBased/BFS-tiled-32x32.wgsl", 50),
#     ("BFS tiled reduction 32x32",   "../Production-shaders/32x32/StorageBufferBased/BFS-tiled-32x32-reduction.wgsl", 50),
# ]

i = 370 

# variants = [
#     # Storage buffer based variants:
#     ("BFS tiled 8x8",   "../Production-shaders/8x8/StorageBufferBased/BFS-tiled-8x8.wgsl", i),
#     ("BFS tiled 16x16", "../Production-shaders/16x16/StorageBufferBased/BFS-tiled-16x16.wgsl", i),
#     ("BFS tiled 32x32", "../Production-shaders/32x32/StorageBufferBased/BFS-tiled-32x32.wgsl", i),
# ]

variants = [
    ("BFS tiled 8x8",     "../Production-shaders/8x8/StorageBufferBased/BFS-tiled-8x8.wgsl", i),
    ("BFS tiled + library 8x8",     "../Production-shaders/8x8/StorageBufferBased/BFS-tiled-library-8x8.wgsl", i),
    ("BFS tex tiled 8x8", "../Production-shaders/8x8/StorageTextureBased/BFS-tex-tiled-8x8.wgsl", i),
    ("BFS tex tiled + library 8x8", "../Production-shaders/8x8/StorageTextureBased/BFS-tex-tiled-library-8x8.wgsl", i),
]

results = {}
for name, shader, iters in variants:
    if "tex" in name:
        field, ticks = run_bfs_tex_variant(name, shader, iters)
    else:
        field, ticks = run_bfs_variant(name, shader, iters)
    results[name] = {"distance_field": field, "ticks": ticks}

# 'results' now holds distance fields and timing info for all four.
