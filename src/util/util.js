/**
 * Utility helpers for WebGPU feature detection and device acquisition.
 * All functions are functional (no classes) and return `null` on failure.
 */


/**
 * Returns true if the browser exposes a WebGPU implementation (navigator.gpu).
 */
export function isWebGPUSupported() {
    try {
        return typeof navigator !== 'undefined' && !!navigator.gpu;
    } catch (e) {
        return false;
    }
}


/**
 * Request a WebGPU adapter and device.
 *
 * Options:
 *  - requiredFeatures: array of feature strings (optional)
 *  - requiredLimits: limits object passed to requestDevice (optional)
 *  - requestTimestamp: boolean to request timestamp-query feature (optional)
 *
 * Returns: { adapter, device } on success, or null on any failure.
 */
export async function requestWebGPUDevice(options = {}) {
    const { requiredFeatures = [], requiredLimits = {}, requestTimestamp = false } = options;

    if (!isWebGPUSupported()) return null;

    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return null;

        // If the caller asked for timestamp support, add the feature name
        const features = [...requiredFeatures];
        if (requestTimestamp && !features.includes('timestamp-query')) {
            features.push('timestamp-query');
        }

        // Check adapter feature support if caller asked for required features
        if (features && features.length) {
            for (const f of features) {
                if (!adapter.features.has(f)) {
                    // Feature missing — return null so caller can handle fallback
                    console.warn(`WebGPU adapter missing required feature: ${f}`);
                    return null;
                }
            }
        }

        const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
        if (!device) return null;

        return { adapter, device };
    } catch (err) {
        // Platform may refuse or throw; surface a warning and return null
        console.warn('requestWebGPUDevice failed:', err);
        return null;
    }
}


/* -------------------------------------------------------------------- */
/* Timestamp query helpers (compute shader timing)                      */
/* -------------------------------------------------------------------- */


/**
 * Return true if the device supports timestamp queries.
 */
export function supportsTimestampQueries(deviceOrAdapter) {
    try {
        if (!deviceOrAdapter) return false;
        // Both GPUAdapter and GPUDevice expose `features`.
        return !!deviceOrAdapter.features && deviceOrAdapter.features.has('timestamp-query');
    } catch (e) {
        return false;
    }
}


/**
 * Create a `GPUQuerySet` of type 'timestamp'. Returns null if unsupported.
 */
export function createTimestampQuerySet(device, count = 2) {
    if (!device || !supportsTimestampQueries(device)) return null;
    try {
        return device.createQuerySet({ type: 'timestamp', count: Math.max(1, count) });
    } catch (e) {
        console.warn('createTimestampQuerySet failed:', e);
        return null;
    }
}


/**
 * Resolve a range of queries from a `GPUQuerySet` into a mapped read buffer and
 * return an array of BigInt timestamp values in nanoseconds.
 *
 * Note: Timestamp values are implementation-defined and may be quantized (typically
 * to 100 microseconds) for security reasons. The values are already in nanoseconds
 * and don't require conversion.
 *
 * Usage: call `passEncoder.writeTimestamp(querySet, index)` at points to measure,
 * then call this function with the `commandEncoder` that contains the passes.
 */
export async function resolveAndReadTimestamps(device, commandEncoder, querySet, firstQuery = 0, queryCount = 1) {
    if (!device || !commandEncoder || !querySet) return null;
    const bytes = queryCount * 8; // each timestamp is a 64-bit value

    const readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    try {
        commandEncoder.resolveQuerySet(querySet, firstQuery, queryCount, readback, 0);
        const cmd = commandEncoder.finish();
        device.queue.submit([cmd]);

        await readback.mapAsync(GPUMapMode.READ);
        const arrayBuffer = readback.getMappedRange().slice(0);
        readback.unmap();

        // Parse little-endian 64-bit unsigned integers into BigInt values (nanoseconds)
        const dv = new DataView(arrayBuffer);
        const out = [];
        for (let i = 0; i < queryCount; i++) {
            const low = BigInt(dv.getUint32(i * 8, true));
            const high = BigInt(dv.getUint32(i * 8 + 4, true));
            out.push((high << 32n) + low);
        }
        return out;
    } catch (e) {
        console.warn('resolveAndReadTimestamps failed:', e);
        return null;
    }
}


/**
 * Convert timestamp difference to milliseconds.
 * Timestamps are in nanoseconds, so divide by 1,000,000.
 */
export function timestampDeltaToMillis(startTimestamp, endTimestamp) {
    if (typeof startTimestamp !== 'bigint' || typeof endTimestamp !== 'bigint') {
        return null;
    }
    const deltaNs = endTimestamp - startTimestamp;
    return Number(deltaNs) / 1_000_000;
}
