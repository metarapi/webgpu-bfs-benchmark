# webgpu-bfs-benchmark

WebGPU benchmark dashboard for comparing grid-based Bellman-Ford relaxation shaders on 2D terrain maps. [Try it here.](https://metarapi.github.io/webgpu-bfs-benchmark/)

This app runs multiple shader variants in the browser, measures performance, and visualizes both the terrain input and the resulting distance field.

## Overview

This project benchmarks Bellman-Ford style relaxation kernels on 2D grids using WebGPU.

Current variants include:
- storage buffer and storage texture implementations
- tiled 8x8, 16x16, and 32x32 workgroup variants
- library-assisted tiled variants
- a subgroup-enabled texture variant on supported GPUs (notably bypasses LDS and minimizes barrier synchronization by sharing state directly across registers)

> Note: the repository is named `bfs` by mistake. The project benchmarks Bellman-Ford style relaxation shaders, not breadth-first search.

## Features

- browser-based benchmark dashboard
- terrain heatmap and output distance-field heatmap
- configurable grid size, terrain type, cost mode, and iteration counts
- shader selection by storage mode
- GPU timestamp timing when supported
- subgroup shader support on compatible devices