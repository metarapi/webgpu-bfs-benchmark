import express from 'express';
import compression from 'compression';
import helmet from 'helmet';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3004;

// Security headers with relaxed settings for WebGPU
app.use(helmet({
    contentSecurityPolicy: false 
}));

// Enable gzip compression
app.use(compression());

// Serve static files from dist directory (including index.html)
// Some deployments (GitHub Pages) serve the site under a sub-path
// (e.g. '/webgpu-bfs-benchmark/'). Vite's `base` may add that
// prefix to built asset URLs. To make the same build work when
// serving locally, expose the `dist` both at the root and at the
// repository base path.
const repoBasePath = '/webgpu-bfs-benchmark';

app.use(express.static(path.join(__dirname, 'dist')));
app.use(repoBasePath, express.static(path.join(__dirname, 'dist')));

// SPA fallback for client-side routing — serve index.html for unknown
// routes. Use a regex-based route to avoid path parsing issues.
app.get(/.*/, (req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
    console.log(`Also serving build under ${repoBasePath}/`);
});