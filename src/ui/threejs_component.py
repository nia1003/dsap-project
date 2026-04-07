"""
Generates a self-contained Three.js HTML scene for embedding point cloud.

Points are PCA-projected to 3D. Each speaker gets a unique color.
On query, the query point pulses and Top-K neighbours glow with connecting lines.
"""

import json
import numpy as np
from sklearn.decomposition import PCA


def _to_hex(rgb: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


def _speaker_colors(n: int) -> list[str]:
    """Generate visually distinct colors for n speakers using HSL."""
    colors = []
    for i in range(n):
        h = i / n
        # Convert HSL(h, 0.7, 0.6) → RGB
        import colorsys
        r, g, b = colorsys.hls_to_rgb(h, 0.6, 0.7)
        colors.append(_to_hex((r, g, b)))
    return colors


def build_threejs_html(
    embeddings: np.ndarray,
    labels: np.ndarray,
    speaker_ids: list[str],
    query_idx: int | None = None,
    neighbor_indices: np.ndarray | None = None,
    height: int = 600,
) -> str:
    """
    Returns a self-contained HTML string with a Three.js 3D point cloud.

    Args:
        embeddings:      (N, D) float32
        labels:          (N,)   int — speaker index
        speaker_ids:     list of speaker name strings
        query_idx:       index of the query point (highlighted in white)
        neighbor_indices: indices of Top-K neighbours (highlighted in gold)
        height:          iframe height in pixels
    """
    # PCA to 3D
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(embeddings).tolist()

    n_speakers = len(set(labels))
    colors = _speaker_colors(n_speakers)
    point_colors = [colors[int(l)] for l in labels]

    query_idx_js = query_idx if query_idx is not None else -1
    neighbor_set = set(neighbor_indices.tolist()) if neighbor_indices is not None else set()

    data = {
        "coords": coords,
        "colors": point_colors,
        "labels": labels.tolist(),
        "speakerIds": speaker_ids,
        "queryIdx": query_idx_js,
        "neighborIdxs": list(neighbor_set),
    }

    data_json = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ margin: 0; background: #0a0a1a; overflow: hidden; }}
  canvas {{ display: block; }}
  #info {{
    position: absolute; top: 10px; left: 10px;
    color: #aaa; font: 12px monospace;
    pointer-events: none;
  }}
</style>
</head>
<body>
<div id="info">Drag to rotate · Scroll to zoom · Speaker embeddings (PCA 3D)</div>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = {data_json};

// ── Scene setup ──────────────────────────────────────────────────────────────
const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(60, window.innerWidth / {height}, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, {height});
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

camera.position.set(0, 0, 8);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// ── Point cloud ───────────────────────────────────────────────────────────────
const N = DATA.coords.length;
const positions = new Float32Array(N * 3);
const colorsArr = new Float32Array(N * 3);
const sizes     = new Float32Array(N);

const neighborSet = new Set(DATA.neighborIdxs);

for (let i = 0; i < N; i++) {{
  positions[i*3]   = DATA.coords[i][0];
  positions[i*3+1] = DATA.coords[i][1];
  positions[i*3+2] = DATA.coords[i][2];

  let hex = DATA.colors[i];
  const c = new THREE.Color(hex);

  if (i === DATA.queryIdx) {{
    colorsArr[i*3] = 1; colorsArr[i*3+1] = 1; colorsArr[i*3+2] = 1;
    sizes[i] = 18;
  }} else if (neighborSet.has(i)) {{
    colorsArr[i*3] = 1; colorsArr[i*3+1] = 0.84; colorsArr[i*3+2] = 0;
    sizes[i] = 12;
  }} else {{
    colorsArr[i*3] = c.r * 0.6; colorsArr[i*3+1] = c.g * 0.6; colorsArr[i*3+2] = c.b * 0.6;
    sizes[i] = 4;
  }}
}}

const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geo.setAttribute('color',    new THREE.BufferAttribute(colorsArr, 3));
geo.setAttribute('size',     new THREE.BufferAttribute(sizes, 1));

const mat = new THREE.ShaderMaterial({{
  uniforms: {{ time: {{ value: 0 }} }},
  vertexShader: `
    attribute float size;
    varying vec3 vColor;
    uniform float time;
    void main() {{
      vColor = color;
      vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * (200.0 / -mvPos.z);
      gl_Position = projectionMatrix * mvPos;
    }}
  `,
  fragmentShader: `
    varying vec3 vColor;
    void main() {{
      float d = length(gl_PointCoord - 0.5) * 2.0;
      if (d > 1.0) discard;
      float alpha = 1.0 - smoothstep(0.4, 1.0, d);
      gl_FragColor = vec4(vColor, alpha);
    }}
  `,
  vertexColors: true,
  transparent: true,
  depthWrite: false,
}});

const points = new THREE.Points(geo, mat);
scene.add(points);

// ── Lines from query to neighbours ───────────────────────────────────────────
if (DATA.queryIdx >= 0 && DATA.neighborIdxs.length > 0) {{
  const qx = DATA.coords[DATA.queryIdx];
  DATA.neighborIdxs.forEach(ni => {{
    const nx = DATA.coords[ni];
    const lineGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...qx),
      new THREE.Vector3(...nx),
    ]);
    const lineMat = new THREE.LineBasicMaterial({{ color: 0xffd700, transparent: true, opacity: 0.5 }});
    scene.add(new THREE.Line(lineGeo, lineMat));
  }});
}}

// ── Animate ───────────────────────────────────────────────────────────────────
let t = 0;
function animate() {{
  requestAnimationFrame(animate);
  t += 0.016;
  mat.uniforms.time.value = t;

  // Pulse query point size
  if (DATA.queryIdx >= 0) {{
    sizes[DATA.queryIdx] = 14 + 6 * Math.sin(t * 3);
    geo.attributes.size.needsUpdate = true;
  }}

  controls.update();
  renderer.render(scene, camera);
}}
animate();

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / {height};
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, {height});
}});
</script>
</body>
</html>"""
    return html
