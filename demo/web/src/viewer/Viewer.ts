import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { CameraSet, ColorMode } from "../lib/types";

/** Sequential single-hue ramp, light -> dark (dataviz reference palette, blue 100..700). */
export const RAMP = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
  "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];
const ACCENT = "#3987e5";
const CAM_MUTED = "#6f6e69";
const CAM_ACTIVE = "#f0efec";
const TRAJ_ARRIVAL = "#898781";
const BACKGROUND = "#0d0d0d";

/** Part of the ramp used on the dark viewer background: its darkest steps would vanish into it. */
const RAMP_SPAN = 0.78;

function rampTexture(): THREE.DataTexture {
  const n = 256;
  const stops = RAMP.map((h) => new THREE.Color(h));
  const data = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (stops.length - 1);
    const a = Math.floor(t);
    const b = Math.min(stops.length - 1, a + 1);
    const f = t - a;
    // sRGB values, interpolated in sRGB like the server's colour maps; written straight to the framebuffer
    const ca = new THREE.Color(RAMP[a]);
    const cb = new THREE.Color(RAMP[b]);
    const pa = ca.getRGB(new THREE.Color(), THREE.SRGBColorSpace);
    const pb = cb.getRGB(new THREE.Color(), THREE.SRGBColorSpace);
    data[i * 4] = Math.round(255 * (pa.r + (pb.r - pa.r) * f));
    data[i * 4 + 1] = Math.round(255 * (pa.g + (pb.g - pa.g) * f));
    data[i * 4 + 2] = Math.round(255 * (pa.b + (pb.b - pa.b) * f));
    data[i * 4 + 3] = 255;
  }
  const tex = new THREE.DataTexture(data, n, 1, THREE.RGBAFormat);
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

const VERT = /* glsl */ `
attribute vec3 rgb;
attribute float conf;
attribute float view;
attribute float flags;
uniform float uSize;
uniform float uScale;
uniform float uKeep;
uniform int uMode;
uniform float uHideSky;
uniform float uViewMax;
uniform float uViewCount;
uniform float uSelected;
uniform float uIsolate;
uniform float uRampSpan;
uniform sampler2D uRamp;
varying vec3 vColor;

void main() {
  bool sky = uHideSky > 0.5 && mod(flags, 2.0) > 0.5;
  bool lowConf = conf < 1.0 - uKeep - 0.002;
  bool other = uIsolate > 0.5 && uSelected >= 0.0 && abs(view - uSelected) > 0.5;
  if (sky || lowConf || other || view > uViewMax + 0.5) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    gl_PointSize = 0.0;
    return;
  }
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = clamp(uSize * uScale / max(-mv.z, 1e-6), 1.0, 48.0);
  if (uMode == 0) {
    vColor = rgb;
  } else if (uMode == 1) {
    vColor = texture2D(uRamp, vec2((1.0 - conf) * uRampSpan, 0.5)).rgb;
  } else {
    float t = view / max(uViewCount - 1.0, 1.0);
    vColor = texture2D(uRamp, vec2((1.0 - t) * uRampSpan, 0.5)).rgb;
  }
  if (uSelected >= 0.0 && uIsolate < 0.5 && abs(view - uSelected) > 0.5) vColor *= 0.38;
}
`;

const FRAG = /* glsl */ `
varying vec3 vColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}
`;

interface Cam {
  R: number[]; // 3x3 row-major camera-to-world rotation
  C: [number, number, number];
  K: [number, number, number, number];
}

export interface ViewerSettings {
  colorMode: ColorMode;
  keep: number; // fraction of the most confident points per view
  size: number; // point size multiplier
  camSize: number; // frustum size multiplier
  showCameras: boolean;
  showImages: boolean;
  showTrajectory: boolean;
  trajectory: "arrival" | "refined";
  hideSky: boolean;
  isolate: boolean;
  selected: number | null;
  viewMax: number; // views / frames above this index are hidden (timeline)
  display: "points" | "splats"; // the point cloud or the Gaussian splats of a splat run
}

export const DEFAULT_SETTINGS: ViewerSettings = {
  colorMode: "rgb",
  keep: 0.5,
  size: 1,
  camSize: 1,
  showCameras: true,
  showImages: false,
  showTrajectory: true,
  trajectory: "arrival",
  hideSky: true,
  isolate: false,
  selected: null,
  viewMax: Infinity,
  display: "points",
};

type Spark = typeof import("@sparkjsdev/spark");
export type SplatState = { state: "loading" | "ready" | "error"; url: string; detail?: string };

function parseCams(set: CameraSet): Cam[] {
  return set.c2w.map((m, i) => ({
    R: [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]],
    C: [m[3], m[7], m[11]],
    K: set.K[i] as [number, number, number, number],
  }));
}

export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
  readonly controls: OrbitControls;
  private root = new THREE.Group(); // world (OpenCV, frame of view 0) -> viewer axes (y up)
  private pointsGroup = new THREE.Group();
  private material: THREE.ShaderMaterial;
  private chunks: THREE.Points[] = [];
  private frusta: THREE.LineSegments;
  private trajectory: THREE.Line;
  private exitTrajectory: THREE.Line;
  private planes = new THREE.Group();
  private planeMeshes = new Map<number, THREE.Mesh>();
  private textures = new Map<string, THREE.Texture>();
  private cams: (Cam | undefined)[] = [];
  private exitCams: (Cam | undefined)[] = [];
  private updated = new Set<number>();
  private hw: [number, number] = [336, 504];
  private aligned = false;
  private scaleFixed = false;
  private autoFrame = true; // re-frame on new data until the user moves the view
  private sceneScale = 1;
  private center = new THREE.Vector3();
  private needsRender = true;
  private raf = 0;
  private resizeObs: ResizeObserver;
  private down: { x: number; y: number } | null = null;
  private disposed = false;
  settings: ViewerSettings = { ...DEFAULT_SETTINGS };
  onSelect?: (view: number | null) => void;
  imageUrl?: (view: number, size: number) => string | null;
  onSplatState?: (s: SplatState) => void;
  hasContent = false;
  private spark: InstanceType<Spark["SparkRenderer"]> | null = null;
  private splat: InstanceType<Spark["SplatMesh"]> | null = null;
  private splatUrl: string | null = null;

  constructor(private container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(BACKGROUND);
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.display = "block";
    this.renderer.domElement.style.touchAction = "none";

    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uSize: { value: 0.004 },
        uScale: { value: 600 },
        uKeep: { value: this.settings.keep },
        uMode: { value: 0 },
        uHideSky: { value: 1 },
        uViewMax: { value: 1e9 },
        uViewCount: { value: 1 },
        uSelected: { value: -1 },
        uIsolate: { value: 0 },
        uRampSpan: { value: RAMP_SPAN },
        uRamp: { value: rampTexture() },
      },
    });

    this.frusta = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }),
    );
    this.frusta.frustumCulled = false;
    this.trajectory = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: TRAJ_ARRIVAL }));
    this.trajectory.frustumCulled = false;
    this.exitTrajectory = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ACCENT }));
    this.exitTrajectory.frustumCulled = false;

    this.root.matrixAutoUpdate = false;
    this.root.add(this.pointsGroup, this.frusta, this.trajectory, this.exitTrajectory, this.planes);
    this.scene.add(this.root);

    this.camera.position.set(0, 1, 3);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.addEventListener("change", () => this.requestRender());
    this.controls.addEventListener("start", () => (this.autoFrame = false));

    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", (e) => (this.down = { x: e.clientX, y: e.clientY }));
    el.addEventListener("pointerup", (e) => {
      if (!this.down) return;
      const moved = Math.hypot(e.clientX - this.down.x, e.clientY - this.down.y);
      this.down = null;
      if (moved < 5) this.pick(e);
    });

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.resize();
    this.loop();
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.controls.dispose();
    this.clearPoints();
    this.splat?.dispose();
    this.spark?.dispose?.();
    this.textures.forEach((t) => t.dispose());
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ------------------------------------------------------------------ rendering
  requestRender() {
    this.needsRender = true;
  }

  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    if (!this.needsRender) return;
    this.needsRender = false;
    this.renderer.render(this.scene, this.camera);
  };

  private resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.updateScaleUniform();
    this.requestRender();
  }

  private updateScaleUniform() {
    const hPx = this.renderer.getDrawingBufferSize(new THREE.Vector2()).y;
    this.material.uniforms.uScale.value = hPx / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
    this.material.uniforms.uSize.value = 0.0035 * this.sceneScale * this.settings.size;
  }

  // ------------------------------------------------------------------ data
  // ------------------------------------------------------------------ splats
  /** Shows the Gaussian splats at url (a .splat file in the world frame of the solve), replacing the previous ones;
   * null removes them. Spark (the splat renderer) is loaded on first use. */
  async setSplats(url: string | null) {
    if (url === this.splatUrl) return;
    this.splatUrl = url;
    if (this.splat) {
      this.root.remove(this.splat);
      this.splat.dispose();
      this.splat = null;
      this.applyDisplay();
    }
    if (!url) return;
    this.onSplatState?.({ state: "loading", url });
    try {
      const spark: Spark = await import("@sparkjsdev/spark");
      if (this.disposed || this.splatUrl !== url) return;
      if (!this.spark) {
        this.spark = new spark.SparkRenderer({ renderer: this.renderer, onDirty: () => this.requestRender() });
        this.scene.add(this.spark);
      }
      const mesh = new spark.SplatMesh({ url, fileType: spark.SplatFileType.SPLAT });
      await mesh.initialized;
      if (this.disposed || this.splatUrl !== url) {
        mesh.dispose();
        return;
      }
      this.splat = mesh;
      this.root.add(mesh);
      this.applyDisplay();
      this.onSplatState?.({ state: "ready", url });
    } catch (e) {
      if (this.splatUrl === url) {
        this.splatUrl = null;
        this.onSplatState?.({ state: "error", url, detail: (e as Error).message });
      }
    }
  }

  hasSplats() {
    return this.splat !== null;
  }

  /** Points or splats: the cameras, trajectory and image planes stay either way. */
  private applyDisplay() {
    const splats = this.settings.display === "splats" && this.splat !== null;
    this.pointsGroup.visible = !splats;
    if (this.splat) this.splat.visible = splats;
    this.requestRender();
  }

  /** Clears everything for a new reconstruction. */
  reset(hw?: [number, number]) {
    void this.setSplats(null);
    this.clearPoints();
    this.cams = [];
    this.exitCams = [];
    this.updated.clear();
    this.aligned = false;
    this.scaleFixed = false;
    this.autoFrame = true;
    this.sceneScale = 1;
    this.hasContent = false;
    if (hw) this.hw = hw;
    this.planeMeshes.forEach((m) => {
      m.geometry.dispose();
      (m.material as THREE.Material).dispose();
    });
    this.planeMeshes.clear();
    this.planes.clear();
    this.rebuildCameraGeometry();
    this.requestRender();
  }

  setImageSize(hw: [number, number]) {
    this.hw = hw;
  }

  private clearPoints() {
    for (const p of this.chunks) {
      p.geometry.dispose();
      this.pointsGroup.remove(p);
    }
    this.chunks = [];
  }

  private makePoints(buffer: ArrayBuffer, n: number): THREE.Points {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buffer, 0, n * 3), 3));
    g.setAttribute("view", new THREE.BufferAttribute(new Uint16Array(buffer, 12 * n, n), 1));
    g.setAttribute("rgb", new THREE.BufferAttribute(new Uint8Array(buffer, 14 * n, n * 3), 3, true));
    g.setAttribute("conf", new THREE.BufferAttribute(new Uint8Array(buffer, 17 * n, n), 1, true));
    g.setAttribute("flags", new THREE.BufferAttribute(new Uint8Array(buffer, 18 * n, n), 1));
    const p = new THREE.Points(g, this.material);
    p.frustumCulled = false;
    return p;
  }

  /** Replaces the point cloud (offline / incremental solves). While the user has not moved the view, every update
   * re-frames it: the geometry of early calls can differ a lot in scale from the converged one. */
  setCloud(buffer: ArrayBuffer, n: number) {
    this.clearPoints();
    if (n === 0) return;
    const p = this.makePoints(buffer, n);
    this.chunks.push(p);
    this.pointsGroup.add(p);
    this.hasContent = true;
    if (this.autoFrame || !this.scaleFixed) {
      this.estimateScale(new Float32Array(buffer, 0, n * 3));
      if (this.autoFrame) this.fit();
    }
    this.requestRender();
  }

  /** Adds a chunk of points (one streamed frame). The view is framed on the first chunk (the anchors). */
  appendCloud(buffer: ArrayBuffer, n: number) {
    if (n === 0) return;
    const p = this.makePoints(buffer, n);
    this.chunks.push(p);
    this.pointsGroup.add(p);
    if (!this.scaleFixed) this.estimateScale(new Float32Array(buffer, 0, n * 3));
    const first = !this.hasContent;
    this.hasContent = true;
    if (first && this.autoFrame) this.fit();
    this.requestRender();
  }

  /** Re-estimates the scale from all points and frames the scene, unless the user has moved the view. */
  refit(force = false) {
    if (!force && !this.autoFrame) return;
    const parts = this.chunks.map((c) => c.geometry.getAttribute("position").array as Float32Array);
    const total = parts.reduce((a, b) => a + b.length, 0);
    if (!total) return;
    const step = Math.max(1, Math.floor(total / 3 / 60000)) * 3;
    const sample: number[] = [];
    let k = 0;
    for (const a of parts) {
      for (; k < a.length; k += step) sample.push(a[k], a[k + 1], a[k + 2]);
      k -= a.length;
    }
    this.estimateScale(new Float32Array(sample));
    this.fit();
  }

  pointCount(): number {
    return this.chunks.reduce((a, p) => a + (p.geometry.getAttribute("position")?.count ?? 0), 0);
  }

  /** Scene scale (robust radius) from a sample of points, fixed for the rest of the run. */
  private estimateScale(xyz: Float32Array) {
    const n = xyz.length / 3;
    if (n < 100) return;
    const step = Math.max(1, Math.floor(n / 20000));
    const xs: number[] = [], ys: number[] = [], zs: number[] = [];
    for (let i = 0; i < n; i += step) {
      xs.push(xyz[3 * i]);
      ys.push(xyz[3 * i + 1]);
      zs.push(xyz[3 * i + 2]);
    }
    const med = (a: number[]) => [...a].sort((p, q) => p - q)[Math.floor(a.length / 2)];
    const c = new THREE.Vector3(med(xs), med(ys), med(zs));
    const d = xs.map((x, i) => Math.hypot(x - c.x, ys[i] - c.y, zs[i] - c.z)).sort((p, q) => p - q);
    this.sceneScale = Math.max(1e-6, d[Math.floor(d.length * 0.8)]);
    this.center.copy(c);
    this.scaleFixed = true;
    this.updateScaleUniform();
    this.rebuildCameraGeometry();
  }

  /** All cameras of a solve, replacing the previous ones (index = view in the solve). */
  setCameras(set: CameraSet, updated?: number[] | null) {
    this.cams = parseCams(set);
    this.updated = new Set(updated ?? []);
    this.align();
    this.rebuildCameraGeometry();
    this.requestRender();
  }

  /** Cameras of streamed frames (index = frame). */
  addCameras(frames: number[], set: CameraSet) {
    const cams = parseCams(set);
    frames.forEach((f, i) => (this.cams[f] = cams[i]));
    this.updated = new Set(frames);
    this.align();
    this.rebuildCameraGeometry();
    this.requestRender();
  }

  setExitTrajectory(c2w: number[][] | null) {
    this.exitCams = c2w ? parseCams({ c2w, K: c2w.map(() => [1, 1, 0, 0]) }) : [];
    this.rebuildCameraGeometry();
    this.requestRender();
  }

  cameraCount() {
    return this.cams.filter(Boolean).length;
  }

  /** Viewer axes from the cameras: y along their average up, the capture looking along -z. Fixed for the run. */
  private align() {
    if (this.aligned) return;
    const cams = this.cams.filter(Boolean) as Cam[];
    if (!cams.length) return;
    const up = new THREE.Vector3();
    const fwd = new THREE.Vector3();
    for (const c of cams) {
      up.add(new THREE.Vector3(-c.R[1], -c.R[4], -c.R[7]));
      fwd.add(new THREE.Vector3(c.R[2], c.R[5], c.R[8]));
    }
    up.normalize();
    fwd.addScaledVector(up, -fwd.dot(up));
    if (fwd.length() < 1e-3) {
      const c = cams[0];
      fwd.set(-c.R[1], -c.R[4], -c.R[7]).cross(new THREE.Vector3(1, 0, 0));
    }
    fwd.normalize();
    const z = fwd.clone().negate();
    const x = new THREE.Vector3().crossVectors(up, z).normalize();
    const m = new THREE.Matrix4().makeBasis(x, up, z).transpose(); // rows x, y, z: world -> viewer
    this.root.matrix.copy(m);
    this.root.matrixWorldNeedsUpdate = true;
    this.aligned = true;
    if (!this.scaleFixed) {
      // camera spread as a provisional scale until points arrive
      const mean = new THREE.Vector3();
      cams.forEach((c) => mean.add(new THREE.Vector3(...c.C)));
      mean.divideScalar(cams.length);
      const spread = cams.reduce((a, c) => a + mean.distanceTo(new THREE.Vector3(...c.C)), 0) / cams.length;
      this.sceneScale = Math.max(spread * 2, 0.5);
      this.updateScaleUniform();
    }
  }

  private camColor(i: number, count: number): THREE.Color {
    const s = this.settings;
    if (s.selected === i) return new THREE.Color(ACCENT);
    if (this.updated.has(i) && this.updated.size < count) return new THREE.Color(CAM_ACTIVE);
    if (s.colorMode === "order") {
      const t = (1 - i / Math.max(count - 1, 1)) * RAMP_SPAN;
      return new THREE.Color(RAMP[Math.round(t * (RAMP.length - 1))]);
    }
    return new THREE.Color(CAM_MUTED);
  }

  /** Frustum corners (world): apex and the 4 image corners at depth d. */
  private corners(c: Cam, d: number): THREE.Vector3[] {
    const [H, W] = this.hw;
    const [fx, fy, cx, cy] = c.K;
    const pts = [
      [0, 0],
      [W, 0],
      [W, H],
      [0, H],
    ].map(([u, v]) => {
      const x = ((u - cx) / fx) * d;
      const y = ((v - cy) / fy) * d;
      return new THREE.Vector3(
        c.R[0] * x + c.R[1] * y + c.R[2] * d + c.C[0],
        c.R[3] * x + c.R[4] * y + c.R[5] * d + c.C[1],
        c.R[6] * x + c.R[7] * y + c.R[8] * d + c.C[2],
      );
    });
    return [new THREE.Vector3(...c.C), ...pts];
  }

  /** Frustum depth: smaller when there are many cameras (a stream), so the trajectory stays readable. */
  private frustumDepth() {
    const n = this.cams.length;
    return 0.07 * this.sceneScale * this.settings.camSize * (n > 120 ? Math.sqrt(120 / n) : 1);
  }

  private rebuildCameraGeometry() {
    const s = this.settings;
    const count = this.cams.length;
    const d = this.frustumDepth();
    const pos: number[] = [];
    const col: number[] = [];
    const viewMax = s.viewMax;
    this.cams.forEach((c, i) => {
      if (!c || i > viewMax) return;
      const [o, a, b, cc, dd] = this.corners(c, d);
      const segs = [o, a, o, b, o, cc, o, dd, a, b, b, cc, cc, dd, dd, a];
      // a small triangle above the top edge marks "up" in the image
      const mid = a.clone().add(b).multiplyScalar(0.5);
      const upv = a.clone().sub(dd).multiplyScalar(0.25);
      const tip = mid.clone().add(upv);
      const l = a.clone().lerp(b, 0.35);
      const r = a.clone().lerp(b, 0.65);
      segs.push(l, tip, tip, r);
      const color = this.camColor(i, count);
      for (const p of segs) {
        pos.push(p.x, p.y, p.z);
        col.push(color.r, color.g, color.b);
      }
    });
    const g = this.frusta.geometry;
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.computeBoundingSphere();
    this.frusta.visible = s.showCameras;

    const line = (cams: (Cam | undefined)[], obj: THREE.Line) => {
      const p: number[] = [];
      cams.forEach((c, i) => c && i <= viewMax && p.push(...c.C));
      obj.geometry.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
      obj.geometry.computeBoundingSphere();
    };
    line(this.cams, this.trajectory);
    line(this.exitCams, this.exitTrajectory);
    const hasExit = this.exitCams.length > 0;
    this.trajectory.visible = s.showTrajectory && (!hasExit || s.trajectory === "arrival");
    this.exitTrajectory.visible = s.showTrajectory && hasExit && s.trajectory === "refined";
    this.updatePlanes(d);
  }

  private texture(url: string): THREE.Texture {
    let t = this.textures.get(url);
    if (!t) {
      t = new THREE.TextureLoader().load(url, () => this.requestRender());
      t.colorSpace = THREE.SRGBColorSpace;
      this.textures.set(url, t);
      if (this.textures.size > 600) {
        const [k, old] = this.textures.entries().next().value as [string, THREE.Texture];
        old.dispose();
        this.textures.delete(k);
      }
    }
    return t;
  }

  /** Image planes: the selected view always, every view with showImages (up to 400). */
  private updatePlanes(d: number) {
    const s = this.settings;
    const want = new Set<number>();
    if (s.showCameras) {
      if (s.showImages && this.cams.length <= 400) this.cams.forEach((c, i) => c && i <= s.viewMax && want.add(i));
      if (s.selected !== null && this.cams[s.selected]) want.add(s.selected);
    }
    for (const [i, m] of this.planeMeshes) {
      if (!want.has(i)) {
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
        this.planes.remove(m);
        this.planeMeshes.delete(i);
      }
    }
    for (const i of want) {
      const c = this.cams[i]!;
      const [, a, b, cc, dd] = this.corners(c, d);
      let m = this.planeMeshes.get(i);
      if (!m) {
        const url = this.imageUrl?.(i, s.selected === i ? 640 : 160);
        if (!url) continue;
        const g = new THREE.BufferGeometry();
        g.setAttribute("uv", new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2));
        g.setIndex([0, 1, 2, 0, 2, 3]);
        const mat = new THREE.MeshBasicMaterial({
          map: this.texture(url),
          side: THREE.DoubleSide,
          transparent: true,
          opacity: s.selected === i ? 0.95 : 0.8,
          depthWrite: false,
        });
        m = new THREE.Mesh(g, mat);
        m.frustumCulled = false;
        this.planeMeshes.set(i, m);
        this.planes.add(m);
      }
      m.geometry.setAttribute("position", new THREE.Float32BufferAttribute([a, b, cc, dd].flatMap((p) => [p.x, p.y, p.z]), 3));
    }
  }

  // ------------------------------------------------------------------ settings
  apply(next: Partial<ViewerSettings>) {
    const prevSel = this.settings.selected;
    this.settings = { ...this.settings, ...next };
    const s = this.settings;
    const u = this.material.uniforms;
    u.uMode.value = s.colorMode === "rgb" ? 0 : s.colorMode === "confidence" ? 1 : 2;
    u.uKeep.value = s.keep;
    u.uHideSky.value = s.hideSky ? 1 : 0;
    u.uIsolate.value = s.isolate ? 1 : 0;
    u.uSelected.value = s.selected ?? -1;
    u.uViewMax.value = Number.isFinite(s.viewMax) ? s.viewMax : 1e9;
    u.uViewCount.value = Math.max(1, this.cams.length);
    this.updateScaleUniform();
    if (prevSel !== s.selected && prevSel !== null) {
      // the selected plane uses a sharper image: rebuild it
      const m = this.planeMeshes.get(prevSel);
      if (m) {
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
        this.planes.remove(m);
        this.planeMeshes.delete(prevSel);
      }
    }
    if (s.selected !== null && prevSel !== s.selected) {
      const m = this.planeMeshes.get(s.selected);
      if (m) {
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
        this.planes.remove(m);
        this.planeMeshes.delete(s.selected);
      }
    }
    this.rebuildCameraGeometry();
    this.applyDisplay();
    this.requestRender();
  }

  setViewCount(n: number) {
    this.material.uniforms.uViewCount.value = Math.max(1, n);
    this.requestRender();
  }

  // ------------------------------------------------------------------ navigation
  private toViewer(v: THREE.Vector3): THREE.Vector3 {
    this.root.updateMatrixWorld(true);
    return v.clone().applyMatrix4(this.root.matrixWorld);
  }

  private dirToViewer(v: THREE.Vector3): THREE.Vector3 {
    return v.clone().applyMatrix3(new THREE.Matrix3().setFromMatrix4(this.root.matrix)).normalize();
  }

  /** Frames the whole scene from behind and above the first camera. */
  fit() {
    this.camera.fov = 50;
    this.camera.updateProjectionMatrix();
    this.updateScaleUniform();
    const r = this.sceneScale;
    const target = this.toViewer(this.center);
    const c0 = this.cams.find(Boolean);
    let dir = new THREE.Vector3(0, 0.4, 1);
    if (c0) {
      const p = this.toViewer(new THREE.Vector3(...c0.C));
      dir = p.clone().sub(target);
      if (dir.length() < 0.25 * r) dir = this.dirToViewer(new THREE.Vector3(c0.R[2], c0.R[5], c0.R[8])).negate();
      dir.normalize();
      dir.y += 0.45;
      dir.normalize();
    }
    this.camera.near = r * 0.002;
    this.camera.far = r * 400;
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(target).addScaledVector(dir, r * 2.6);
    this.controls.target.copy(target);
    this.controls.update();
    this.requestRender();
  }

  /** Looks through camera `view`, with its field of view. */
  goToCamera(view: number) {
    const c = this.cams[view];
    if (!c) return;
    this.autoFrame = false;
    const p = this.toViewer(new THREE.Vector3(...c.C));
    const f = this.dirToViewer(new THREE.Vector3(c.R[2], c.R[5], c.R[8]));
    this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(this.hw[0] / 2 / c.K[1]));
    this.camera.updateProjectionMatrix();
    this.updateScaleUniform();
    this.camera.position.copy(p);
    this.controls.target.copy(p).addScaledVector(f, this.sceneScale * 0.5);
    this.controls.update();
    this.requestRender();
  }

  private pick(e: PointerEvent) {
    if (!this.settings.showCameras) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    let best = -1;
    let bestD = 18;
    this.root.updateMatrixWorld(true);
    const d = this.frustumDepth();
    this.cams.forEach((c, i) => {
      if (!c || i > this.settings.viewMax) return;
      // the centre of the frustum's image plane is easier to hit than the apex
      const [, a, , cc] = this.corners(c, d);
      for (const w of [new THREE.Vector3(...c.C), a.clone().add(cc).multiplyScalar(0.5)]) {
        const q = w.applyMatrix4(this.root.matrixWorld).project(this.camera);
        if (q.z > 1) continue;
        const sx = ((q.x + 1) / 2) * rect.width;
        const sy = ((1 - q.y) / 2) * rect.height;
        const dist = Math.hypot(sx - x, sy - y);
        if (dist < bestD) {
          bestD = dist;
          best = i;
        }
      }
    });
    this.onSelect?.(best >= 0 ? best : null);
  }

  screenshot(): string {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL("image/png");
  }
}
