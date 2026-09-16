// Ghost replay: fly your own flights back through the reconstruction they
// built, two at once, on a shared clock.
//
// Occlusion is the whole problem here. This viewer sorts and alpha-blends
// splats with no depth buffer at all -- gl_Position.z was hardcoded to 0 --
// which is why the flight path had to be baked into the .splat as Gaussians
// to be occluded by terrain. Moving geometry cannot do that: the sort is a
// throttled pass over millions of splats, not something to redo per frame.
//
// The obvious fix -- draw the drone first, then splats over it -- does not
// work here. The splat pass blends ONE_MINUS_DST_ALPHA/ONE: front-to-back
// "under" compositing, where dst alpha is accumulated coverage. Anything drawn
// beforehand with alpha 1 makes every subsequent splat multiply by (1-1) and
// the entire scene disappears.
//
// So main.js now writes real NDC depth from the splat pass, and the ghosts are
// drawn twice, on either side of it:
//
//   clear colour+depth
//   -> ghosts DEPTH ONLY, colour masked off      (stakes out the depth buffer)
//   -> splats, blended, depth TEST on/WRITE off  (splats behind a ghost are
//                                                 rejected and never accumulate)
//   -> ghosts in colour, same "under" blend      (composited beneath exactly
//                                                 the splats that are in front)
//
// The second pass gets correct occlusion for free from the blend operator:
// a ghost is dimmed by precisely the coverage accumulated in front of it, and
// the prepass is what guarantees nothing behind it ever contributed. Cost is
// one extra pass over a few hundred triangles.
(function (global) {
    "use strict";

    // ---------------------------------------------------------------- maths
    const mul4 = (a, b) => {
        const o = new Array(16);
        for (let c = 0; c < 4; c++)
            for (let r = 0; r < 4; r++)
                o[c * 4 + r] =
                    a[r] * b[c * 4] +
                    a[4 + r] * b[c * 4 + 1] +
                    a[8 + r] * b[c * 4 + 2] +
                    a[12 + r] * b[c * 4 + 3];
        return o;
    };
    const norm3 = (v) => {
        const l = Math.hypot(v[0], v[1], v[2]) || 1;
        return [v[0] / l, v[1] / l, v[2] / l];
    };
    const cross3 = (a, b) => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const sub3 = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

    // quaternion (x,y,z,w) -> 3x3, returned row-major as [r0c0,r0c1,...]
    function quatToM3(q) {
        const [x, y, z, w] = q;
        return [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ];
    }
    // model matrix (column-major) from row-major R and world position C
    function modelFrom(R, C) {
        return [
            R[0], R[3], R[6], 0,
            R[1], R[4], R[7], 0,
            R[2], R[5], R[8], 0,
            C[0], C[1], C[2], 1,
        ];
    }
    // world->camera for a camera with camera-to-world R (row-major) at C
    function viewFrom(R, C) {
        return [
            R[0], R[1], R[2], 0,
            R[3], R[4], R[5], 0,
            R[6], R[7], R[8], 0,
            -(R[0] * C[0] + R[3] * C[1] + R[6] * C[2]),
            -(R[1] * C[0] + R[4] * C[1] + R[7] * C[2]),
            -(R[2] * C[0] + R[5] * C[1] + R[8] * C[2]),
            1,
        ];
    }
    // world->camera looking from eye at target, camera convention X right,
    // Y down, Z forward (matches COLMAP, which is what the splats use)
    function lookAt(eye, target, worldUp) {
        const f = norm3(sub3(target, eye));
        let r = cross3(f, worldUp);
        if (Math.hypot(r[0], r[1], r[2]) < 1e-6) r = cross3(f, [1, 0, 0]);
        r = norm3(r);
        const d = cross3(f, r); // camera "down"
        // camera-to-world columns are (r, d, f) -> row-major R
        const R = [r[0], d[0], f[0], r[1], d[1], f[1], r[2], d[2], f[2]];
        return { view: viewFrom(R, eye), R, C: eye };
    }
    function slerp(a, b, t) {
        let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        let bb = b;
        if (d < 0) {
            bb = [-b[0], -b[1], -b[2], -b[3]];
            d = -d;
        }
        if (d > 0.9995) {
            const o = [
                a[0] + (bb[0] - a[0]) * t, a[1] + (bb[1] - a[1]) * t,
                a[2] + (bb[2] - a[2]) * t, a[3] + (bb[3] - a[3]) * t,
            ];
            const l = Math.hypot(o[0], o[1], o[2], o[3]) || 1;
            return [o[0] / l, o[1] / l, o[2] / l, o[3] / l];
        }
        const th0 = Math.acos(d), th = th0 * t;
        const s0 = Math.sin(th0), s1 = Math.sin(th0 - th) / s0, s2 = Math.sin(th) / s0;
        return [
            a[0] * s1 + bb[0] * s2, a[1] * s1 + bb[1] * s2,
            a[2] * s1 + bb[2] * s2, a[3] * s1 + bb[3] * s2,
        ];
    }

    // ------------------------------------------------------------ drone mesh
    // Built in the camera/body frame the poses are already in: X right,
    // Y DOWN, Z forward. Up is -Y. trajectory.py has already rolled the
    // camera's mount tilt out, so +Z here is where the drone is going.
    function buildDrone() {
        const pos = [], nrm = [], cen = [], kind = [], shade = [];

        function tri(a, b, c, ct, kd, sh) {
            const n = norm3(cross3(sub3(b, a), sub3(c, a)));
            for (const v of [a, b, c]) {
                pos.push(v[0], v[1], v[2]);
                nrm.push(n[0], n[1], n[2]);
                cen.push(ct[0], ct[1], ct[2]);
                kind.push(kd);
                shade.push(sh);
            }
        }
        function box(cx, cy, cz, sx, sy, sz, sh) {
            const x0 = cx - sx, x1 = cx + sx, y0 = cy - sy, y1 = cy + sy,
                z0 = cz - sz, z1 = cz + sz;
            const P = [
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ];
            const F = [
                [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                [3, 7, 6, 2], [0, 4, 7, 3], [1, 2, 6, 5],
            ];
            const c = [cx, cy, cz];
            for (const f of F) {
                tri(P[f[0]], P[f[1]], P[f[2]], c, 0, sh);
                tri(P[f[0]], P[f[2]], P[f[3]], c, 0, sh);
            }
        }
        // three-bladed prop, so the spin actually reads when you get close
        function prop(cx, cy, cz, r) {
            const c = [cx, cy, cz], hub = 0.1 * r;
            for (let b = 0; b < 3; b++) {
                const a0 = (b / 3) * Math.PI * 2;
                const w = 0.22;
                const p1 = [cx + Math.cos(a0) * hub, cy, cz + Math.sin(a0) * hub];
                const p2 = [cx + Math.cos(a0 - w) * r, cy - 0.06 * r, cz + Math.sin(a0 - w) * r];
                const p3 = [cx + Math.cos(a0 + w) * r, cy + 0.06 * r, cz + Math.sin(a0 + w) * r];
                tri(p1, p2, p3, c, 1, 0.85);
                tri(p1, p3, p2, c, 1, 0.85);
            }
        }

        box(0, 0, 0.05, 0.30, 0.13, 0.45, 1.0);          // body
        box(0, -0.16, 0.02, 0.16, 0.06, 0.26, 0.75);     // canopy
        // nose wedge -- makes heading unmistakable from any angle
        tri([-0.16, -0.06, 0.5], [0.16, -0.06, 0.5], [0, 0.02, 0.86], [0, 0, 0.6], 0, 1.5);
        tri([-0.16, 0.08, 0.5], [0, 0.02, 0.86], [0.16, 0.08, 0.5], [0, 0, 0.6], 0, 1.5);
        const A = 0.52;
        for (const [sx, sz] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
            const ex = sx * A, ez = sz * A;
            // arm: a thin box from the body out to the motor
            const mx = ex / 2, mz = ez / 2;
            box(mx, 0, mz, Math.abs(ex) / 2 + 0.05, 0.045, 0.06, 0.6);
            box(ex, -0.05, ez, 0.09, 0.09, 0.09, 0.5);   // motor
            prop(ex, -0.17, ez, 0.42);
        }
        return {
            pos: new Float32Array(pos), nrm: new Float32Array(nrm),
            cen: new Float32Array(cen), kind: new Float32Array(kind),
            shade: new Float32Array(shade), count: pos.length / 3,
        };
    }

    // --------------------------------------------------------------- shaders
    const DRONE_VS = `#version 300 es
precision highp float;
in vec3 a_pos; in vec3 a_nrm; in vec3 a_cen; in float a_kind; in float a_shade;
uniform mat4 u_mvp; uniform mat3 u_rot; uniform float u_spin; uniform float u_scale;
out vec3 vN; out float vShade;
void main(){
  vec3 p = a_pos; vec3 n = a_nrm;
  if (a_kind > 0.5) {                       // props spin about local Y
    vec3 d = p - a_cen; float c = cos(u_spin), s = sin(u_spin);
    p = a_cen + vec3(c*d.x + s*d.z, d.y, -s*d.x + c*d.z);
    n = vec3(c*n.x + s*n.z, n.y, -s*n.x + c*n.z);
  }
  vN = u_rot * n; vShade = a_shade;
  gl_Position = u_mvp * vec4(p * u_scale, 1.0);
}`;

    const DRONE_FS = `#version 300 es
precision highp float;
in vec3 vN; in float vShade;
uniform vec3 u_color;
out vec4 fragColor;
void main(){
  vec3 L = normalize(vec3(0.4, -1.0, 0.25));
  float d = max(dot(normalize(vN), -L), 0.0);
  vec3 c = u_color * vShade * (0.40 + 0.60 * d);
  c += vec3(0.10) * pow(d, 8.0);
  fragColor = vec4(min(c, vec3(1.6)), 1.0);
}`;

    // Trail: the ENTIRE trajectory is uploaded once as a screen-space-widened
    // ribbon, so animating the growing tail is just moving the drawArrays
    // range -- no per-frame buffer churn.
    // Widened in VIEW space, not screen space. The screen-space version had to
    // divide by clip w to hold a constant pixel width, and in chase view the
    // tail runs right past the camera where w <= 0: the offsets blew up into
    // viewport-sized triangles which, in the depth prepass, staked out the near
    // plane and rejected every splat behind them. The whole scene went black.
    // Offsetting before projection has no division to explode, is clipped
    // correctly by the near plane, and tapers with distance like real geometry.
    const TRAIL_VS = `#version 300 es
precision highp float;
in vec3 a_pos; in vec3 a_dir; in float a_side; in float a_i;
uniform mat4 u_view; uniform mat4 u_projection;
uniform float u_head; uniform float u_fade; uniform float u_radius;
uniform float u_taper;
out float vT; out float vD;
void main(){
  vec4 v = u_view * vec4(a_pos, 1.0);
  vD = length(v.xyz);
  vec3 tv = mat3(u_view) * a_dir;
  // perpendicular to the trail and to the eye ray, so the ribbon always
  // presents its face to the camera
  vec3 off = cross(tv, v.xyz);
  float l = length(off);
  off = l > 1e-9 ? off / l : vec3(1.0, 0.0, 0.0);
  // Taper the width to nothing as the ribbon approaches the eye. A constant
  // world width is ~7% of the viewport where it crosses znear (0.2), so the
  // tail flared into a wedge across the view in onboard mode, where the head
  // of the trail sits exactly at the camera. Beyond u_taper the width is
  // constant, so the chase view is untouched.
  v.xyz += off * a_side * u_radius * clamp(vD / u_taper, 0.0, 1.0);
  vT = clamp((a_i - (u_head - u_fade)) / u_fade, 0.0, 1.0);
  gl_Position = u_projection * v;
}`;

    const TRAIL_FS = `#version 300 es
precision highp float;
in float vT; in float vD;
uniform vec3 u_color; uniform float u_dim; uniform float u_depthPass;
uniform float u_near;
out vec4 fragColor;
void main(){
  float a = vT * vT;                    // comet tail, fades out behind the drone
  // Fade out as the ribbon approaches the eye. A constant world-space radius
  // subtends a huge angle from a few centimetres away, and onboard the head of
  // the trail sits exactly at the camera -- it flared into a wedge across the
  // whole view. Fading by view distance fixes onboard and close fly-bys alike.
  a *= smoothstep(u_near, u_near * 3.0, vD);
  // In the depth prepass only the solid part of the tail may stake out depth;
  // letting the faint end write depth would reject splats behind something the
  // viewer cannot even see.
  if (u_depthPass > 0.5 && a < 0.5) discard;
  fragColor = vec4(u_color * u_dim * a, a);   // premultiplied, to match splats
}`;

    function compile(gl, vs, fs) {
        const p = gl.createProgram();
        for (const [t, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
            const s = gl.createShader(t);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
                console.error("ghosts shader:", gl.getShaderInfoLog(s));
            gl.attachShader(p, s);
        }
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS))
            console.error("ghosts link:", gl.getProgramInfoLog(p));
        return p;
    }

    const PALETTE = [
        [0.16, 0.95, 1.00],   // cyan
        [1.00, 0.35, 0.85],   // magenta
        [1.00, 0.78, 0.20],   // amber
        [0.45, 1.00, 0.45],   // green
    ];
    const CSS = ["#29f2ff", "#ff59d9", "#ffc733", "#73ff73"];
    const MODES = ["free", "chase", "onboard"];

    // ---------------------------------------------------------------- class
    class Ghosts {
        constructor(gl, data, opts) {
            this.gl = gl;
            this.opts = opts;
            this.clips = data.clips;
            this.enabled = true;
            this.playing = true;
            this.showTrails = true;
            this.t = 0;
            this.rate = 1;
            this.mode = opts.mode || 0;
            this.active = 0;
            this.spin = 0;
            this.scale = opts.scale;
            this.videoSize = opts.videoSize;   // 0 off, 1 corner, 2 large
            this.videoRaw = false;             // U: raw fisheye vs undistorted
            this.duration = Math.max(...this.clips.map((c) => c.duration));
            this.smoothEye = null;

            this.droneProg = compile(gl, DRONE_VS, DRONE_FS);
            this.trailProg = compile(gl, TRAIL_VS, TRAIL_FS);

            const mesh = buildDrone();
            this.mesh = mesh;
            this.dVao = gl.createVertexArray();
            gl.bindVertexArray(this.dVao);
            const bind = (prog, name, arr, size) => {
                const b = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, b);
                gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
                const loc = gl.getAttribLocation(prog, name);
                if (loc >= 0) {
                    gl.enableVertexAttribArray(loc);
                    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
                }
            };
            bind(this.droneProg, "a_pos", mesh.pos, 3);
            bind(this.droneProg, "a_nrm", mesh.nrm, 3);
            bind(this.droneProg, "a_cen", mesh.cen, 3);
            bind(this.droneProg, "a_kind", mesh.kind, 1);
            bind(this.droneProg, "a_shade", mesh.shade, 1);
            gl.bindVertexArray(null);

            // one ribbon per clip, uploaded once
            for (const c of this.clips) {
                const n = c.n;
                const P = new Float32Array(n * 6);
                const D = new Float32Array(n * 6);
                const S = new Float32Array(n * 2);
                const I = new Float32Array(n * 2);
                for (let i = 0; i < n; i++) {
                    const j = Math.min(i + 1, n - 1), k = Math.max(i - 1, 0);
                    const d = norm3([
                        c.pos[j * 3] - c.pos[k * 3],
                        c.pos[j * 3 + 1] - c.pos[k * 3 + 1],
                        c.pos[j * 3 + 2] - c.pos[k * 3 + 2],
                    ]);
                    for (let s = 0; s < 2; s++) {
                        P[(i * 2 + s) * 3] = c.pos[i * 3];
                        P[(i * 2 + s) * 3 + 1] = c.pos[i * 3 + 1];
                        P[(i * 2 + s) * 3 + 2] = c.pos[i * 3 + 2];
                        D[(i * 2 + s) * 3] = d[0];
                        D[(i * 2 + s) * 3 + 1] = d[1];
                        D[(i * 2 + s) * 3 + 2] = d[2];
                        S[i * 2 + s] = s ? 1 : -1;
                        I[i * 2 + s] = i;
                    }
                }
                c.vao = gl.createVertexArray();
                gl.bindVertexArray(c.vao);
                bind(this.trailProg, "a_pos", P, 3);
                bind(this.trailProg, "a_dir", D, 3);
                bind(this.trailProg, "a_side", S, 1);
                bind(this.trailProg, "a_i", I, 1);
                gl.bindVertexArray(null);
            }
            this.buildHud();
        }

        // sample clip c at time t -> {P, q, speed, alive}
        // t is absolute clip time (seconds from the start of the source video),
        // which is why it can drive the footage overlay directly. c.t0 is the
        // time of the first REGISTERED frame, which is 0 on every clip so far
        // but need not be: clamp in absolute time, then index from t0.
        sample(c, t) {
            const end = c.t0 + c.duration;
            const alive = t <= end;
            const u = Math.max(0, Math.min(c.n - 1, (Math.min(t, end) - c.t0) / c.dt));
            const i = Math.floor(u), f = u - i, j = Math.min(i + 1, c.n - 1);
            const P = [
                c.pos[i * 3] + (c.pos[j * 3] - c.pos[i * 3]) * f,
                c.pos[i * 3 + 1] + (c.pos[j * 3 + 1] - c.pos[i * 3 + 1]) * f,
                c.pos[i * 3 + 2] + (c.pos[j * 3 + 2] - c.pos[i * 3 + 2]) * f,
            ];
            const q = slerp(
                [c.quat[i * 4], c.quat[i * 4 + 1], c.quat[i * 4 + 2], c.quat[i * 4 + 3]],
                [c.quat[j * 4], c.quat[j * 4 + 1], c.quat[j * 4 + 2], c.quat[j * 4 + 3]],
                f,
            );
            return { P, q, speed: c.speed[i], idx: u, alive };
        }

        // Nearest approach of `other`'s whole path to point P, and how far
        // apart in TIME the two flights were at that place. This is the
        // comparison that means something when two flights are not the same
        // route: "you were here 4 s earlier than last run", or honestly,
        // "the other flight never came within X of here".
        gapTo(other, P, t) {
            let best = Infinity, bi = 0;
            const n = other.n, p = other.pos;
            for (let i = 0; i < n; i++) {
                const dx = p[i * 3] - P[0], dy = p[i * 3 + 1] - P[1], dz = p[i * 3 + 2] - P[2];
                const d = dx * dx + dy * dy + dz * dz;
                if (d < best) { best = d; bi = i; }
            }
            return { dist: Math.sqrt(best), dt: (other.t0 + bi * other.dt) - t };
        }

        update(dt) {
            if (!this.enabled) return;
            if (this.playing) {
                this.t += dt * this.rate;
                if (this.t > this.duration) this.t = 0;
                if (this.t < 0) this.t = this.duration;
            }
            this.spin += dt * 42;
            this.updateHud();
            this.syncVideo();
        }

        // Keep the original footage locked to the replay clock. The ghost clock
        // IS absolute clip time, so the mapping is the identity -- no offset to
        // track. In onboard mode this puts the source frame next to a render of
        // the same pose, which is the cheapest honest quality check there is.
        syncVideo() {
            const v = this.video;
            if (!v) return;
            const show = this.enabled && this.videoSize > 0;
            // Must be an explicit "block": the stylesheet rule for #ghostvid is
            // display:none (so it stays hidden before this ever runs), and an
            // inline "" only clears the inline value, falling straight back to
            // that none. The HUD gets away with "" because it has no such rule.
            this.vidBox.style.display = show ? "block" : "none";
            this.vidBox.classList.toggle("big", this.videoSize === 2);
            if (!show) {
                if (!v.paused) v.pause();
                return;
            }
            const c = this.clips[this.active];
            // The default proxy is undistorted into the SAME pinhole camera the
            // render uses, so it can be compared with the render directly.
            // "_raw" is the untouched fisheye, kept for the before/after.
            const key = c.name + (this.videoRaw ? "_raw" : "");
            if (v.dataset.clip !== key) {
                v.dataset.clip = key;
                this.videoErr = false;
                v.src = this.opts.videoBase + key + ".mp4";
                v.load();   // currentTime resets to 0; the drift correction
                            // below seeks it back on the next frame
            }
            if (this.videoErr) return;
            const want = Math.min(this.t, c.t0 + c.duration);
            const cap = document.getElementById("gh-vcap");
            if (cap) {
                cap.textContent =
                    c.name + (this.videoRaw ? " original (raw fisheye)"
                                            : " original (undistorted)") +
                    (this.mode === 2 ? "  -- same pose as the render" : "");
            }
            // Let it play natively at 1x-ish and only correct on drift; seeking
            // every frame stutters badly. Scrubbing or odd rates seek instead.
            const canPlay = this.playing && this.rate >= 0.25 && this.rate <= 4;
            if (canPlay) {
                if (v.playbackRate !== this.rate) v.playbackRate = this.rate;
                if (v.paused) v.play().catch(() => {});
                if (v.readyState >= 2 && Math.abs(v.currentTime - want) > 0.3)
                    v.currentTime = want;
            } else {
                if (!v.paused) v.pause();
                if (v.readyState >= 1 && Math.abs(v.currentTime - want) > 0.04)
                    v.currentTime = want;
            }
        }

        // view matrix override for chase / onboard, or null to leave the
        // user's own camera alone
        viewOverride() {
            if (!this.enabled || this.mode === 0) return null;
            const c = this.clips[this.active];
            if (!c) return null;
            const s = this.sample(c, this.t);
            const R = quatToM3(s.q);
            if (this.mode === 2) {
                // onboard: the drone's own camera, tilt put back so it matches
                // what the pilot actually saw
                const tl = (-(c.tilt_deg || 0) * Math.PI) / 180;
                const ct = Math.cos(tl), st = Math.sin(tl);
                const Rx = [1, 0, 0, 0, ct, -st, 0, st, ct];
                const M = new Array(9);
                for (let r = 0; r < 3; r++)
                    for (let cc = 0; cc < 3; cc++)
                        M[r * 3 + cc] =
                            R[r * 3] * Rx[cc] + R[r * 3 + 1] * Rx[3 + cc] + R[r * 3 + 2] * Rx[6 + cc];
                return viewFrom(M, s.P);
            }
            // chase: sit behind and above, smoothed so the camera does not
            // inherit the drone's own jitter
            const fwd = [R[2], R[5], R[8]];
            const d = this.opts.chaseDist, h = this.opts.chaseHeight;
            // World up is +Y here: ply2splat levels Brush's Vertical axis (which
            // points DOWN, along gravity) onto (0,-1,0). Confirmed against the
            // capture poses -- mean camera-up has Y component +0.78.
            const want = [
                s.P[0] - fwd[0] * d,
                s.P[1] - fwd[1] * d + h,
                s.P[2] - fwd[2] * d,
            ];
            if (!this.smoothEye) this.smoothEye = want;
            const k = this.opts.chaseLag;
            this.smoothEye = [
                this.smoothEye[0] + (want[0] - this.smoothEye[0]) * k,
                this.smoothEye[1] + (want[1] - this.smoothEye[1]) * k,
                this.smoothEye[2] + (want[2] - this.smoothEye[2]) * k,
            ];
            return lookAt(this.smoothEye, s.P, [0, 1, 0]).view;
        }

        // Pass 1: depth only. Colour is masked off entirely, so this cannot
        // disturb the front-to-back accumulation the splat pass depends on.
        renderDepth(projection, view) {
            if (!this.enabled) return;
            const gl = this.gl;
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LESS);
            gl.depthMask(true);
            gl.disable(gl.BLEND);
            gl.colorMask(false, false, false, false);
            this.draw(projection, view, 1);
            gl.colorMask(true, true, true, true);
        }

        // Pass 2: colour, composited under whatever the splats accumulated in
        // front. Same blend function as the splats; LEQUAL so the geometry
        // matches the depth it wrote itself in pass 1.
        renderColor(projection, view) {
            if (!this.enabled) return;
            const gl = this.gl;
            gl.enable(gl.DEPTH_TEST);
            gl.depthFunc(gl.LEQUAL);
            gl.depthMask(false);
            gl.enable(gl.BLEND);
            this.draw(projection, view, 0);
            gl.depthFunc(gl.LESS);
        }

        draw(projection, view, depthPass) {
            const gl = this.gl;
            const viewProj = mul4(projection, view);

            if (this.showTrails) {
                gl.useProgram(this.trailProg);
                const u = (n) => gl.getUniformLocation(this.trailProg, n);
                const spp = this.clips[0].n / this.clips[0].duration;
                const fadeSamples = this.opts.trailFade * spp;
                gl.uniformMatrix4fv(u("u_view"), false, new Float32Array(view));
                gl.uniformMatrix4fv(u("u_projection"), false, new Float32Array(projection));
                gl.uniform1f(u("u_radius"), this.opts.trailRadius);
                gl.uniform1f(u("u_near"), this.opts.trailNear);
                gl.uniform1f(u("u_taper"), this.opts.trailTaper);
                gl.uniform1f(u("u_fade"), fadeSamples);
                gl.uniform1f(u("u_depthPass"), depthPass);
                for (let ci = 0; ci < this.clips.length; ci++) {
                    const c = this.clips[ci];
                    const s = this.sample(c, this.t);
                    const head = Math.floor(s.idx);
                    // only the faded window is visible, so only draw that
                    const from = Math.max(0, Math.floor(head - fadeSamples));
                    if (head - from < 1) continue;
                    const col = PALETTE[ci % PALETTE.length];
                    gl.uniform3f(u("u_color"), col[0], col[1], col[2]);
                    gl.uniform1f(u("u_head"), s.idx);
                    gl.uniform1f(u("u_dim"), ci === this.active ? 1.0 : 0.62);
                    gl.bindVertexArray(c.vao);
                    gl.drawArrays(gl.TRIANGLE_STRIP, from * 2, (head - from + 1) * 2);
                }
            }

            gl.useProgram(this.droneProg);
            const u = (n) => gl.getUniformLocation(this.droneProg, n);
            gl.bindVertexArray(this.dVao);
            for (let ci = 0; ci < this.clips.length; ci++) {
                // in onboard you are inside this one -- drawing it fills the screen
                if (this.mode === 2 && ci === this.active) continue;
                const c = this.clips[ci];
                const s = this.sample(c, this.t);
                const R = quatToM3(s.q);
                const model = modelFrom(R, s.P);
                const col = PALETTE[ci % PALETTE.length];
                const k = s.alive ? 1.0 : 0.3;
                gl.uniformMatrix4fv(u("u_mvp"), false, new Float32Array(mul4(viewProj, model)));
                gl.uniformMatrix3fv(u("u_rot"), false, new Float32Array([
                    R[0], R[3], R[6], R[1], R[4], R[7], R[2], R[5], R[8],
                ]));
                gl.uniform3f(u("u_color"), col[0] * k, col[1] * k, col[2] * k);
                gl.uniform1f(u("u_spin"), s.alive ? this.spin * (1 + ci * 0.13) : 0);
                gl.uniform1f(u("u_scale"), this.scale);
                gl.drawArrays(gl.TRIANGLES, 0, this.mesh.count);
            }
            gl.bindVertexArray(null);
        }

        // ------------------------------------------------------------- HUD
        buildHud() {
            const st = document.createElement("style");
            st.textContent = `
#ghosthud{position:fixed;left:12px;bottom:12px;z-index:120;font:12px/1.45
 ui-monospace,SFMono-Regular,Menlo,monospace;color:#e9edf2;background:rgba(12,15,20,.82);
 border:1px solid rgba(255,255,255,.13);border-radius:9px;padding:9px 11px;
 min-width:236px;backdrop-filter:blur(7px);user-select:none}
#ghosthud .r{display:flex;justify-content:space-between;gap:12px}
#ghosthud .hd{font-weight:600;letter-spacing:.08em;text-transform:uppercase;
 font-size:10px;opacity:.55;margin-bottom:5px}
#ghosthud .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
 margin-right:6px;vertical-align:1px}
#ghosthud .g{margin-top:5px;padding-top:5px;border-top:1px solid rgba(255,255,255,.09)}
#ghosthud .sub{opacity:.55;font-size:11px}
#ghosthud .act{font-weight:700}
#ghostbar{height:3px;background:rgba(255,255,255,.14);border-radius:2px;margin-top:7px}
#ghostbar div{height:100%;background:#29f2ff;border-radius:2px;width:0}
#ghostkeys{position:fixed;right:12px;bottom:12px;z-index:120;font:11px/1.5
 ui-monospace,Menlo,monospace;color:#c8d0d8;background:rgba(12,15,20,.72);
 border:1px solid rgba(255,255,255,.1);border-radius:8px;padding:7px 10px;
 opacity:.72;user-select:none}
#ghostkeys b{color:#fff;font-weight:600}
@media (max-width:760px){#ghostkeys{display:none}}
#ghostvid{position:fixed;top:12px;right:12px;z-index:121;display:none;
 border:1px solid rgba(255,255,255,.18);border-radius:9px;overflow:hidden;
 background:#000;box-shadow:0 6px 26px rgba(0,0,0,.55);width:26vw;max-width:420px}
#ghostvid.big{width:46vw;max-width:760px}
#ghostvid video{display:block;width:100%;height:auto}
#ghostvid .cap{position:absolute;left:0;right:0;bottom:0;padding:4px 8px;
 font:11px/1.4 ui-monospace,Menlo,monospace;color:#e9edf2;
 background:linear-gradient(transparent,rgba(0,0,0,.78));letter-spacing:.03em}
#ghostvid .warn{padding:10px;font:11px/1.5 ui-monospace,Menlo,monospace;color:#ffb4b4}`;
            document.head.appendChild(st);
            const d = document.createElement("div");
            d.id = "ghosthud";
            d.innerHTML =
                `<div class="hd">ghost replay</div><div class="r"><span id="gh-t"></span>` +
                `<span id="gh-mode" class="sub"></span></div>` +
                `<div id="ghostbar"><div id="gh-fill"></div></div><div id="gh-list"></div>`;
            document.body.appendChild(d);
            const vid = document.createElement("div");
            vid.id = "ghostvid";
            vid.innerHTML =
                '<video id="gh-video" muted playsinline preload="auto"></video>' +
                '<div class="cap" id="gh-vcap"></div>' +
                '<div class="warn" id="gh-vwarn" style="display:none"></div>';
            document.body.appendChild(vid);
            this.vidBox = vid;
            this.video = vid.querySelector("video");
            this.videoErr = false;
            // Show the warning ALONGSIDE the <video>, never by replacing the
            // box's innerHTML -- that detaches the element permanently, so a
            // proxy that is merely still being written (faststart puts the moov
            // atom last, so a partial file reports NO_SOURCE) could never
            // recover once ffmpeg finished. Retry instead.
            this.video.addEventListener("error", () => {
                this.videoErr = true;
                const w = document.getElementById("gh-vwarn");
                if (w) {
                    w.style.display = "";
                    w.innerHTML =
                        "no playable proxy for " + (this.video.dataset.clip || "?") +
                        " &mdash; still transcoding, or missing.<br>press <b>O</b> twice to retry<br><br>" +
                        "ffmpeg -i clips/&lt;clip&gt;.MP4 -vf scale=640:480 -c:v libx264 " +
                        "-crf 30 -an -movflags +faststart viewer/video/&lt;clip&gt;.mp4";
                }
            });
            this.video.addEventListener("loadedmetadata", () => {
                this.videoErr = false;
                const w = document.getElementById("gh-vwarn");
                if (w) w.style.display = "none";
            });

            const k = document.createElement("div");
            k.id = "ghostkeys";
            k.innerHTML =
                `<b>space</b> play/pause &nbsp; <b>C</b> camera &nbsp; <b>N</b> follow<br>` +
                `<b>[ ]</b> scrub 5s &nbsp; <b>, .</b> speed &nbsp; <b>R</b> restart<br>` +
                `<b>T</b> trails &nbsp; <b>O</b> footage &nbsp; <b>U</b> undistort` +
                ` &nbsp; <b>G</b> hide ghosts`;
            document.body.appendChild(k);
            this.hud = d;
            this.keys = k;
        }

        updateHud() {
            if (!this.hud) return;
            this.hud.style.display = this.enabled ? "" : "none";
            this.keys.style.display = this.enabled ? "" : "none";
            if (!this.enabled) return;
            const fmt = (t) =>
                `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
            document.getElementById("gh-t").textContent =
                `${fmt(this.t)} / ${fmt(this.duration)}` +
                (this.playing ? "" : "  paused") +
                (this.rate !== 1 ? `  ${this.rate}x` : "");
            document.getElementById("gh-mode").textContent = MODES[this.mode];
            document.getElementById("gh-fill").style.width =
                (100 * this.t) / this.duration + "%";
            const act = this.clips[this.active];
            const as = act && this.sample(act, this.t);
            let html = "";
            for (let i = 0; i < this.clips.length; i++) {
                const c = this.clips[i];
                const s = this.sample(c, this.t);
                let note;
                if (!s.alive) {
                    note = "landed";
                } else if (i === this.active) {
                    note = `${s.speed.toFixed(2)} u/s`;
                } else {
                    const g = this.gapTo(c, as.P, this.t);
                    note =
                        g.dist > this.opts.gapNear
                            ? `${g.dist.toFixed(2)} away`
                            : `${g.dt >= 0 ? "+" : ""}${g.dt.toFixed(1)}s here`;
                }
                html +=
                    `<div class="g r ${i === this.active ? "act" : ""}">` +
                    `<span><i class="dot" style="background:${CSS[i % CSS.length]}"></i>` +
                    `${c.name}</span><span class="sub">${note}</span></div>`;
            }
            document.getElementById("gh-list").innerHTML = html;
        }

        onKey(code) {
            if (code === "KeyG") {
                this.enabled = !this.enabled;
                if (!this.enabled) this.mode = 0;
                return true;
            }
            if (!this.enabled) return false;
            switch (code) {
                case "Space": this.playing = !this.playing; return true;
                case "KeyC":
                    this.mode = (this.mode + 1) % MODES.length;
                    this.smoothEye = null;
                    return true;
                case "KeyN":
                    this.active = (this.active + 1) % this.clips.length;
                    this.smoothEye = null;
                    return true;
                case "KeyR": this.t = 0; this.smoothEye = null; return true;
                case "KeyT": this.showTrails = !this.showTrails; return true;
                case "KeyO":
                    this.videoSize = (this.videoSize + 1) % 3;
                    if (this.videoSize && this.videoErr && this.video) {
                        this.videoErr = false;
                        this.video.dataset.clip = "";   // forces a fresh load
                    }
                    return true;
                case "KeyU":
                    this.videoRaw = !this.videoRaw;
                    this.videoErr = false;
                    if (this.video) this.video.dataset.clip = "";
                    return true;
                case "BracketLeft": this.t = Math.max(0, this.t - 5); return true;
                case "BracketRight":
                    this.t = Math.min(this.duration, this.t + 5); return true;
                case "Comma":
                    this.rate = Math.max(0.125, this.rate / 2); return true;
                case "Period":
                    this.rate = Math.min(8, this.rate * 2); return true;
            }
            return false;
        }
    }

    async function create(gl, url, opts) {
        let data;
        try {
            const r = await fetch(url);
            if (!r.ok) throw new Error(r.status + " " + r.statusText);
            data = await r.json();
        } catch (e) {
            console.warn("ghosts: no trajectory at", url, "-", e.message);
            return null;
        }
        if (!data.clips || !data.clips.length) return null;
        const o = Object.assign(
            {
                scale: 0.055,      // drone size in scene units
                trailRadius: 0.007, // scene units, half-width of the ribbon
                trailNear: 0.07,    // fade the ribbon out inside this of the eye
                trailTaper: 0.55,   // and narrow it to nothing over the same run-in
                trailFade: 12,     // seconds of visible tail
                chaseDist: 0.42,
                chaseHeight: 0.13,
                chaseLag: 0.14,
                gapNear: 0.35,     // scene units within which a time gap is meaningful
                mode: 0,
                videoSize: 1,      // 0 off, 1 corner, 2 large
                videoBase: "video/",
            },
            opts || {},
        );
        console.log(
            `ghosts: ${data.clips.length} flights, ` +
            data.clips.map((c) => `${c.name} ${c.duration.toFixed(0)}s`).join(", "),
        );
        return new Ghosts(gl, data, o);
    }

    global.Ghosts = { create };
})(window);
