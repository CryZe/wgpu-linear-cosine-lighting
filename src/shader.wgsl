struct Uniforms {
    mvp: mat4x4<f32>,
    model: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>, // For sky rendering
    camera_pos: vec3<f32>,
    roughness: f32,
    light_p0: vec3<f32>,
    ground_roughness: f32,
    light_p1: vec3<f32>,
    light_intensity: f32,
    light_p2: vec3<f32>,
    apply_gamma: f32,
    light_p3: vec3<f32>,
    _p3: f32,
    // L2 Spherical Harmonics coefficients (9 RGB values)
    sh0: vec3<f32>, _sh0: f32,
    sh1: vec3<f32>, _sh1: f32,
    sh2: vec3<f32>, _sh2: f32,
    sh3: vec3<f32>, _sh3: f32,
    sh4: vec3<f32>, _sh4: f32,
    sh5: vec3<f32>, _sh5: f32,
    sh6: vec3<f32>, _sh6: f32,
    sh7: vec3<f32>, _sh7: f32,
    sh8: vec3<f32>, _sh8: f32,
    // Emissive cube
    cube_center: vec3<f32>,
    cube_half_size: f32,
    cube_rotation: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var ltc_1: texture_2d<f32>;

@group(0) @binding(2)
var ltc_2: texture_2d<f32>;

@group(0) @binding(3)
var ltc_sampler: sampler;

@group(0) @binding(4)
var ground_texture: texture_2d<f32>;

@group(0) @binding(5)
var ground_sampler: sampler;

const PI: f32 = 3.14159265359;

// Evaluate L2 Spherical Harmonics irradiance for a given normal
fn evaluateSH(n: vec3<f32>) -> vec3<f32> {
    // SH basis functions
    let c1 = 0.429043;
    let c2 = 0.511664;
    let c3 = 0.743125;
    let c4 = 0.886227;
    let c5 = 0.247708;

    // Evaluate irradiance from SH coefficients
    // Using the standard irradiance convolution formula
    return c4 * uniforms.sh0
         + 2.0 * c2 * (uniforms.sh1 * n.y + uniforms.sh2 * n.z + uniforms.sh3 * n.x)
         + 2.0 * c1 * (uniforms.sh4 * n.x * n.y + uniforms.sh5 * n.y * n.z
                     + uniforms.sh7 * n.x * n.z)
         + c3 * uniforms.sh6 * (n.z * n.z - 1.0 / 3.0)
         + c1 * uniforms.sh8 * (n.x * n.x - n.y * n.y);
}

const LUT_SIZE: f32 = 64.0;
const LUT_SCALE: f32 = (64.0 - 1.0) / 64.0;
const LUT_BIAS: f32 = 0.5 / 64.0;

// Evaluate LTC for a single cube face (with rotation applied)
fn evaluateCubeFace(N: vec3<f32>, V: vec3<f32>, P: vec3<f32>, Minv: mat3x3<f32>, identity: mat3x3<f32>,
                    center: vec3<f32>, half_size: f32, axis: i32, sign: f32, rotation: f32) -> vec2<f32> {
    // Build quad points for this face (CCW when viewed from outside the cube)
    var face_pts: array<vec3<f32>, 4>;
    let offset = sign * half_size;
    let hs = half_size;

    if axis == 0 { // X axis faces
        let x = center.x + offset;
        if sign > 0.0 { // +X face - CCW from +X direction
            face_pts[0] = vec3<f32>(x, center.y + hs, center.z + hs);
            face_pts[1] = vec3<f32>(x, center.y + hs, center.z - hs);
            face_pts[2] = vec3<f32>(x, center.y - hs, center.z - hs);
            face_pts[3] = vec3<f32>(x, center.y - hs, center.z + hs);
        } else { // -X face - CCW from -X direction
            face_pts[0] = vec3<f32>(x, center.y + hs, center.z - hs);
            face_pts[1] = vec3<f32>(x, center.y + hs, center.z + hs);
            face_pts[2] = vec3<f32>(x, center.y - hs, center.z + hs);
            face_pts[3] = vec3<f32>(x, center.y - hs, center.z - hs);
        }
    } else if axis == 1 { // Y axis faces
        let y = center.y + offset;
        if sign > 0.0 { // +Y face (top) - facing upward
            face_pts[0] = vec3<f32>(center.x - hs, y, center.z - hs);
            face_pts[1] = vec3<f32>(center.x + hs, y, center.z - hs);
            face_pts[2] = vec3<f32>(center.x + hs, y, center.z + hs);
            face_pts[3] = vec3<f32>(center.x - hs, y, center.z + hs);
        } else { // -Y face (bottom) - facing downward
            face_pts[0] = vec3<f32>(center.x - hs, y, center.z + hs);
            face_pts[1] = vec3<f32>(center.x + hs, y, center.z + hs);
            face_pts[2] = vec3<f32>(center.x + hs, y, center.z - hs);
            face_pts[3] = vec3<f32>(center.x - hs, y, center.z - hs);
        }
    } else { // Z axis faces
        let z = center.z + offset;
        if sign > 0.0 { // +Z face - CCW from +Z direction
            face_pts[0] = vec3<f32>(center.x - hs, center.y + hs, z);
            face_pts[1] = vec3<f32>(center.x + hs, center.y + hs, z);
            face_pts[2] = vec3<f32>(center.x + hs, center.y - hs, z);
            face_pts[3] = vec3<f32>(center.x - hs, center.y - hs, z);
        } else { // -Z face - CCW from -Z direction
            face_pts[0] = vec3<f32>(center.x + hs, center.y + hs, z);
            face_pts[1] = vec3<f32>(center.x - hs, center.y + hs, z);
            face_pts[2] = vec3<f32>(center.x - hs, center.y - hs, z);
            face_pts[3] = vec3<f32>(center.x + hs, center.y - hs, z);
        }
    }

    // Apply rotation around Y axis (around cube center)
    for (var i = 0; i < 4; i++) {
        let local = face_pts[i] - center;
        let rotated = rotateY(local, rotation);
        face_pts[i] = rotated + center;
    }

    let spec = LTC_Evaluate(N, V, P, Minv, face_pts, false);
    let diff = LTC_Evaluate(N, V, P, identity, face_pts, false);
    return vec2<f32>(spec, diff);
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) object_id: u32, // 0 = cube, 1 = ground, 2 = light
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) @interpolate(flat) object_id: u32,
};

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

// Rotate a point around Y axis
fn rotateY(p: vec3<f32>, angle: f32) -> vec3<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec3<f32>(
        c * p.x + s * p.z,
        p.y,
        -s * p.x + c * p.z
    );
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    var pos = in.position;
    var norm = in.normal;

    // Rotate cube vertices (object_id == 0)
    if in.object_id == 0u {
        let cube_center = uniforms.cube_center;
        // Translate to origin, rotate, translate back
        let local_pos = pos - cube_center;
        let rotated_pos = rotateY(local_pos, uniforms.cube_rotation);
        pos = rotated_pos + cube_center;
        norm = rotateY(norm, uniforms.cube_rotation);
    }

    out.clip_position = uniforms.mvp * vec4<f32>(pos, 1.0);
    out.color = in.color;
    out.world_normal = (uniforms.model * vec4<f32>(norm, 0.0)).xyz;
    out.world_pos = (uniforms.model * vec4<f32>(pos, 1.0)).xyz;
    out.uv = in.uv;
    out.object_id = in.object_id;
    return out;
}

// Vector form factor (edge integration)
fn IntegrateEdgeVec(v1: vec3<f32>, v2: vec3<f32>) -> vec3<f32> {
    let x = dot(v1, v2);
    let y = abs(x);

    let a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    let b = 3.4175940 + (4.1616724 + y) * y;
    let v = a / b;

    var theta_sintheta: f32;
    if x > 0.0 {
        theta_sintheta = v;
    } else {
        theta_sintheta = 0.5 * (1.0 / sqrt(max(1.0 - x * x, 1e-7))) - v;
    }

    return cross(v1, v2) * theta_sintheta;
}

fn IntegrateEdge(v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    return IntegrateEdgeVec(v1, v2).z;
}

// Clip quad to horizon (z >= 0)
fn ClipQuadToHorizon(L: ptr<function, array<vec3<f32>, 5>>, n: ptr<function, i32>) {
    // Build config bitmask
    var config: i32 = 0;
    if (*L)[0].z > 0.0 { config += 1; }
    if (*L)[1].z > 0.0 { config += 2; }
    if (*L)[2].z > 0.0 { config += 4; }
    if (*L)[3].z > 0.0 { config += 8; }

    if config == 0 {
        *n = 0;
    } else if config == 1 { // V1 clip V2 V3 V4
        *n = 3;
        (*L)[1] = -(*L)[1].z * (*L)[0] + (*L)[0].z * (*L)[1];
        (*L)[2] = -(*L)[3].z * (*L)[0] + (*L)[0].z * (*L)[3];
    } else if config == 2 { // V2 clip V1 V3 V4
        *n = 3;
        (*L)[0] = -(*L)[0].z * (*L)[1] + (*L)[1].z * (*L)[0];
        (*L)[2] = -(*L)[2].z * (*L)[1] + (*L)[1].z * (*L)[2];
    } else if config == 3 { // V1 V2 clip V3 V4
        *n = 4;
        (*L)[2] = -(*L)[2].z * (*L)[1] + (*L)[1].z * (*L)[2];
        (*L)[3] = -(*L)[3].z * (*L)[0] + (*L)[0].z * (*L)[3];
    } else if config == 4 { // V3 clip V1 V2 V4
        *n = 3;
        (*L)[0] = -(*L)[3].z * (*L)[2] + (*L)[2].z * (*L)[3];
        (*L)[1] = -(*L)[1].z * (*L)[2] + (*L)[2].z * (*L)[1];
    } else if config == 5 { // V1 V3 clip V2 V4 - impossible
        *n = 0;
    } else if config == 6 { // V2 V3 clip V1 V4
        *n = 4;
        (*L)[0] = -(*L)[0].z * (*L)[1] + (*L)[1].z * (*L)[0];
        (*L)[3] = -(*L)[3].z * (*L)[2] + (*L)[2].z * (*L)[3];
    } else if config == 7 { // V1 V2 V3 clip V4
        *n = 5;
        (*L)[4] = -(*L)[3].z * (*L)[0] + (*L)[0].z * (*L)[3];
        (*L)[3] = -(*L)[3].z * (*L)[2] + (*L)[2].z * (*L)[3];
    } else if config == 8 { // V4 clip V1 V2 V3
        *n = 3;
        (*L)[0] = -(*L)[0].z * (*L)[3] + (*L)[3].z * (*L)[0];
        (*L)[1] = -(*L)[2].z * (*L)[3] + (*L)[3].z * (*L)[2];
        (*L)[2] = (*L)[3];
    } else if config == 9 { // V1 V4 clip V2 V3
        *n = 4;
        (*L)[1] = -(*L)[1].z * (*L)[0] + (*L)[0].z * (*L)[1];
        (*L)[2] = -(*L)[2].z * (*L)[3] + (*L)[3].z * (*L)[2];
    } else if config == 10 { // V2 V4 clip V1 V3 - impossible
        *n = 0;
    } else if config == 11 { // V1 V2 V4 clip V3
        *n = 5;
        (*L)[4] = (*L)[3];
        (*L)[3] = -(*L)[2].z * (*L)[3] + (*L)[3].z * (*L)[2];
        (*L)[2] = -(*L)[2].z * (*L)[1] + (*L)[1].z * (*L)[2];
    } else if config == 12 { // V3 V4 clip V1 V2
        *n = 4;
        (*L)[1] = -(*L)[1].z * (*L)[2] + (*L)[2].z * (*L)[1];
        (*L)[0] = -(*L)[0].z * (*L)[3] + (*L)[3].z * (*L)[0];
    } else if config == 13 { // V1 V3 V4 clip V2
        *n = 5;
        (*L)[4] = (*L)[3];
        (*L)[3] = (*L)[2];
        (*L)[2] = -(*L)[1].z * (*L)[2] + (*L)[2].z * (*L)[1];
        (*L)[1] = -(*L)[1].z * (*L)[0] + (*L)[0].z * (*L)[1];
    } else if config == 14 { // V2 V3 V4 clip V1
        *n = 5;
        (*L)[4] = -(*L)[0].z * (*L)[3] + (*L)[3].z * (*L)[0];
        (*L)[0] = -(*L)[0].z * (*L)[1] + (*L)[1].z * (*L)[0];
    } else if config == 15 { // V1 V2 V3 V4
        *n = 4;
    }

    if *n == 3 {
        (*L)[3] = (*L)[0];
    }
    if *n == 4 {
        (*L)[4] = (*L)[0];
    }
}

// LTC Evaluate for quad area light
fn LTC_Evaluate(N: vec3<f32>, V: vec3<f32>, P: vec3<f32>, Minv: mat3x3<f32>, points: array<vec3<f32>, 4>, twoSided: bool) -> f32 {
    // Construct orthonormal basis around N
    var T1 = normalize(V - N * dot(V, N));
    let T2 = cross(N, T1);

    // Rotate area light in (T1, T2, N) basis
    let M = Minv * transpose(mat3x3<f32>(T1, T2, N));

    // Polygon (allocate 5 vertices for clipping)
    var L: array<vec3<f32>, 5>;
    L[0] = M * (points[0] - P);
    L[1] = M * (points[1] - P);
    L[2] = M * (points[2] - P);
    L[3] = M * (points[3] - P);
    L[4] = vec3<f32>(0.0);

    // Clip to horizon
    var n: i32 = 4;
    ClipQuadToHorizon(&L, &n);

    if n == 0 {
        return 0.0;
    }

    // Project onto sphere
    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    L[4] = normalize(L[4]);

    // Integrate
    var sum: f32 = 0.0;
    sum += IntegrateEdge(L[0], L[1]);
    sum += IntegrateEdge(L[1], L[2]);
    sum += IntegrateEdge(L[2], L[3]);
    if n >= 4 {
        sum += IntegrateEdge(L[3], L[4]);
    }
    if n == 5 {
        sum += IntegrateEdge(L[4], L[0]);
    }

    if twoSided {
        sum = abs(sum);
    } else {
        sum = max(0.0, sum);
    }

    return sum;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    // Sample ALL textures FIRST before any non-uniform control flow
    // This is required for WebGPU uniform control flow rules
    let tex_color = textureSample(ground_texture, ground_sampler, in.uv).rgb;

    // Pre-compute LUT coordinates (we'll use a default roughness for the sample)
    let N = normalize(in.world_normal);
    let V = normalize(uniforms.camera_pos - in.world_pos);
    let ndotv = clamp(dot(N, V), 0.0, 1.0);

    // Store normal for SSAO (encode to 0-1 range)
    let encoded_normal = N * 0.5 + 0.5;

    // Determine roughness for LUT lookup
    var roughness = uniforms.roughness;
    let is_ground = in.object_id == 1u;
    let is_wall = in.object_id == 3u;
    if is_ground || is_wall {
        let luminance = dot(tex_color, vec3<f32>(0.299, 0.587, 0.114));
        roughness = min(uniforms.ground_roughness + luminance * 0.4, 1.0);
    } else if in.object_id == 4u {
        // object - shiny metallic surface
        roughness = uniforms.ground_roughness;
    }

    let uv = vec2<f32>(roughness, sqrt(1.0 - ndotv)) * LUT_SCALE + LUT_BIAS;
    let t1 = textureSample(ltc_1, ltc_sampler, uv);
    let t2 = textureSample(ltc_2, ltc_sampler, uv);

    // NOW we can do non-uniform control flow (early returns)

    // Check if this is an emissive surface (light source - object_id == 2)
    if in.object_id == 2u {
        // This is the area light - render as emissive
        let emissive = in.color;
        // Tone map the emissive
        let final_emissive = emissive / (emissive + vec3<f32>(1.0));
        var out: FragmentOutput;
        out.color = vec4<f32>(final_emissive, 1.0);
        out.normal = vec4<f32>(encoded_normal, 1.0);
        return out;
    }

    let P = in.world_pos;

    // Determine surface color
    var surface_color = in.color;
    if is_ground || is_wall {
        surface_color = tex_color;
    }

    // Construct inverse M matrix for specular
    // Layout from reference: columns are [t1.x,0,t1.y], [0,1,0], [t1.z,0,t1.w]
    // In WGSL mat3x3 constructor takes columns
    let Minv = mat3x3<f32>(
        vec3<f32>(t1.x, 0.0, t1.y),
        vec3<f32>(0.0,  1.0, 0.0),
        vec3<f32>(t1.z, 0.0, t1.w)
    );

    // Area light points
    var points: array<vec3<f32>, 4>;
    points[0] = uniforms.light_p0;
    points[1] = uniforms.light_p1;
    points[2] = uniforms.light_p2;
    points[3] = uniforms.light_p3;

    // Evaluate LTC for specular
    let spec = LTC_Evaluate(N, V, P, Minv, points, true);

    // Fresnel/shadowing from LUT (t2.x = magnitude/norm, t2.y = fresnel)
    // Reference formula: spec *= scol * t2.x + (1 - scol) * t2.y
    // scol is the specular color/reflectance at normal incidence
    let is_object = in.object_id == 4u;
    var F0: vec3<f32>;
    if is_ground || is_wall {
        F0 = vec3<f32>(0.95); // High reflectance for mirror
    } else if is_object {
        F0 = vec3<f32>(0.7); // Shiny metallic object
    } else {
        F0 = vec3<f32>(0.04); // Standard dielectric
    }
    let spec_scaled = spec * (F0 * t2.x + (vec3<f32>(1.0) - F0) * t2.y);

    // Evaluate LTC for diffuse (identity matrix = cosine distribution)
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0)
    );
    let diff = LTC_Evaluate(N, V, P, identity, points, true);

    // Light color/intensity for main area light
    let light_color = vec3<f32>(1.0, 0.95, 0.9) * uniforms.light_intensity;

    // Evaluate 6 cube faces as emissive lights
    let cube_center = uniforms.cube_center;
    let cube_hs = uniforms.cube_half_size;
    let cube_rot = uniforms.cube_rotation;

    // Face colors matching the cube geometry (intensity scaled)
    let cube_intensity = 3.0;
    let color_px = vec3<f32>(1.0, 0.0, 1.0) * cube_intensity; // +X magenta
    let color_nx = vec3<f32>(0.0, 1.0, 1.0) * cube_intensity; // -X cyan
    let color_py = vec3<f32>(0.0, 0.0, 1.0) * cube_intensity; // +Y blue (top)
    let color_ny = vec3<f32>(1.0, 1.0, 0.0) * cube_intensity; // -Y yellow (bottom)
    let color_pz = vec3<f32>(1.0, 0.0, 0.0) * cube_intensity; // +Z red (front)
    let color_nz = vec3<f32>(0.0, 1.0, 0.0) * cube_intensity; // -Z green (back)

    // Evaluate each face
    let face_px = evaluateCubeFace(N, V, P, Minv, identity, cube_center, cube_hs, 0, 1.0, cube_rot);
    let face_nx = evaluateCubeFace(N, V, P, Minv, identity, cube_center, cube_hs, 0, -1.0, cube_rot);
    let face_py = evaluateCubeFace(N, V, P, Minv, identity, cube_center, cube_hs, 1, 1.0, cube_rot);
    let face_ny = evaluateCubeFace(N, V, P, Minv, identity, cube_center, cube_hs, 1, -1.0, cube_rot);
    let face_pz = evaluateCubeFace(N, V, P, Minv, identity, cube_center, cube_hs, 2, 1.0, cube_rot);
    let face_nz = evaluateCubeFace(N, V, P, Minv, identity, cube_center, cube_hs, 2, -1.0, cube_rot);

    // Accumulate colored contributions from each face
    let cube_diffuse_contrib = surface_color * (
        face_px.y * color_px +
        face_nx.y * color_nx +
        face_py.y * color_py +
        face_ny.y * color_ny +
        face_pz.y * color_pz +
        face_nz.y * color_nz
    ) / PI;

    let cube_spec_raw =
        face_px.x * color_px +
        face_nx.x * color_nx +
        face_py.x * color_py +
        face_ny.x * color_ny +
        face_pz.x * color_pz +
        face_nz.x * color_nz;
    let cube_specular_contrib = cube_spec_raw * (F0 * t2.x + (vec3<f32>(1.0) - F0) * t2.y);

    // Combine diffuse and specular from main light
    var diffuse_amount = 1.0;
    let diffuse_contrib = surface_color * diff / PI * diffuse_amount;

    // Specular contribution
    let specular_contrib = spec_scaled;

    let col = light_color * (diffuse_contrib + specular_contrib)
            + cube_diffuse_contrib + cube_specular_contrib;

    // Evaluate L2 Spherical Harmonics for ambient lighting
    // SH basis functions for normal direction
    let sh_ambient = evaluateSH(N);
    let ambient = surface_color * sh_ambient * 0.4;

    // Tone mapping (simple Reinhard)
    let final_col = col / (col + vec3<f32>(1.0)) + ambient;

    // Note: Keep linear for SSAO, gamma applied in composite

    var out: FragmentOutput;
    out.color = vec4<f32>(final_col, 1.0);
    out.normal = vec4<f32>(encoded_normal, 1.0);
    return out;
}

// ============== Sky Shader ==============

struct SkyVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) view_dir: vec3<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) vertex_index: u32) -> SkyVertexOutput {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    let pos = positions[vertex_index];

    var out: SkyVertexOutput;
    out.clip_position = vec4<f32>(pos, 0.9999, 1.0); // Near far plane

    // Reconstruct world-space view direction using inverse view-projection
    let near_point = uniforms.inv_view_proj * vec4<f32>(pos, -1.0, 1.0);
    let far_point = uniforms.inv_view_proj * vec4<f32>(pos, 1.0, 1.0);
    let near_world = near_point.xyz / near_point.w;
    let far_world = far_point.xyz / far_point.w;
    out.view_dir = normalize(far_world - near_world);

    return out;
}

@fragment
fn fs_sky(in: SkyVertexOutput) -> FragmentOutput {
    // Normalize view direction
    let dir = normalize(in.view_dir);

    // Evaluate SH for sky color
    var sky_color = evaluateSH(dir);

    // Make sky brighter and more saturated
    sky_color = max(sky_color, vec3<f32>(0.0));

    // Note: Keep linear for SSAO, gamma applied in composite

    var out: FragmentOutput;
    out.color = vec4<f32>(sky_color, 1.0);
    // Sky normal pointing up (will be ignored due to depth = 1.0)
    out.normal = vec4<f32>(0.5, 1.0, 0.5, 1.0);
    return out;
}

// ============== SSAO Shader ==============

@group(1) @binding(0)
var scene_color: texture_2d<f32>;

@group(1) @binding(1)
var scene_depth: texture_depth_2d;

@group(1) @binding(2)
var scene_normal: texture_2d<f32>;

@group(1) @binding(3)
var ssao_sampler: sampler;

struct SSAOVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_ssao(@builtin(vertex_index) vertex_index: u32) -> SSAOVertexOutput {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    let pos = positions[vertex_index];

    var out: SSAOVertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y; // Flip Y for texture sampling

    return out;
}

// SSAO kernel - hemisphere samples (normalized, z always positive)
fn get_ssao_sample(i: u32) -> vec3<f32> {
    // Pre-computed hemisphere samples with varying lengths (closer samples weighted more)
    var samples = array<vec3<f32>, 16>(
        vec3<f32>(0.536, 0.215, 0.088),
        vec3<f32>(-0.385, 0.482, 0.158),
        vec3<f32>(0.183, -0.418, 0.238),
        vec3<f32>(-0.242, 0.138, 0.425),
        vec3<f32>(0.465, 0.342, 0.258),
        vec3<f32>(-0.341, -0.463, 0.344),
        vec3<f32>(0.278, -0.128, 0.572),
        vec3<f32>(-0.189, 0.354, 0.486),
        vec3<f32>(0.142, 0.395, 0.298),
        vec3<f32>(-0.358, -0.187, 0.512),
        vec3<f32>(0.324, 0.268, 0.638),
        vec3<f32>(-0.198, 0.442, 0.356),
        vec3<f32>(0.456, -0.324, 0.378),
        vec3<f32>(-0.478, -0.198, 0.398),
        vec3<f32>(0.298, 0.456, 0.524),
        vec3<f32>(-0.324, 0.378, 0.656)
    );
    return normalize(samples[i % 16u]);
}

// Noise function for randomizing samples
fn noise2d(uv: vec2<f32>) -> vec2<f32> {
    let n = fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let m = fract(sin(dot(uv, vec2<f32>(93.989, 67.345))) * 24634.6345);
    return vec2<f32>(n, m) * 2.0 - 1.0;
}

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

@fragment
fn fs_ssao(in: SSAOVertexOutput) -> @location(0) vec4<f32> {
    let tex_size = vec2<f32>(textureDimensions(scene_depth));
    let texel = 1.0 / tex_size;
    let coords = vec2<i32>(in.uv * tex_size);

    // Sample depth
    let depth = textureLoad(scene_depth, coords, 0);

    // Skip sky (depth = 1.0)
    if depth > 0.999 {
        return vec4<f32>(1.0);
    }

    // Camera parameters
    let near = 0.1;
    let far = 100.0;
    let linear_depth = linearize_depth(depth, near, far);

    // Screen-space SSAO with fixed pixel radius
    let base_radius = 150.0;
    let radius_pixels = base_radius * (1.0 / linear_depth);
    let radius_pixels_clamped = clamp(radius_pixels, 4.0, 100.0);

    let num_samples = 32u;
    // Higher bias to avoid artifacts on flat surfaces at grazing angles
    let bias = 0.1 + linear_depth * 0.02;

    // Random rotation per pixel to reduce banding
    let noise = noise2d(in.uv * tex_size * 0.25);
    let base_angle = noise.x * 6.283185;

    var occlusion = 0.0;
    var valid_samples = 0.0;

    for (var i = 0u; i < num_samples; i = i + 1u) {
        // Distribute samples in a spiral pattern
        let fi = f32(i);
        let angle = base_angle + fi * 2.399963; // Golden angle
        let radius_scale = (fi + 1.0) / f32(num_samples);
        let sample_radius = radius_pixels_clamped * sqrt(radius_scale);

        let offset = vec2<f32>(cos(angle), sin(angle)) * sample_radius * texel;
        let sample_uv = in.uv + offset;

        // Bounds check
        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        // Get depth at sample position
        let sample_coords = vec2<i32>(sample_uv * tex_size);
        let sample_depth_raw = textureLoad(scene_depth, sample_coords, 0);

        // Skip sky samples
        if sample_depth_raw > 0.999 {
            continue;
        }

        let sample_linear_depth = linearize_depth(sample_depth_raw, near, far);

        // Depth difference (positive = sample is closer to camera)
        let depth_diff = linear_depth - sample_linear_depth;

        // Occlusion if sample is closer and within reasonable range
        let max_range = 0.5 * linear_depth; // Scale range with depth
        if depth_diff > bias && depth_diff < max_range {
            let falloff = 1.0 - (depth_diff / max_range);
            occlusion = occlusion + falloff;
        }
        valid_samples = valid_samples + 1.0;
    }

    if valid_samples > 0.0 {
        occlusion = occlusion / valid_samples;
    }

    // Convert to AO (1 = no occlusion, 0 = full occlusion)
    let ao = 1.0 - clamp(occlusion * 1.5, 0.0, 0.7);

    return vec4<f32>(ao, ao, ao, 1.0);
}

// ============== Composite Shader ==============

@group(1) @binding(4)
var ssao_texture: texture_2d<f32>;

@vertex
fn vs_composite(@builtin(vertex_index) vertex_index: u32) -> SSAOVertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );

    let pos = positions[vertex_index];

    var out: SSAOVertexOutput;
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y;

    return out;
}

@fragment
fn fs_composite(in: SSAOVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(scene_color, ssao_sampler, in.uv).rgb;

    // Simple box blur on AO to reduce noise (3x3)
    let tex_size = vec2<f32>(textureDimensions(ssao_texture));
    let texel = 1.0 / tex_size;
    var ao = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            ao = ao + textureSample(ssao_texture, ssao_sampler, in.uv + offset).r;
        }
    }
    ao = ao / 9.0;

    // Apply AO to color (in linear space)
    let ao_strength = 1.0;
    let final_ao = mix(1.0, ao, ao_strength);
    var result = color * final_ao;

    // Apply gamma correction if surface is not sRGB
    if uniforms.apply_gamma > 0.5 {
        result = pow(result, vec3<f32>(1.0 / 2.2));
    }

    return vec4<f32>(result, 1.0);
}
