use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::ltc_lut;

#[cfg(target_arch = "wasm32")]
thread_local! {
    static PENDING_STATE: std::cell::RefCell<Option<GfxState>> = std::cell::RefCell::new(None);
}

// Embedded OBJ file
const OBJECT_OBJ: &str = include_str!("../object.obj");

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    object_id: u32, // 0 = cube, 1 = ground, 2 = light
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x2, 4 => Uint32];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4], // For sky rendering
    camera_pos: [f32; 3],
    roughness: f32,
    light_p0: [f32; 3],
    ground_roughness: f32,
    light_p1: [f32; 3],
    light_intensity: f32,
    light_p2: [f32; 3],
    apply_gamma: f32,
    light_p3: [f32; 3],
    _p3: f32,
    // L2 Spherical Harmonics coefficients (9 RGB values)
    sh0: [f32; 3],
    _sh0: f32,
    sh1: [f32; 3],
    _sh1: f32,
    sh2: [f32; 3],
    _sh2: f32,
    sh3: [f32; 3],
    _sh3: f32,
    sh4: [f32; 3],
    _sh4: f32,
    sh5: [f32; 3],
    _sh5: f32,
    sh6: [f32; 3],
    _sh6: f32,
    sh7: [f32; 3],
    _sh7: f32,
    sh8: [f32; 3],
    _sh8: f32,
    // Emissive cube
    cube_center: [f32; 3],
    cube_half_size: f32,
    cube_rotation: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

struct Camera {
    position: Vec3,
    yaw: f32,   // Horizontal rotation (radians)
    pitch: f32, // Vertical rotation (radians)
}

impl Camera {
    fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        Self {
            position,
            yaw,
            pitch,
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Vec3::Y)
    }
}

#[derive(Default)]
struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    mouse_delta: (f64, f64),
    // Arrow keys for light movement
    light_up: bool,
    light_down: bool,
    light_left: bool,
    light_right: bool,
    // Ground roughness controls
    roughness_up: bool,
    roughness_down: bool,
    // Light width controls (bottom)
    light_wider: bool,
    light_narrower: bool,
    // Light width controls (top)
    light_top_wider: bool,
    light_top_narrower: bool,
}

/// Parse a simple OBJ file with v, vn, and f (v//vn format) lines
fn parse_obj(
    obj_str: &str,
    color: [f32; 3],
    object_id: u32,
    scale: f32,
    offset: [f32; 3],
) -> (Vec<Vertex>, Vec<u32>) {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for line in obj_str.lines() {
        let line = line.trim();
        if line.starts_with("v ") {
            // Vertex position
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                positions.push([
                    x * scale + offset[0],
                    y * scale + offset[1],
                    z * scale + offset[2],
                ]);
            }
        } else if line.starts_with("vn ") {
            // Vertex normal
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1].parse().unwrap_or(0.0);
                let y: f32 = parts[2].parse().unwrap_or(0.0);
                let z: f32 = parts[3].parse().unwrap_or(0.0);
                normals.push([x, y, z]);
            }
        } else if line.starts_with("f ") {
            // Face - format: f v//vn v//vn v//vn (triangles)
            let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
            if parts.len() >= 3 {
                // Parse each vertex of the face
                let mut face_indices: Vec<u32> = Vec::new();
                for part in &parts {
                    // Parse v//vn format
                    let indices_parts: Vec<&str> = part.split("//").collect();
                    if indices_parts.len() >= 2 {
                        let v_idx: usize = indices_parts[0].parse::<usize>().unwrap_or(1) - 1;
                        let n_idx: usize = indices_parts[1].parse::<usize>().unwrap_or(1) - 1;

                        let pos = positions.get(v_idx).copied().unwrap_or([0.0, 0.0, 0.0]);
                        let norm = normals.get(n_idx).copied().unwrap_or([0.0, 1.0, 0.0]);

                        // Add vertex and get its index
                        let vertex_index = vertices.len() as u32;
                        vertices.push(Vertex {
                            position: pos,
                            color,
                            normal: norm,
                            uv: [0.0, 0.0], // No UV in this OBJ
                            object_id,
                        });
                        face_indices.push(vertex_index);
                    }
                }
                // Triangulate if more than 3 vertices (fan triangulation)
                for i in 1..face_indices.len() - 1 {
                    indices.push(face_indices[0]);
                    indices.push(face_indices[i]);
                    indices.push(face_indices[i + 1]);
                }
            }
        }
    }

    (vertices, indices)
}

// Cube vertices with colors and normals for each face
const VERTICES: &[Vertex] = &[
    // Front face (red) - normal: +Z
    Vertex {
        position: [-0.5, 0.5, 0.5],
        color: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        color: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 1.5, 0.5],
        color: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [1.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 1.5, 0.5],
        color: [1.0, 0.0, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 1.0],
        object_id: 0,
    },
    // Back face (green) - normal: -Z
    Vertex {
        position: [-0.5, 0.5, -0.5],
        color: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        uv: [0.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 1.5, -0.5],
        color: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        uv: [0.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 1.5, -0.5],
        color: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        uv: [1.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        color: [0.0, 1.0, 0.0],
        normal: [0.0, 0.0, -1.0],
        uv: [1.0, 0.0],
        object_id: 0,
    },
    // Top face (blue) - normal: +Y
    Vertex {
        position: [-0.5, 1.5, -0.5],
        color: [0.0, 0.0, 1.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 1.5, 0.5],
        color: [0.0, 0.0, 1.0],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 1.5, 0.5],
        color: [0.0, 0.0, 1.0],
        normal: [0.0, 1.0, 0.0],
        uv: [1.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 1.5, -0.5],
        color: [0.0, 0.0, 1.0],
        normal: [0.0, 1.0, 0.0],
        uv: [1.0, 0.0],
        object_id: 0,
    },
    // Bottom face (yellow) - normal: -Y
    Vertex {
        position: [-0.5, 0.5, -0.5],
        color: [1.0, 1.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        uv: [0.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        color: [1.0, 1.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        uv: [1.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        color: [1.0, 1.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        uv: [1.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        color: [1.0, 1.0, 0.0],
        normal: [0.0, -1.0, 0.0],
        uv: [0.0, 1.0],
        object_id: 0,
    },
    // Right face (magenta) - normal: +X
    Vertex {
        position: [0.5, 0.5, -0.5],
        color: [1.0, 0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 1.5, -0.5],
        color: [1.0, 0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 1.5, 0.5],
        color: [1.0, 0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        color: [1.0, 0.0, 1.0],
        normal: [1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
        object_id: 0,
    },
    // Left face (cyan) - normal: -X
    Vertex {
        position: [-0.5, 0.5, -0.5],
        color: [0.0, 1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        color: [0.0, 1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 1.5, 0.5],
        color: [0.0, 1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
        object_id: 0,
    },
    Vertex {
        position: [-0.5, 1.5, -0.5],
        color: [0.0, 1.0, 1.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
        object_id: 0,
    },
    // Ground plane (gray) - normal: +Y - UV scaled for tiling
    Vertex {
        position: [-15.0, -0.5, -15.0],
        color: [0.4, 0.4, 0.4],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        object_id: 1,
    },
    Vertex {
        position: [-15.0, -0.5, 15.0],
        color: [0.4, 0.4, 0.4],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0, 15.0],
        object_id: 1,
    },
    Vertex {
        position: [15.0, -0.5, 15.0],
        color: [0.4, 0.4, 0.4],
        normal: [0.0, 1.0, 0.0],
        uv: [15.0, 15.0],
        object_id: 1,
    },
    Vertex {
        position: [15.0, -0.5, -15.0],
        color: [0.4, 0.4, 0.4],
        normal: [0.0, 1.0, 0.0],
        uv: [15.0, 0.0],
        object_id: 1,
    },
    // Area light quad (emissive) - normal: -X (facing left, toward scene)
    // Using color > 1.0 to mark as emissive in shader
    Vertex {
        position: [4.0, -0.5, -0.5],
        color: [100.0, 95.0, 90.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 0.0],
        object_id: 2,
    },
    Vertex {
        position: [4.0, -0.5, 0.5],
        color: [100.0, 95.0, 90.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 0.0],
        object_id: 2,
    },
    Vertex {
        position: [4.0, 2.5, 0.5],
        color: [100.0, 95.0, 90.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [1.0, 1.0],
        object_id: 2,
    },
    Vertex {
        position: [4.0, 2.5, -0.5],
        color: [100.0, 95.0, 90.0],
        normal: [-1.0, 0.0, 0.0],
        uv: [0.0, 1.0],
        object_id: 2,
    },
    // Left wall (gray) - normal: +Z (facing toward camera), behind the cube
    Vertex {
        position: [-4.0, -0.5, -4.0],
        color: [0.5, 0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 0.0],
        object_id: 3,
    },
    Vertex {
        position: [2.5, -0.5, -4.0],
        color: [0.5, 0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
        uv: [3.25, 0.0],
        object_id: 3,
    },
    Vertex {
        position: [2.5, 3.0, -4.0],
        color: [0.5, 0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
        uv: [3.25, 1.75],
        object_id: 3,
    },
    Vertex {
        position: [-4.0, 3.0, -4.0],
        color: [0.5, 0.5, 0.5],
        normal: [0.0, 0.0, 1.0],
        uv: [0.0, 1.75],
        object_id: 3,
    },
];

#[rustfmt::skip]
const INDICES: &[u16] = &[
    0,  1,  2,  0,  2,  3,  // front
    4,  5,  6,  4,  6,  7,  // back
    8,  9,  10, 8,  10, 11, // top
    12, 13, 14, 12, 14, 15, // bottom
    16, 17, 18, 16, 18, 19, // right
    20, 21, 22, 20, 22, 23, // left
    24, 25, 26, 24, 26, 27, // ground
    28, 29, 30, 28, 30, 31, // area light
    32, 33, 34, 32, 34, 35, // left wall
];

struct GfxState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    ssao_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    ssao_bind_group_layout: wgpu::BindGroupLayout,
    ssao_bind_group: wgpu::BindGroup,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::TextureView,
    // Offscreen render targets for SSAO
    scene_color_texture: wgpu::Texture,
    scene_color_view: wgpu::TextureView,
    scene_normal_texture: wgpu::Texture,
    scene_normal_view: wgpu::TextureView,
    ssao_texture: wgpu::Texture,
    ssao_view: wgpu::TextureView,
    camera: Camera,
    light_pos: Vec3,         // Center position of the light
    light_width_bottom: f32, // Half-width at bottom of the area light
    light_width_top: f32,    // Half-width at top of the area light
    ground_roughness: f32,
    apply_gamma: f32, // 1.0 if manual gamma needed (non-sRGB surface), 0.0 otherwise
    clear_color: wgpu::Color, // Pre-adjusted for surface format
    // Ground textures
    ground_texture_views: [wgpu::TextureView; 2], // 0: checkerboard, 1: noise
    current_texture: usize,
    ltc_1_view: wgpu::TextureView,
    ltc_2_view: wgpu::TextureView,
    ltc_sampler: wgpu::Sampler,
    ground_sampler: wgpu::Sampler,
    time: f32,
}

impl GfxState {
    async fn new(window: Arc<Window>) -> Self {
        let mut size = window.inner_size();

        // On web, the canvas might not have a size yet, use a default
        if size.width == 0 || size.height == 0 {
            size = PhysicalSize::new(1280, 720);
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

        // Prefer sRGB format for correct gamma handling
        // First try to find a native sRGB format
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        log::info!("Selected surface format: {:?}", surface_format);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            // Explicitly allow the sRGB view format
            view_formats: if surface_format.is_srgb() {
                vec![]
            } else {
                // If base format is not sRGB, try to add sRGB view
                vec![surface_format.add_srgb_suffix()]
            },
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Load object and combine with static vertices
        let object_color = [0.6, 0.4, 0.2]; // Copper/bronze color
        let object_scale = 1.5;
        let object_offset = [-2.5, -0.5, 0.5]; // Position the object to the left of the cube
        let (object_vertices, object_indices) =
            parse_obj(OBJECT_OBJ, object_color, 4, object_scale, object_offset);

        // Combine static vertices with object vertices
        let mut all_vertices: Vec<Vertex> = VERTICES.to_vec();
        let static_vertex_count = all_vertices.len() as u32;
        all_vertices.extend(object_vertices);

        // Combine indices - object indices need to be offset
        let mut all_indices: Vec<u32> = INDICES.iter().map(|&i| i as u32).collect();
        for idx in object_indices {
            all_indices.push(idx + static_vertex_count);
        }

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&all_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = all_indices.len() as u32;

        // Create LTC lookup textures
        let (_, ltc_1_view) = Self::create_ltc_texture(&device, &queue, true);
        let (_, ltc_2_view) = Self::create_ltc_texture(&device, &queue, false);

        // Create ground textures (checkerboard and noise patterns)
        let checkerboard_texture_view = Self::create_checkerboard_texture(&device, &queue);
        let noise_texture_view = Self::create_noise_texture(&device, &queue);

        let ltc_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ground_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms {
                mvp: Mat4::IDENTITY.to_cols_array_2d(),
                model: Mat4::IDENTITY.to_cols_array_2d(),
                inv_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 0.0],
                roughness: 0.25,
                light_p0: [0.0; 3],
                ground_roughness: 0.1,
                light_p1: [0.0; 3],
                light_intensity: 1.0,
                light_p2: [0.0; 3],
                apply_gamma: if surface_format.is_srgb() { 0.0 } else { 1.0 },
                light_p3: [0.0; 3],
                _p3: 0.0,
                // Blue sky with subtle ground bounce SH coefficients
                sh0: [0.6, 0.7, 1.0],
                _sh0: 0.0, // L00 - overall brightness (more blue)
                sh1: [0.3, 0.4, 0.6],
                _sh1: 0.0, // L1-1 - Y gradient (brighter above, darker below)
                sh2: [0.0, 0.0, 0.0],
                _sh2: 0.0, // L10
                sh3: [0.0, 0.0, 0.0],
                _sh3: 0.0, // L11
                sh4: [0.0, 0.0, 0.0],
                _sh4: 0.0, // L2-2
                sh5: [0.0, 0.0, 0.0],
                _sh5: 0.0, // L2-1
                sh6: [0.3, 0.2, -0.2],
                _sh6: 0.0, // L20 - strong vertical contrast
                sh7: [0.0, 0.0, 0.0],
                _sh7: 0.0, // L21
                sh8: [0.0, 0.0, 0.0],
                _sh8: 0.0, // L22
                // Emissive cube at origin, half-size 0.5 (full cube size = 1)
                cube_center: [0.0, 1.0, 0.0],
                cube_half_size: 0.5,
                cube_rotation: 0.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&ltc_1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&ltc_2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&ltc_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&checkerboard_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&ground_sampler),
                },
            ],
            label: Some("bind_group"),
        });

        let ground_texture_views = [checkerboard_texture_view, noise_texture_view];

        let depth_texture = Self::create_depth_texture(&device, &config);

        // Create offscreen render targets for SSAO
        let (scene_color_texture, scene_color_view) = Self::create_render_texture(
            &device,
            &config,
            "Scene Color",
            wgpu::TextureFormat::Rgba16Float,
        );
        let (scene_normal_texture, scene_normal_view) = Self::create_render_texture(
            &device,
            &config,
            "Scene Normal",
            wgpu::TextureFormat::Rgba16Float,
        );
        let (ssao_texture, ssao_view) =
            Self::create_render_texture(&device, &config, "SSAO", wgpu::TextureFormat::R8Unorm);

        // SSAO bind group layout (group 1) - for SSAO pass
        let ssao_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Depth,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("ssao_bind_group_layout"),
            });

        // Composite bind group layout (group 1) - for composite pass (adds ssao_texture)
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
                label: Some("composite_bind_group_layout"),
            });

        let ssao_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ssao_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &ssao_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&scene_normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                },
            ],
            label: Some("ssao_bind_group"),
        });

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ssao_view),
                },
            ],
            label: Some("composite_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Main render pipeline outputs to 2 offscreen targets (color + normal) for SSAO
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // Offscreen color target
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // Offscreen normal target
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Disable culling so light is visible from both sides
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Sky pipeline - no vertex buffers, uses vertex_index to generate fullscreen triangle
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                buffers: &[], // No vertex buffers needed
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // Offscreen color target
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // Offscreen normal target
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false, // Don't write depth for sky
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // SSAO pipeline layout
        let ssao_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSAO Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &ssao_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Composite pipeline layout
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout, &composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        // SSAO pipeline
        let ssao_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSAO Pipeline"),
            layout: Some(&ssao_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_ssao"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_ssao"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Composite pipeline
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_composite"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_composite"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            sky_pipeline,
            ssao_pipeline,
            composite_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            uniform_buffer,
            bind_group,
            bind_group_layout,
            ssao_bind_group_layout,
            ssao_bind_group,
            composite_bind_group_layout,
            composite_bind_group,
            depth_texture,
            scene_color_texture,
            scene_color_view,
            scene_normal_texture,
            scene_normal_view,
            ssao_texture,
            ssao_view,
            camera: Camera::new(Vec3::new(0.0, 1.5, 3.0), -std::f32::consts::FRAC_PI_2, -0.3),
            light_pos: Vec3::new(4.0, 1.5, 0.0), // Initial light position
            light_width_bottom: 1.0,             // Initial half-width at bottom
            light_width_top: 0.5,                // Initial half-width at top
            ground_roughness: 0.05,              // Start with shiny ground
            apply_gamma: if surface_format.is_srgb() { 0.0 } else { 1.0 },
            clear_color: {
                // 0.1 linear -> need to convert to sRGB if surface is non-sRGB
                let c = if surface_format.is_srgb() {
                    0.1
                } else {
                    0.1_f64.powf(1.0 / 2.2)
                };
                wgpu::Color {
                    r: c,
                    g: c,
                    b: c,
                    a: 1.0,
                }
            },
            ground_texture_views,
            current_texture: 0,
            ltc_1_view,
            ltc_2_view,
            ltc_sampler,
            ground_sampler,
            time: 0.0,
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };
        let texture = device.create_texture(&desc);
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_render_texture(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        label: &str,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_ltc_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        is_ltc1: bool,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        // Use pre-computed LTC lookup table data from reference implementation
        // 64x64 RGBA texture
        let size = 64u32;

        // Get the reference data (already 64*64*4 = 16384 floats)
        let data: &[f32; 16384] = if is_ltc1 {
            ltc_lut::LTC_1
        } else {
            ltc_lut::LTC_2
        };

        // Convert f32 data to f16 for Rgba16Float format
        let data_f16: Vec<half::f16> = data.iter().map(|&f| half::f16::from_f32(f)).collect();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(if is_ltc1 {
                "LTC1 Texture"
            } else {
                "LTC2 Texture"
            }),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data_f16),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4 * 2), // 4 f16s * 2 bytes per f16
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_checkerboard_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> wgpu::TextureView {
        // Create a checkerboard texture for the ground
        let size = 512u32;
        let checker_size = 64u32; // Size of each checker square
        let mut data = vec![0u8; (size * size * 4) as usize];

        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;
                let checker_x = (x / checker_size) % 2;
                let checker_y = (y / checker_size) % 2;
                let is_white = (checker_x + checker_y).is_multiple_of(2);

                if is_white {
                    // Light gray
                    data[idx] = 200;
                    data[idx + 1] = 200;
                    data[idx + 2] = 200;
                } else {
                    // Dark gray
                    data[idx] = 50;
                    data[idx + 1] = 50;
                    data[idx + 2] = 50;
                }
                data[idx + 3] = 255;
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Checkerboard Texture"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn create_noise_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
        // Create a procedural noise texture using value noise
        let size = 512u32;
        let mut data = vec![0u8; (size * size * 4) as usize];

        // Simple hash function for pseudo-random values
        fn hash(x: u32, y: u32) -> u32 {
            let mut h = x.wrapping_mul(374761393);
            h = h.wrapping_add(y.wrapping_mul(668265263));
            h = (h ^ (h >> 13)).wrapping_mul(1274126177);
            h ^ (h >> 16)
        }

        // Smoothstep for interpolation
        fn smoothstep(t: f32) -> f32 {
            t * t * (3.0 - 2.0 * t)
        }

        // Tileable value noise function - wraps at tile_size
        fn value_noise_tiled(x: f32, y: f32, tile_size: u32) -> f32 {
            let xi = x.floor() as u32;
            let yi = y.floor() as u32;
            let xf = x - x.floor();
            let yf = y - y.floor();

            // Wrap coordinates at tile boundary
            let x0 = xi % tile_size;
            let y0 = yi % tile_size;
            let x1 = (xi + 1) % tile_size;
            let y1 = (yi + 1) % tile_size;

            // Get corner values with wrapped coordinates
            let v00 = (hash(x0, y0) & 0xFF) as f32 / 255.0;
            let v10 = (hash(x1, y0) & 0xFF) as f32 / 255.0;
            let v01 = (hash(x0, y1) & 0xFF) as f32 / 255.0;
            let v11 = (hash(x1, y1) & 0xFF) as f32 / 255.0;

            // Smooth interpolation
            let sx = smoothstep(xf);
            let sy = smoothstep(yf);

            let v0 = v00 + sx * (v10 - v00);
            let v1 = v01 + sx * (v11 - v01);
            v0 + sy * (v1 - v0)
        }

        // Tileable Fractal Brownian Motion (fBm)
        // Each octave must tile at increasing frequencies
        fn fbm_tiled(x: f32, y: f32, base_tile: u32) -> f32 {
            let mut value = 0.0;
            let mut amplitude = 0.5;
            let mut frequency = 1.0f32;
            let mut tile_size = base_tile;
            for _ in 0..5 {
                value += amplitude * value_noise_tiled(x * frequency, y * frequency, tile_size);
                amplitude *= 0.5;
                frequency *= 2.0;
                tile_size *= 2; // Double tile size for each octave to maintain tiling
            }
            value
        }

        // Base tile size - noise will tile after this many units
        let base_tile = 8u32;

        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;

                // Scale coordinates so we get base_tile tiles across the texture
                let nx = x as f32 / size as f32 * base_tile as f32;
                let ny = y as f32 / size as f32 * base_tile as f32;

                // Generate tileable noise value
                let noise = fbm_tiled(nx, ny, base_tile);

                // Map to grayscale with some contrast
                let gray = (noise * 200.0 + 5.0).clamp(5.0, 230.0) as u8;

                data[idx] = gray;
                data[idx + 1] = gray;
                data[idx + 2] = gray;
                data[idx + 3] = 255;
            }
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Noise Texture"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );

        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn switch_texture(&mut self) {
        // Toggle between textures
        self.current_texture = (self.current_texture + 1) % 2;

        // Recreate bind group with the new texture
        self.bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.ltc_1_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.ltc_2_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.ltc_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(
                        &self.ground_texture_views[self.current_texture],
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.ground_sampler),
                },
            ],
            label: Some("bind_group"),
        });

        let tex_name = if self.current_texture == 0 {
            "checkerboard"
        } else {
            "noise"
        };
        println!("Switched to {} texture", tex_name);
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = Self::create_depth_texture(&self.device, &self.config);

            // Recreate offscreen textures for SSAO
            let (scene_color_texture, scene_color_view) = Self::create_render_texture(
                &self.device,
                &self.config,
                "Scene Color",
                wgpu::TextureFormat::Rgba16Float,
            );
            let (scene_normal_texture, scene_normal_view) = Self::create_render_texture(
                &self.device,
                &self.config,
                "Scene Normal",
                wgpu::TextureFormat::Rgba16Float,
            );
            let (ssao_texture, ssao_view) = Self::create_render_texture(
                &self.device,
                &self.config,
                "SSAO",
                wgpu::TextureFormat::R8Unorm,
            );

            self.scene_color_texture = scene_color_texture;
            self.scene_color_view = scene_color_view;
            self.scene_normal_texture = scene_normal_texture;
            self.scene_normal_view = scene_normal_view;
            self.ssao_texture = ssao_texture;
            self.ssao_view = ssao_view;

            // Recreate SSAO bind group with new textures
            let ssao_sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            self.ssao_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Bind Group"),
                layout: &self.ssao_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.depth_texture),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.scene_normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                    },
                ],
            });

            self.composite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Bind Group"),
                layout: &self.composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.scene_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&ssao_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&self.ssao_view),
                    },
                ],
            });
        }
    }

    fn update(&mut self, input: &mut InputState) {
        const MOVE_SPEED: f32 = 0.05;
        const LIGHT_MOVE_SPEED: f32 = 0.05;
        const MOUSE_SENSITIVITY: f32 = 0.002;

        // Update time for cube rotation
        self.time += 0.02; // ~1.2 rad/sec at 60fps

        // Update camera rotation from mouse
        self.camera.yaw += input.mouse_delta.0 as f32 * MOUSE_SENSITIVITY;
        self.camera.pitch -= input.mouse_delta.1 as f32 * MOUSE_SENSITIVITY;
        self.camera.pitch = self.camera.pitch.clamp(-1.5, 1.5); // Limit vertical look
        input.mouse_delta = (0.0, 0.0); // Reset mouse delta

        // Update camera position from keyboard
        let forward = self.camera.forward();
        let right = self.camera.right();
        let mut movement = Vec3::ZERO;

        if input.forward {
            movement += forward;
        }
        if input.backward {
            movement -= forward;
        }
        if input.right {
            movement += right;
        }
        if input.left {
            movement -= right;
        }

        if movement.length_squared() > 0.0 {
            self.camera.position += movement.normalize() * MOVE_SPEED;
        }

        // Update light position from arrow keys
        if input.light_up {
            self.light_pos.y += LIGHT_MOVE_SPEED;
        }
        if input.light_down {
            self.light_pos.y -= LIGHT_MOVE_SPEED;
        }
        if input.light_left {
            self.light_pos.z -= LIGHT_MOVE_SPEED;
        }
        if input.light_right {
            self.light_pos.z += LIGHT_MOVE_SPEED;
        }

        // Update ground roughness with [ and ] keys
        const ROUGHNESS_SPEED: f32 = 0.01;
        if input.roughness_up {
            self.ground_roughness = (self.ground_roughness + ROUGHNESS_SPEED).min(1.0);
        }
        if input.roughness_down {
            self.ground_roughness = (self.ground_roughness - ROUGHNESS_SPEED).max(0.001);
        }

        // Update light width with - and = keys (bottom), 9 and 0 keys (top)
        const WIDTH_SPEED: f32 = 0.02;
        if input.light_wider {
            self.light_width_bottom = (self.light_width_bottom + WIDTH_SPEED).min(5.0);
        }
        if input.light_narrower {
            self.light_width_bottom = (self.light_width_bottom - WIDTH_SPEED).max(0.1);
        }
        if input.light_top_wider {
            self.light_width_top = (self.light_width_top + WIDTH_SPEED).min(5.0);
        }
        if input.light_top_narrower {
            self.light_width_top = (self.light_width_top - WIDTH_SPEED).max(0.1);
        }

        let aspect = self.size.width as f32 / self.size.height as f32;
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), aspect, 0.1, 100.0);
        let view = self.camera.view_matrix();
        let model = Mat4::IDENTITY;
        let mvp = proj * view * model;
        let inv_view_proj = (proj * view).inverse();

        // Area light trapezoid dimensions
        let light_half_width_bottom = self.light_width_bottom;
        let light_half_width_top = self.light_width_top;
        let light_half_height = 1.5;

        // Corners in counter-clockwise order when viewed from the left (normal -X)
        // Bottom uses light_half_width_bottom, top uses light_half_width_top
        let light_p0 = Vec3::new(
            self.light_pos.x,
            self.light_pos.y - light_half_height,
            self.light_pos.z - light_half_width_bottom,
        );
        let light_p1 = Vec3::new(
            self.light_pos.x,
            self.light_pos.y - light_half_height,
            self.light_pos.z + light_half_width_bottom,
        );
        let light_p2 = Vec3::new(
            self.light_pos.x,
            self.light_pos.y + light_half_height,
            self.light_pos.z + light_half_width_top,
        );
        let light_p3 = Vec3::new(
            self.light_pos.x,
            self.light_pos.y + light_half_height,
            self.light_pos.z - light_half_width_top,
        );

        // Update the light quad vertices in the vertex buffer
        let light_vertices = [
            Vertex {
                position: light_p0.to_array(),
                color: [100.0, 95.0, 90.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [0.0, 0.0],
                object_id: 2,
            },
            Vertex {
                position: light_p1.to_array(),
                color: [100.0, 95.0, 90.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [1.0, 0.0],
                object_id: 2,
            },
            Vertex {
                position: light_p2.to_array(),
                color: [100.0, 95.0, 90.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [1.0, 1.0],
                object_id: 2,
            },
            Vertex {
                position: light_p3.to_array(),
                color: [100.0, 95.0, 90.0],
                normal: [-1.0, 0.0, 0.0],
                uv: [0.0, 1.0],
                object_id: 2,
            },
        ];
        // Light quad starts at vertex index 28 (after cube 24 + ground 4)
        let light_offset = 28 * std::mem::size_of::<Vertex>();
        self.queue.write_buffer(
            &self.vertex_buffer,
            light_offset as u64,
            bytemuck::cast_slice(&light_vertices),
        );

        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                mvp: mvp.to_cols_array_2d(),
                model: model.to_cols_array_2d(),
                inv_view_proj: inv_view_proj.to_cols_array_2d(),
                camera_pos: self.camera.position.to_array(),
                roughness: 0.2,
                light_p0: light_p1.to_array(), // Reversed winding: p1,p0,p3,p2
                ground_roughness: self.ground_roughness,
                light_p1: light_p0.to_array(),
                light_intensity: 8.0,
                light_p2: light_p3.to_array(),
                apply_gamma: self.apply_gamma,
                light_p3: light_p2.to_array(),
                _p3: 0.0,
                // Blue sky with subtle ground bounce SH coefficients
                sh0: [0.6, 0.7, 1.0],
                _sh0: 0.0,
                sh1: [0.3, 0.4, 0.6],
                _sh1: 0.0,
                sh2: [0.0, 0.0, 0.0],
                _sh2: 0.0,
                sh3: [0.0, 0.0, 0.0],
                _sh3: 0.0,
                sh4: [0.0, 0.0, 0.0],
                _sh4: 0.0,
                sh5: [0.0, 0.0, 0.0],
                _sh5: 0.0,
                sh6: [0.3, 0.2, -0.2],
                _sh6: 0.0,
                sh7: [0.0, 0.0, 0.0],
                _sh7: 0.0,
                sh8: [0.0, 0.0, 0.0],
                _sh8: 0.0,
                cube_center: [0.0, 1.0, 0.0],
                cube_half_size: 0.5,
                cube_rotation: self.time,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            }]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Pass 1: Render scene to offscreen targets (color + normal)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.clear_color),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.scene_normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.5,
                                g: 0.5,
                                b: 1.0,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Draw sky first (at far plane)
            render_pass.set_pipeline(&self.sky_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1); // Fullscreen triangle

            // Draw scene
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        // Pass 2: SSAO
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 1.0,
                            b: 1.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.ssao_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &self.ssao_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        // Pass 3: Composite (combine scene color with SSAO)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.composite_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &self.composite_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct App {
    window: Option<Arc<Window>>,
    gfx_state: Option<GfxState>,
    input: InputState,
    mouse_captured: bool,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gfx_state: None,
            input: InputState::default(),
            mouse_captured: false,
        }
    }

    fn capture_mouse(&mut self, capture: bool) {
        if let Some(window) = &self.window {
            self.mouse_captured = capture;
            if capture {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                window.set_cursor_visible(false);
            } else {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
            }
        }
    }
}

impl ApplicationHandler for App {
    #[allow(unused_mut)]
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = Window::default_attributes()
            .with_title("WGPU Cube - Click to capture mouse, Escape to release")
            .with_inner_size(LogicalSize::new(1280, 720));

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.get_element_by_id("canvas"))
                .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
                .expect("Failed to find canvas element");
            window_attributes = window_attributes.with_canvas(Some(canvas));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        self.window = Some(window.clone());

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.gfx_state = Some(pollster::block_on(GfxState::new(window)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            // On web, we need to spawn the async initialization
            // For simplicity, we'll use a static to store the state later
            let window_clone = window.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let state = GfxState::new(window_clone).await;
                // Store state in a thread-local for later retrieval
                PENDING_STATE.with(|ps| {
                    *ps.borrow_mut() = Some(state);
                });
            });
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(state) = &mut self.gfx_state {
                    state.resize(physical_size);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == ElementState::Pressed {
                    self.capture_mouse(true);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) => self.input.forward = pressed,
                    PhysicalKey::Code(KeyCode::KeyS) => self.input.backward = pressed,
                    PhysicalKey::Code(KeyCode::KeyA) => self.input.left = pressed,
                    PhysicalKey::Code(KeyCode::KeyD) => self.input.right = pressed,
                    PhysicalKey::Code(KeyCode::ArrowUp) => self.input.light_up = pressed,
                    PhysicalKey::Code(KeyCode::ArrowDown) => self.input.light_down = pressed,
                    PhysicalKey::Code(KeyCode::ArrowLeft) => self.input.light_left = pressed,
                    PhysicalKey::Code(KeyCode::ArrowRight) => self.input.light_right = pressed,
                    PhysicalKey::Code(KeyCode::BracketRight) => self.input.roughness_up = pressed,
                    PhysicalKey::Code(KeyCode::BracketLeft) => self.input.roughness_down = pressed,
                    PhysicalKey::Code(KeyCode::Equal) => self.input.light_wider = pressed,
                    PhysicalKey::Code(KeyCode::Minus) => self.input.light_narrower = pressed,
                    PhysicalKey::Code(KeyCode::Digit0) => self.input.light_top_wider = pressed,
                    PhysicalKey::Code(KeyCode::Digit9) => self.input.light_top_narrower = pressed,
                    PhysicalKey::Code(KeyCode::KeyT) if pressed => {
                        if let Some(state) = &mut self.gfx_state {
                            state.switch_texture();
                        }
                    }
                    PhysicalKey::Code(KeyCode::Escape) if pressed => self.capture_mouse(false),
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                // On web, check if async initialization completed
                #[cfg(target_arch = "wasm32")]
                if self.gfx_state.is_none() {
                    PENDING_STATE.with(|ps| {
                        if let Some(state) = ps.borrow_mut().take() {
                            self.gfx_state = Some(state);
                        }
                    });
                }

                if let Some(state) = &mut self.gfx_state {
                    state.update(&mut self.input);
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if self.mouse_captured
            && let DeviceEvent::MouseMotion { delta } = event
        {
            self.input.mouse_delta.0 += delta.0;
            self.input.mouse_delta.1 += delta.1;
        }
    }
}

pub fn run() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let app = App::new();

    #[cfg(not(target_arch = "wasm32"))]
    event_loop.run_app(&mut { app }).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}
