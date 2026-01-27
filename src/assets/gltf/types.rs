use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct TextureInfo {
    pub(super) index: i32,
    #[serde(default = "TextureInfo::default_tex_coord")]
    pub(super) tex_coord: i32,
}

impl TextureInfo {
    fn default_tex_coord() -> i32 {
        0
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct Texture {
    pub(super) sampler: Option<i32>,
    pub(super) source: Option<i32>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Skin {
    pub(super) inverse_bind_matrices: Option<i32>,
    pub(super) skeleton: Option<i32>,
    pub(super) joints: Vec<i32>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Scene {
    pub nodes: Vec<i32>,
    pub name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Sampler {
    pub(super) mag_filter: Option<i32>,
    pub(super) min_filter: Option<i32>,
    #[serde(default = "Sampler::default_wrap_s")]
    pub(super) wrap_s: i32,
    #[serde(default = "Sampler::default_wrap_t")]
    pub(super) wrap_t: i32,
    pub(super) name: Option<String>,
}

impl Sampler {
    fn default_wrap_s() -> i32 {
        10497
    }

    fn default_wrap_t() -> i32 {
        10497
    }
}

#[derive(Serialize, Deserialize)]
pub struct Node {
    pub camera: Option<i32>,
    pub children: Option<Vec<i32>>,
    pub skin: Option<i32>,
    pub matrix: Option<[f32; 16]>,
    pub mesh: Option<i32>,
    #[serde(default = "Node::default_rotation")]
    pub rotation: [f32; 4],
    #[serde(default = "Node::default_scale")]
    pub scale: [f32; 3],
    #[serde(default = "Node::default_translation")]
    pub translation: [f32; 3],
    pub weights: Option<Vec<f32>>,
    pub name: Option<String>,
}

impl Node {
    fn default_rotation() -> [f32; 4] {
        [0.0, 0.0, 0.0, 1.0]
    }

    fn default_scale() -> [f32; 3] {
        [1.0, 1.0, 1.0]
    }

    fn default_translation() -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct Primitive {
    pub(super) attributes: HashMap<String, i32>,
    pub(super) indices: Option<i32>,
    pub(super) material: Option<i32>,
    #[serde(default = "Primitive::default_mode")]
    pub(super) mode: i32,
}

impl Primitive {
    fn default_mode() -> i32 {
        4
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct GltfMesh {
    pub(super) primitives: Vec<Primitive>,
    pub(super) weights: Option<Vec<f32>>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct PbrMetallicRoughness {
    #[serde(default = "PbrMetallicRoughness::default_base_color_factor")]
    pub(super) base_color_factor: [f32; 4],
    pub(super) base_color_texture: Option<TextureInfo>,
    #[serde(default = "PbrMetallicRoughness::default_metallic_factor")]
    pub(super) metallic_factor: f32,
    #[serde(default = "PbrMetallicRoughness::default_roughness_factor")]
    pub(super) roughness_factor: f32,
    pub(super) metallic_roughness_texture: Option<TextureInfo>,
}

impl PbrMetallicRoughness {
    fn default_base_color_factor() -> [f32; 4] {
        [1.0, 1.0, 1.0, 1.0]
    }

    fn default_metallic_factor() -> f32 {
        1.0
    }

    fn default_roughness_factor() -> f32 {
        1.0
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct NormalTexture {
    pub(super) index: i32,
    #[serde(default = "NormalTexture::default_tex_coord")]
    pub(super) tex_coord: i32,
    #[serde(default = "NormalTexture::default_scale")]
    pub(super) scale: f32,
}

impl NormalTexture {
    fn default_tex_coord() -> i32 {
        0
    }

    fn default_scale() -> f32 {
        1.0
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct OcclusionTexture {
    pub(super) index: i32,
    #[serde(default = "OcclusionTexture::default_tex_coord")]
    pub(super) tex_coord: i32,
    #[serde(default = "OcclusionTexture::default_strength")]
    pub(super) strength: f32,
}

impl OcclusionTexture {
    fn default_tex_coord() -> i32 {
        0
    }

    fn default_strength() -> f32 {
        1.0
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Material {
    pub(super) pbr_metallic_roughness: Option<PbrMetallicRoughness>,
    pub(super) normal_texture: Option<NormalTexture>,
    pub(super) occlusion_texture: Option<OcclusionTexture>,
    pub(super) emissive_texture: Option<TextureInfo>,
    #[serde(
        rename = "emissiveFactor",
        default = "Material::default_emissive_factor"
    )]
    pub(super) emissive_factor: [f32; 3],
    #[serde(default = "Material::default_alpha_mode")]
    pub(super) alpha_mode: String,
    #[serde(default = "Material::default_alpha_cutoff")]
    pub(super) alpha_cutoff: f32,
    #[serde(default = "Material::default_double_sided")]
    pub(super) double_sided: bool,
    pub(super) name: Option<String>,
}

impl Material {
    fn default_emissive_factor() -> [f32; 3] {
        [0.0, 0.0, 0.0]
    }

    fn default_alpha_mode() -> String {
        "OPAQUE".to_owned()
    }

    fn default_alpha_cutoff() -> f32 {
        0.5
    }

    fn default_double_sided() -> bool {
        false
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Image {
    pub(super) uri: Option<String>,
    pub(super) mime_type: Option<String>,
    pub(super) buffer_view: Option<i32>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Camera {}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct BufferView {
    pub(super) buffer: i32,
    #[serde(default)]
    pub(super) byte_offset: i32,
    pub(super) byte_length: i32,
    pub(super) byte_stride: Option<i32>,
    pub(super) target: Option<i32>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct GltfBuffer {
    pub(super) uri: Option<String>,
    pub(super) byte_length: i32,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Asset {
    pub(super) copyright: Option<String>,
    pub(super) generator: Option<String>,
    pub(super) version: String,
    pub(super) min_version: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub(super) struct Animation {
    pub(super) channels: Option<()>,
    pub(super) samplers: Option<()>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Accessor {
    pub(super) buffer_view: Option<i32>,
    #[serde(default)]
    pub(super) byte_offset: i32,
    pub(super) component_type: i32,
    #[serde(default)]
    pub(super) normalized: bool,
    pub(super) count: i32,
    #[serde(rename = "type")]
    pub(super) element_type: String,
    pub(super) max: Option<Vec<f32>>,
    pub(super) min: Option<Vec<f32>>,
    pub(super) sparse: Option<()>,
    pub(super) name: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Info {
    pub(super) accessors: Vec<Accessor>,
    pub(super) animations: Option<Vec<Animation>>,
    pub(super) asset: Asset,
    pub(super) buffers: Option<Vec<GltfBuffer>>,
    pub(super) buffer_views: Option<Vec<BufferView>>,
    pub(super) cameras: Option<Vec<Camera>>,
    pub(super) images: Option<Vec<Image>>,
    pub(super) materials: Option<Vec<Material>>,
    pub(super) meshes: Option<Vec<GltfMesh>>,
    pub(super) nodes: Option<Vec<Node>>,
    pub(super) samplers: Option<Vec<Sampler>>,
    pub(super) scene: Option<u32>,
    pub(super) scenes: Option<Vec<Scene>>,
    pub(super) skins: Option<Vec<Skin>>,
    pub(super) textures: Option<Vec<Texture>>,
}
