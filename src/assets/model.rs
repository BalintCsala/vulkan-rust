use ash::vk;
use bevy::ecs::component::Component;

use crate::rendering::resource_manager::{ImageReference, ModelReference, SamplerReference};

#[repr(C)]
pub struct ModelData {
    pub positions: vk::DeviceAddress,
    pub indices: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub tangents: vk::DeviceAddress,
    pub texcoords_0: vk::DeviceAddress,
    pub texcoords_1: vk::DeviceAddress,
    pub colors: vk::DeviceAddress,
    pub joints: vk::DeviceAddress,
    pub weights: vk::DeviceAddress,

    pub base_color_texture_id: ImageReference,
    pub base_color_texcoord_id: u8,
    pub base_color_sampler_id: SamplerReference,

    pub normal_texture_id: ImageReference,
    pub normal_texcoord_id: u8,
    pub normal_sampler_id: SamplerReference,

    pub metallic_roughness_texture_id: ImageReference,
    pub metallic_roughness_texcoord_id: u8,
    pub metallic_roughness_sampler_id: SamplerReference,

    pub emissive_texture_id: ImageReference,
    pub emissive_texcoord_id: u8,
    pub emissive_sampler_id: SamplerReference,

    pub emissive_factor: [f32; 3],
}

#[derive(Component, Clone)]
pub struct Model {
    pub model_ref: ModelReference,
}
