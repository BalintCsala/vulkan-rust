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
}

#[derive(Component, Clone)]
pub struct Model {
    pub model_ref: ModelReference,
}
