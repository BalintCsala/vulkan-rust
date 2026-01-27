use ash::vk;

use crate::rendering::buffer::Buffer;

#[repr(C)]
pub struct ModelData {
    pub positions: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub tangents: vk::DeviceAddress,
    pub texcoords_0: vk::DeviceAddress,
    pub texcoords_1: vk::DeviceAddress,
    pub colors: vk::DeviceAddress,
    pub joints: vk::DeviceAddress,
    pub weights: vk::DeviceAddress,
}

pub struct Model {
    pub model_data: vk::DeviceAddress,
    pub indices: Buffer,
    pub indices_count: u32,
    pub index_type: vk::IndexType,
}
