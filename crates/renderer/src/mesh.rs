use ash::vk;
use bevy::math::{U16Vec4, Vec2, Vec3, Vec4};

pub enum GeometryType {
    Opaque,
    Cutout,
    Translucent,
}

pub trait Mesh {
    fn indices(&self) -> &Vec<u32>;
    fn positions(&self) -> &Vec<Vec3>;
    fn normals(&self) -> Option<&Vec<Vec3>>;
    fn tangents(&self) -> Option<&Vec<Vec4>>;
    fn texcoords_0(&self) -> Option<&Vec<Vec2>>;
    fn texcoords_1(&self) -> Option<&Vec<Vec2>>;
    fn colors(&self) -> Option<&Vec<Vec4>>;
    fn joints(&self) -> Option<&Vec<U16Vec4>>;
    fn weights(&self) -> Option<&Vec<Vec4>>;
    fn geometry_type(&self) -> &GeometryType;
    fn name(&self) -> &str;
}

pub struct GpuMesh {
    pub indices: vk::DeviceAddress,
    pub positions: vk::DeviceAddress,
    pub normals: vk::DeviceAddress,
    pub tangents: vk::DeviceAddress,
    pub texcoords_0: vk::DeviceAddress,
    pub texcoords_1: vk::DeviceAddress,
    pub colors: vk::DeviceAddress,
    pub joints: vk::DeviceAddress,
    pub weights: vk::DeviceAddress,
}
