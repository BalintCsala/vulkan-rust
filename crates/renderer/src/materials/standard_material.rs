use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkan_utils::pipeline_generator::pipeline_types::GraphicsPipeline;

use crate::{
    materials::Material,
    resource_manager::{ImageReference, SamplerReference},
};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct StandardMaterialData {
    pub base_color_texture: ImageReference,
    pub base_color_texcoord_id: u8,
    pub base_color_sampler: SamplerReference,
    pub base_color_factor: [f32; 4],

    pub normal_texture: ImageReference,
    pub normal_texcoord_id: u8,
    pub normal_sampler: SamplerReference,

    pub metallic_roughness_texture: ImageReference,
    pub metallic_roughness_texcoord_id: u8,
    pub metallic_roughness_sampler: SamplerReference,

    pub emissive_texture: ImageReference,
    pub emissive_texcoord_id: u8,
    pub emissive_sampler: SamplerReference,
    pub emissive_factor: [f32; 3],
}

pub struct StandardMaterial {
    pipeline: Arc<GraphicsPipeline>,
    data: StandardMaterialData,
}

impl Material<StandardMaterialData> for StandardMaterial {
    fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.pipeline
    }

    fn data(&self) -> StandardMaterialData {
        self.data
    }
}
