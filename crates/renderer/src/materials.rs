pub mod standard_material;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkan_utils::pipeline_generator::pipeline_types::GraphicsPipeline;

pub trait Material<T: Copy + Clone + Pod + Zeroable> {
    fn pipeline(&self) -> &Arc<GraphicsPipeline>;
    fn data(&self) -> T;
}
