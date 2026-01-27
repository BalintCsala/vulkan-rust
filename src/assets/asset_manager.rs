use std::sync::Arc;

use ash::{ext::debug_utils, vk};
use bevy::ecs::resource::Resource;
use vk_mem::Allocator;

use crate::{assets::model::ModelData, rendering::buffer::Buffer};

const MAX_MODEL_DATA_COUNT: usize = 16384;

#[derive(Resource)]
pub struct AssetManager {
    pub model_datas: Buffer,
    next_model_data: u64,
}

impl AssetManager {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
    ) -> Self {
        let model_datas = Buffer::new(
            device.clone(),
            allocator.clone(),
            vk::BufferUsageFlags::empty(),
            (MAX_MODEL_DATA_COUNT * size_of::<ModelData>()) as u64,
        );
        model_datas.set_name(debug_utils_device, "Model data buffer");
        Self {
            model_datas,
            next_model_data: 0,
        }
    }

    pub fn upload_model_data(&mut self, model_data: ModelData) -> vk::DeviceAddress {
        let index = self.next_model_data;
        self.next_model_data += 1;

        let offset = size_of::<ModelData>() * index as usize;
        self.model_datas
            .write(&[model_data], size_of::<ModelData>() * index as usize);

        self.model_datas.address + offset as u64
    }
}
