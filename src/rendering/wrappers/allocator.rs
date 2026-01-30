use std::{ops::Deref, sync::Arc};

use vk_mem::AllocatorCreateInfo;

use crate::rendering::wrappers::device::Device;

pub struct Allocator {
    handle: vk_mem::Allocator,
    _device: Arc<Device>,
}

impl Allocator {
    pub fn new(device: Arc<Device>, create_info: AllocatorCreateInfo) -> Self {
        let handle = unsafe { vk_mem::Allocator::new(create_info).unwrap() };
        Self {
            _device: device,
            handle,
        }
    }
}

impl Deref for Allocator {
    type Target = vk_mem::Allocator;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
