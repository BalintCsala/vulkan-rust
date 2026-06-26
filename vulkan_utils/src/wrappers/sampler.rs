use std::{ops::Deref, sync::Arc};

use ash::vk;

use crate::wrappers::device::Device;

pub struct Sampler {
    sampler: vk::Sampler,
    device: Arc<Device>,
}

impl Sampler {
    pub fn new(device: Arc<Device>, create_info: &vk::SamplerCreateInfo) -> Self {
        let sampler = unsafe { device.create_sampler(create_info, None).unwrap() };
        Self { sampler, device }
    }
}

impl Deref for Sampler {
    type Target = vk::Sampler;

    fn deref(&self) -> &Self::Target {
        &self.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
        };
    }
}
