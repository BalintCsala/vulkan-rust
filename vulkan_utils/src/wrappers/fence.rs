use std::{ops::Deref, sync::Arc};

use ash::vk;

use crate::wrappers::device::Device;

pub struct Fence {
    fence: vk::Fence,
    device: Arc<Device>,
}

impl Fence {
    pub fn new(device: Arc<Device>, create_info: &vk::FenceCreateInfo) -> Self {
        let fence = unsafe { device.create_fence(create_info, None).unwrap() };
        Self { fence, device }
    }
}

impl Deref for Fence {
    type Target = vk::Fence;

    fn deref(&self) -> &Self::Target {
        &self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.fence, None);
        };
    }
}
