use std::{ops::Deref, sync::Arc};

use ash::vk;

use crate::{utility_functions::assign_debug_name, wrappers::device::Device};

pub struct Fence {
    handle: vk::Fence,
    device: Arc<Device>,
}

impl Fence {
    pub fn new(device: Arc<Device>, create_info: &vk::FenceCreateInfo, name: &str) -> Self {
        let handle = unsafe { device.create_fence(create_info, None).unwrap() };
        assign_debug_name(&device, handle, name);
        Self { handle, device }
    }

    pub fn reset(&self) {
        unsafe {
            self.device.reset_fences(&[self.handle]).unwrap();
        };
    }

    pub fn wait(&self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.handle], true, u64::MAX)
                .unwrap()
        };
    }
}

impl Deref for Fence {
    type Target = vk::Fence;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.handle, None);
        };
    }
}
