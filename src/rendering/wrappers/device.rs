use std::{ops::Deref, sync::Arc};

use ash::vk;

use crate::rendering::wrappers::instance::Instance;

pub struct Device {
    handle: ash::Device,
    _instance: Arc<Instance>,
}

impl Device {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: &vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
    ) -> Self {
        let handle = unsafe {
            instance
                .create_device(*physical_device, create_info, None)
                .unwrap()
        };
        Self {
            handle,
            _instance: instance,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_device(None);
        };
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
