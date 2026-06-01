use std::{ops::Deref, sync::Arc};

use ash::{
    ext::debug_utils,
    khr::{acceleration_structure, ray_tracing_pipeline},
    vk,
};

use crate::wrappers::instance::Instance;

pub struct Device {
    handle: ash::Device,
    pub ray_tracing: ray_tracing_pipeline::Device,
    pub acceleration_structure: acceleration_structure::Device,
    pub debug_utils: debug_utils::Device,
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
        let ray_tracing = ray_tracing_pipeline::Device::new(&instance, &handle);
        let acceleration_structure = acceleration_structure::Device::new(&instance, &handle);
        let debug_utils = debug_utils::Device::new(&instance, &handle);
        Self {
            handle,
            ray_tracing,
            acceleration_structure,
            debug_utils,
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
