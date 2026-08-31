use std::ops::Deref;

use ash::vk;

pub struct Instance {
    handle: ash::Instance,
}

impl Instance {
    pub fn new(entry: &ash::Entry, create_info: &vk::InstanceCreateInfo) -> Self {
        let handle = unsafe { entry.create_instance(create_info, None).unwrap() };
        Self { handle }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_instance(None);
        };
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}
