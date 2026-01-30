use std::{fs::File, sync::Arc};

use ash::{util::read_spv, vk};

use crate::rendering::wrappers::device::Device;

pub struct Shader {
    device: Arc<Device>,
    pub module: vk::ShaderModule,
}

impl Shader {
    pub fn new(device: Arc<Device>, path: &str) -> Result<Self, std::io::Error> {
        let mut file = File::open(path)?;
        let code = read_spv(&mut file)?;
        let module = unsafe {
            device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(code.as_slice()),
                    None,
                )
                .unwrap()
        };
        Ok(Self { device, module })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.module, None);
        };
    }
}
