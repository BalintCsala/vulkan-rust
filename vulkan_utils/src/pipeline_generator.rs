use std::{fs::File, path::PathBuf, process::Command, sync::Arc};

use ash::{util::read_spv, vk};

use crate::wrappers::device::Device;

pub mod pipeline_types;
pub(crate) mod types;

fn compile_shader(path: &str) -> Vec<u32> {
    let path = PathBuf::from(path);
    let file_name = path.file_name().unwrap();
    let out_path = PathBuf::from("spv").join(file_name).with_extension("spv");
    Command::new("slangc")
        .arg(path)
        .arg("-fvk-use-c-layout")
        .arg("-fvk-use-entrypoint-name")
        .arg("-I")
        .arg("shaders/")
        .arg("-o")
        .arg(&out_path)
        .spawn()
        .expect("Failed to build shader {path}")
        .wait()
        .unwrap();

    println!("{:?}", out_path);

    read_spv(&mut File::open(out_path).unwrap()).unwrap()
}

fn create_shader_module(device: &Arc<Device>, path: &str) -> vk::ShaderModule {
    let code = compile_shader(path);
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .unwrap()
    }
}
