use std::{
    fs::{self, File},
    path::PathBuf,
    process::Command,
    sync::Arc,
};

use ash::{util::read_spv, vk};

use crate::{pipeline_generator::spirv_types::SpirvReflection, wrappers::device::Device};

pub mod pipeline_types;
pub(crate) mod spirv_types;
pub(crate) mod types;

pub(crate) struct ShaderData {
    pub(crate) code: Vec<u32>,
    pub(crate) reflection: SpirvReflection,
}

pub(crate) fn compile_shader(path: &str) -> ShaderData {
    let path = PathBuf::from(path);
    let file_name = path.file_name().unwrap();

    let spv_folder = PathBuf::from("spv");
    if !spv_folder.exists() {
        fs::create_dir(&spv_folder).unwrap();
    }

    let out_path = spv_folder.join(file_name).with_extension("spv");

    let reflection_path = std::env::temp_dir().join(file_name).with_extension("json");

    Command::new("slangc")
        .arg(path)
        .arg("-fvk-use-c-layout")
        .arg("-fvk-use-entrypoint-name")
        .arg("-I")
        .arg("shaders/")
        .arg("-o")
        .arg(&out_path)
        .arg("-reflection-json")
        .arg(&reflection_path)
        .spawn()
        .expect("Failed to build shader {path}")
        .wait()
        .unwrap();

    ShaderData {
        code: read_spv(&mut File::open(out_path).unwrap()).unwrap(),
        reflection: serde_json::from_reader(File::open(&reflection_path).unwrap()).unwrap(),
    }
}

fn create_shader_module(device: &Arc<Device>, path: &str) -> vk::ShaderModule {
    let code = compile_shader(path).code;
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .unwrap()
    }
}
