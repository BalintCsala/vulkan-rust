use std::fs;

fn main() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    std::env::set_current_dir(manifest_dir.join("../..")).unwrap();

    // Generate pipeline files
    let generated_pipelines_source = vulkan_utils::generate_pipeline_code();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("generated_pipelines.rs");
    fs::write(&dest_path, generated_pipelines_source).unwrap();

    println!("cargo:rerun-if-changed=../../pipelines/");
    println!("cargo:rerun-if-changed=../../shaders/");
}
