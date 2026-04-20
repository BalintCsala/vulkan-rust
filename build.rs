use std::fs;

fn main() {
    // Generate pipeline files
    let generated_pipelines_source = pipeline_generator::generate_pipeline_code();
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("generated_pipelines.rs");
    fs::write(&dest_path, generated_pipelines_source).unwrap();

    println!("cargo-rerun-if-changed=pipelines/");
}
