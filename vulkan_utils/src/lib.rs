use std::fs::{File, read_dir};

use quote::{format_ident, quote};

use crate::pipeline_generator::{
    compile_shader, spirv_types::to_snake_case, types::PipelineDefinition,
};

pub mod complex_types;
pub mod pipeline_generator;
pub mod utility_functions;
pub mod wrappers;

pub fn generate_pipeline_code() -> String {
    let pipeline_files = read_dir("./pipelines/").unwrap();

    let mut source = prettyplease::unparse(
        &syn::parse2(quote! {
            use std::sync::Arc;
            use ash::{
                vk
            };
            use bytemuck::{Pod, Zeroable};
            use vulkan_utils::{
                pipeline_generator::pipeline_types::{ComputePipeline, GraphicsPipeline, RaytracingPipeline},
                wrappers::{allocator::Allocator, device::Device, instance::Instance},
            };
        })
        .unwrap(),
    );

    for entry in pipeline_files.map_while(Result::ok) {
        if !entry.file_type().unwrap().is_file() {
            continue;
        }

        let path = entry.path().to_str().unwrap().to_owned();
        let file = File::open(&path).unwrap();
        let definition: PipelineDefinition = serde_json::from_reader(file).unwrap();

        let reflection_data = compile_shader(&definition.shader_path).reflection;
        let push_constant_type = reflection_data
            .get_push_constants_type()
            .expect("No push constants found");
        let push_constant_source =
            push_constant_type.to_code(Some(format!("{}PushConstants", definition.struct_name)));

        println!("{}: {:?}", path, reflection_data);

        let constructor_function_name =
            format_ident!("create_{}", to_snake_case(&definition.struct_name));

        let constructor_function = match definition.shader_info {
            pipeline_generator::types::ShaderInfo::Compute { entry: _ } => quote! {
                pub fn #constructor_function_name(device: Arc<Device>, pipeline_layout: vk::PipelineLayout) -> ComputePipeline {
                    ComputePipeline::new(#path.to_owned(), device, pipeline_layout)
                }
            },
            pipeline_generator::types::ShaderInfo::Graphics {
                vertex: _,
                fragment: _,
                color_attachments: _,
                depth_attachment: _,
            } => quote! {
                pub fn #constructor_function_name(device: Arc<Device>, pipeline_layout: vk::PipelineLayout, surface_format: vk::Format) -> GraphicsPipeline {
                    GraphicsPipeline::new(#path.to_owned(), device, pipeline_layout, surface_format)
                }
            },
            pipeline_generator::types::ShaderInfo::Raytracing {
                materials: _,
                raygen: _,
                miss: _,
            } => quote! {
                pub fn #constructor_function_name(
                    instance: Arc<Instance>,
                    physical_device: vk::PhysicalDevice,
                    device: Arc<Device>,
                    allocator: Arc<Allocator>,
                    pipeline_layout: vk::PipelineLayout
                ) -> RaytracingPipeline {
                    RaytracingPipeline::new(#path.to_owned(), instance, physical_device, device, allocator, pipeline_layout)
                }

            },
        };
        println!("{:?}", push_constant_source.to_string());

        source.push_str(&prettyplease::unparse(
            &syn::parse2(quote! {

                #[repr(C)]
                #[derive(Copy, Clone, Pod, Zeroable)]
                pub #push_constant_source

                #constructor_function
            })
            .unwrap(),
        ));

        source.push_str("\n\n");
    }
    source
}
