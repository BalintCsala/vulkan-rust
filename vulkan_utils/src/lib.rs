use std::fs::{File, read_dir};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::pipeline_generator::types::PipelineDefinition;

pub mod complex_types;
pub mod pipeline_generator;
pub mod utility_functions;
pub mod wrappers;

fn create_padding(size: u32, name: &str) -> TokenStream {
    let ident = format_ident!("{}", name);
    match size {
        1 => quote! { pub #ident: u8 },
        2 => quote! { pub #ident: u16 },
        4 => quote! { pub #ident: u32 },
        _ => panic!("Can't pad by {} bytes", size),
    }
}

fn to_snake_case(input: &str) -> String {
    let mut name = String::new();

    for character in input.chars() {
        match character {
            'a'..='z' => name.push(character),
            'A'..='Z' => {
                if !name.is_empty() {
                    name.push('_');
                }
                name.push(character.to_ascii_lowercase());
            }
            _ => panic!("Invalid character in name '{}': {}", name, character),
        }
    }

    name
}

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

        let struct_name = format_ident!("{}", definition.struct_name);
        let create_function_name =
            format_ident!("create_{}", to_snake_case(&definition.struct_name));
        let push_constants_struct_name = format_ident!("{struct_name}PushConstants");

        let mut push_constant_fields = Vec::new();
        if !definition.inputs.is_empty() {
            let mut offset = 0u32;
            let mut required_pads = 0u32;
            let mut max_alignment = 0;
            for input in definition.inputs {
                max_alignment = max_alignment.max(input.ty.alignment());
                let required_padding =
                    offset.div_ceil(input.ty.alignment()) * input.ty.alignment() - offset;
                if required_padding != 0 {
                    push_constant_fields.push(create_padding(
                        required_padding,
                        &format!("_pad{}", required_pads),
                    ));
                    required_pads += 1;
                    offset += required_padding;
                }
                push_constant_fields.push(input.to_code());
                offset += input.ty.size();
            }
            let required_end_padding = offset.div_ceil(max_alignment) * max_alignment - offset;
            if required_end_padding > 0 {
                push_constant_fields.push(create_padding(
                    required_end_padding,
                    &format!("_pad{}", required_pads),
                ));
                offset += required_end_padding;
            }

            if offset > 256 {
                panic!(
                    "Push constants for {} take up {} > 256 bytes",
                    struct_name, offset
                );
            }
        }

        let constructor_function = match definition.shader_info {
            pipeline_generator::types::ShaderInfo::Compute { entry: _ } => quote! {
                pub fn #create_function_name(device: Arc<Device>, pipeline_layout: vk::PipelineLayout) -> ComputePipeline {
                    ComputePipeline::new(#path.to_owned(), device, pipeline_layout)
                }
            },
            pipeline_generator::types::ShaderInfo::Graphics {
                vertex: _,
                fragment: _,
                color_attachments: _,
                depth_attachment: _,
            } => quote! {
                pub fn #create_function_name(device: Arc<Device>, pipeline_layout: vk::PipelineLayout) -> GraphicsPipeline {
                    GraphicsPipeline::new(#path.to_owned(), device, pipeline_layout)
                }
            },
            pipeline_generator::types::ShaderInfo::Raytracing {
                materials: _,
                raygen: _,
                miss: _,
            } => quote! {
                pub fn #create_function_name(
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

        source.push_str(&prettyplease::unparse(
            &syn::parse2(quote! {
                #[repr(C)]
                #[derive(Copy, Clone, Pod, Zeroable)]
                pub struct #push_constants_struct_name {
                    #(#push_constant_fields),*
                }

                #constructor_function
            })
            .unwrap(),
        ));

        source.push_str("\n\n");
    }
    source
}
