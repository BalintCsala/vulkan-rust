use std::{
    ffi::CString,
    fs::{self, File},
};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
enum ColorAttachment {
    #[serde(rename = "r32ui")]
    R32UI,
}

impl ColorAttachment {
    pub fn to_write_mask(&self) -> TokenStream {
        match self {
            ColorAttachment::R32UI => quote! { vk::ColorComponentFlags::R },
        }
    }

    pub fn to_format(&self) -> TokenStream {
        match self {
            ColorAttachment::R32UI => quote! { vk::Format::R32_UINT },
        }
    }
}

#[derive(Serialize, Deserialize)]
enum DepthAttachment {
    #[serde(rename = "d32sfloat")]
    D32Sfloat,
}

impl DepthAttachment {
    pub fn to_format(&self) -> TokenStream {
        match self {
            DepthAttachment::D32Sfloat => quote! { vk::Format::D32_SFLOAT },
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum ShaderInfo {
    Compute {
        entry: String,
    },
    #[serde(rename_all = "camelCase")]
    Graphics {
        vertex: String,
        fragment: String,
        color_attachments: Vec<ColorAttachment>,
        depth_attachment: Option<DepthAttachment>,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum PipelineInput {}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PipelineDefinition {
    struct_name: String,
    shader_path: String,
    shader_info: ShaderInfo,
    inputs: Vec<PipelineInput>,
}

pub fn generate_pipeline_code() -> String {
    let pipeline_files = fs::read_dir("./pipelines/").unwrap();

    let mut source = prettyplease::unparse(
        &syn::parse2(quote! {
            use std::{fs::File, sync::Arc, mem, path::PathBuf, process::Command};

            use ash::{util::read_spv, vk};

            use crate::rendering::{wrappers::device::Device};


            fn compile_shader(path: &str) -> Vec<u32> {
                let path = PathBuf::from(path);
                let file_name = path.file_name().unwrap();
                let out_path = PathBuf::from("spv").join(file_name).with_extension("spv");
                Command::new("slangc")
                    .arg(path)
                    .arg("-fvk-use-c-layout")
                    .arg("-fvk-use-entrypoint-name")
                    .arg("-o")
                    .arg(&out_path)
                    .spawn()
                    .expect("Failed to build shader {path}");

                println!("{:?}", out_path);

                read_spv(&mut File::open(out_path).unwrap()).unwrap()
            }

            pub trait Pipeline {
                fn reload(&mut self);
                fn bind(&self, command_buffer: vk::CommandBuffer);
            }
        })
        .unwrap(),
    );

    for entry in pipeline_files.map_while(Result::ok) {
        if !entry.file_type().unwrap().is_file() {
            continue;
        }
        let file = File::open(entry.path()).unwrap();

        let definition: PipelineDefinition = serde_json::from_reader(file).unwrap();
        let struct_name = format_ident!("{}", definition.struct_name);
        let push_constants_struct_name = format_ident!("{struct_name}PushConstants");
        let shader_source_path = definition.shader_path;

        let bind_point = match definition.shader_info {
            ShaderInfo::Compute { entry: _ } => quote! { vk::PipelineBindPoint::COMPUTE  },
            ShaderInfo::Graphics {
                vertex: _,
                fragment: _,
                color_attachments: _,
                depth_attachment: _,
            } => quote! { vk::PipelineBindPoint::GRAPHICS },
        };

        let pipeline_creation = match definition.shader_info {
            ShaderInfo::Compute { entry } => {
                let entry = CString::new(entry).unwrap();
                quote! {
                    let pipeline = unsafe {
                        device.create_compute_pipelines(
                            vk::PipelineCache::null(),
                            &[vk::ComputePipelineCreateInfo::default()
                                .layout(pipeline_layout)
                                .stage(
                                    vk::PipelineShaderStageCreateInfo::default()
                                        .name(#entry)
                                        .stage(vk::ShaderStageFlags::COMPUTE)
                                        .module(module),
                                )],
                            None,
                        )
                        .unwrap()[0]
                    };
                }
            }
            ShaderInfo::Graphics {
                vertex,
                fragment,
                color_attachments,
                depth_attachment,
            } => {
                let vertex = CString::new(vertex).unwrap();
                let fragment = CString::new(fragment).unwrap();

                let color_attachment_masks =
                    color_attachments.iter().map(ColorAttachment::to_write_mask);
                let color_attachment_formats =
                    color_attachments.iter().map(ColorAttachment::to_format);

                let depth_enabled = depth_attachment.is_some();

                let depth_format = match depth_attachment {
                    Some(depth_attachment) => depth_attachment.to_format(),
                    None => quote! { vk::Format::UNDEFINED },
                };

                quote! {
                    let pipeline = unsafe {
                        device.create_graphics_pipelines(
                                vk::PipelineCache::null(),
                                &[vk::GraphicsPipelineCreateInfo::default()
                                    .layout(pipeline_layout)
                                    .stages(&[
                                        vk::PipelineShaderStageCreateInfo::default()
                                            .name(#vertex)
                                            .stage(vk::ShaderStageFlags::VERTEX)
                                            .module(module),
                                        vk::PipelineShaderStageCreateInfo::default()
                                            .name(#fragment)
                                            .stage(vk::ShaderStageFlags::FRAGMENT)
                                            .module(module),
                                    ])
                                    .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                                    .input_assembly_state(
                                        &vk::PipelineInputAssemblyStateCreateInfo::default()
                                            .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                                    )
                                    .multisample_state(
                                        &vk::PipelineMultisampleStateCreateInfo::default()
                                            .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                                    )
                                    .rasterization_state(
                                        &vk::PipelineRasterizationStateCreateInfo::default()
                                            .line_width(1.0)
                                            .cull_mode(vk::CullModeFlags::BACK)
                                            .front_face(vk::FrontFace::COUNTER_CLOCKWISE),
                                    )
                                    .dynamic_state(
                                        &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                            vk::DynamicState::VIEWPORT,
                                            vk::DynamicState::SCISSOR,
                                        ]),
                                    )
                                    .viewport_state(
                                        &vk::PipelineViewportStateCreateInfo::default()
                                            .viewport_count(1)
                                            .scissor_count(1),
                                    )
                                    .color_blend_state(
                                        &vk::PipelineColorBlendStateCreateInfo::default()
                                            .attachments(&[
                                                #(
                                                    vk::PipelineColorBlendAttachmentState::default()
                                                        .color_write_mask(#color_attachment_masks)
                                                ),*
                                            ]),
                                    )
                                    .depth_stencil_state(
                                        &vk::PipelineDepthStencilStateCreateInfo::default()
                                            .depth_write_enable(#depth_enabled)
                                            .depth_test_enable(#depth_enabled)
                                            .depth_compare_op(vk::CompareOp::LESS),
                                    )
                                    .push_next(
                                        &mut vk::PipelineRenderingCreateInfo::default()
                                            .color_attachment_formats(&[#(#color_attachment_formats),*])
                                            .depth_attachment_format(#depth_format),
                                    )],
                                None,
                            )
                            .unwrap()[0]
                    };
                }
            }
        };

        source.push_str(&prettyplease::unparse(
            &syn::parse2(quote! {
                pub struct #push_constants_struct_name {

                }

                pub struct #struct_name {
                    device: Arc<Device>,
                    pipeline_layout: vk::PipelineLayout,
                    pipeline: vk::Pipeline,
                }

                impl #struct_name {
                    pub fn new(device: Arc<Device>, pipeline_layout: vk::PipelineLayout) -> Self {
                        let pipeline = Self::load_pipeline(&device, pipeline_layout, #shader_source_path).unwrap();
                        Self {
                            device,
                            pipeline_layout,
                            pipeline,
                        }
                    }

                    fn load_pipeline(device: &Arc<Device>, pipeline_layout: vk::PipelineLayout, path: &str) -> Result<vk::Pipeline, std::io::Error> {
                        let code = compile_shader(path);
                        let module = unsafe {
                            device.create_shader_module(
                                    &vk::ShaderModuleCreateInfo::default().code(&code),
                                    None,
                                )
                                .unwrap()
                        };

                        #pipeline_creation

                        unsafe {
                            device.destroy_shader_module(module, None);
                        };

                        Ok(pipeline)
                    }
                }

                impl Pipeline for #struct_name {
                    fn reload(&mut self) {
                        let mut pipeline = Self::load_pipeline(&self.device, self.pipeline_layout, #shader_source_path).unwrap();
                        mem::swap(&mut self.pipeline, &mut pipeline);
                        unsafe {
                            self.device.destroy_pipeline(pipeline, None);
                        };
                    }

                    fn bind(&self, command_buffer: vk::CommandBuffer) {
                        unsafe {
                            self.device.cmd_bind_pipeline(
                                command_buffer,
                                #bind_point,
                                self.pipeline,
                            );
                        };
                    }
                }

                impl Drop for #struct_name {
                    fn drop(&mut self) {
                        unsafe {
                            self.device.destroy_pipeline(self.pipeline, None);
                        };
                    }
                }
            })
            .unwrap(),
        ));

        source.push_str("\n\n");
    }
    source
}
