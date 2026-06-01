use crate::{format_ident, quote};
use ash::vk;
use proc_macro2::TokenStream;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(crate) enum ColorAttachment {
    #[serde(rename = "r32ui")]
    R32UI,
}

impl ColorAttachment {
    pub(crate) fn to_write_mask(&self) -> vk::ColorComponentFlags {
        match self {
            ColorAttachment::R32UI => vk::ColorComponentFlags::R,
        }
    }

    pub(crate) fn to_format(&self) -> vk::Format {
        match self {
            ColorAttachment::R32UI => vk::Format::R32_UINT,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) enum DepthAttachment {
    #[serde(rename = "d32sfloat")]
    D32Sfloat,
}

impl DepthAttachment {
    pub(crate) fn to_format(&self) -> vk::Format {
        match self {
            DepthAttachment::D32Sfloat => vk::Format::D32_SFLOAT,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RaytracingMaterial {
    pub(crate) name: String,
    pub(crate) shader_path: String,
    pub(crate) closest_hit: String,
    pub(crate) any_hit: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub(crate) enum ShaderInfo {
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
    #[serde(rename_all = "camelCase")]
    Raytracing {
        materials: Vec<RaytracingMaterial>,
        raygen: String,
        miss: String,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum PipelineInputType {
    StorageImage,
    SampledImage,
    Sampler,
    Buffer,
    Float,
    Float2,
    Float3,
    Float4,
    Float3x3,
    Float4x4,
    Uint,
    AccelerationStructure,
}

impl PipelineInputType {
    pub(crate) fn alignment(&self) -> u32 {
        match self {
            PipelineInputType::StorageImage => 4,
            PipelineInputType::SampledImage => 4,
            PipelineInputType::Sampler => 4,
            PipelineInputType::Buffer => 8,
            PipelineInputType::Float => 4,
            PipelineInputType::Float2 => 4,
            PipelineInputType::Float3 => 4,
            PipelineInputType::Float4 => 4,
            PipelineInputType::Float3x3 => 4,
            PipelineInputType::Float4x4 => 4,
            PipelineInputType::Uint => 4,
            PipelineInputType::AccelerationStructure => 8,
        }
    }

    pub(crate) fn size(&self) -> u32 {
        match self {
            PipelineInputType::StorageImage => 4,
            PipelineInputType::SampledImage => 4,
            PipelineInputType::Sampler => 4,
            PipelineInputType::Buffer => 8,
            PipelineInputType::Float => 4,
            PipelineInputType::Float2 => 8,
            PipelineInputType::Float3 => 12,
            PipelineInputType::Float4 => 16,
            PipelineInputType::Float3x3 => 36,
            PipelineInputType::Float4x4 => 64,
            PipelineInputType::Uint => 4,
            PipelineInputType::AccelerationStructure => 8,
        }
    }

    pub(crate) fn to_code(&self) -> TokenStream {
        match self {
            PipelineInputType::StorageImage => quote! { i32 },
            PipelineInputType::SampledImage => quote! { i32 },
            PipelineInputType::Sampler => quote! { i32 },
            PipelineInputType::Buffer => quote! { vk::DeviceAddress },
            PipelineInputType::Float => quote! { f32 },
            PipelineInputType::Float2 => quote! { [f32; 2] },
            PipelineInputType::Float3 => quote! { [f32; 3] },
            PipelineInputType::Float4 => quote! { [f32; 4] },
            PipelineInputType::Float3x3 => quote! { [f32; 9] },
            PipelineInputType::Float4x4 => quote! { [f32; 16] },
            PipelineInputType::Uint => quote! { u32 },
            PipelineInputType::AccelerationStructure => quote! { vk::DeviceAddress },
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PipelineInput {
    #[serde(rename = "type")]
    pub(crate) ty: PipelineInputType,
    pub(crate) name: String,
}

impl PipelineInput {
    pub(crate) fn to_code(&self) -> TokenStream {
        let ident = format_ident!("{}", self.name);
        let ty = self.ty.to_code();
        quote! { pub #ident: #ty }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct PipelineDefinition {
    pub(crate) struct_name: String,
    pub(crate) shader_path: String,
    pub(crate) shader_info: ShaderInfo,
    pub(crate) inputs: Vec<PipelineInput>,
}
