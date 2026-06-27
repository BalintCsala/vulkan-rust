use ash::vk;
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
pub(crate) struct PipelineDefinition {
    pub(crate) struct_name: String,
    pub(crate) shader_path: String,
    pub(crate) shader_info: ShaderInfo,
}
