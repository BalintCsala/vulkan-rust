use std::ffi::CString;

use ash::{ext::debug_utils, vk};

pub fn full_subresource_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_MIP_LEVELS)
        .aspect_mask(aspect)
}

pub fn format_to_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

pub fn image_type_to_view_type(image_type: vk::ImageType) -> vk::ImageViewType {
    match image_type {
        vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
        vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
        vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
        _ => unreachable!(),
    }
}

pub fn assign_debug_name<T: vk::Handle>(
    debug_utils_device: &debug_utils::Device,
    handle: T,
    name: &str,
) {
    unsafe {
        debug_utils_device
            .set_debug_utils_object_name(
                &vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(handle)
                    .object_name(&CString::new(name).unwrap()),
            )
            .unwrap()
    }
}
