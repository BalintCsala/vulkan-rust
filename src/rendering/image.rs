use std::sync::Arc;

use ash::{ext::debug_utils, vk};
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, MemoryUsage};

use crate::rendering::{
    vulkan_utils::{
        assign_debug_name, format_to_aspect, full_subresource_range, image_type_to_view_type,
    },
    wrappers::{allocator::Allocator, device::Device},
};

pub struct Image {
    pub layout: vk::ImageLayout,
    pub format: vk::Format,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
    pub handle: vk::Image,
    allocator: Option<Arc<Allocator>>,
    device: Arc<Device>,
}

impl Image {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        image_type: vk::ImageType,
        mip_levels: u32,
        array_layers: u32,
        name: &str,
    ) -> Self {
        let (image, allocation) = unsafe {
            allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .extent(extent)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .mip_levels(mip_levels)
                        .image_type(image_type)
                        .format(format)
                        .usage(usage)
                        .array_layers(array_layers)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    &AllocationCreateInfo {
                        flags: AllocationCreateFlags::DEDICATED_MEMORY,
                        usage: MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap()
        };

        assign_debug_name(debug_utils_device, image, name);

        let view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .components(vk::ComponentMapping::default())
                        .format(format)
                        .subresource_range(full_subresource_range(format_to_aspect(format)))
                        .view_type(image_type_to_view_type(image_type)),
                    None,
                )
                .unwrap()
        };

        assign_debug_name(debug_utils_device, image, &format!("{name} main view"));

        Self {
            device,
            allocator: Some(allocator),
            handle: image,
            allocation: Some(allocation),
            view,
            layout: vk::ImageLayout::UNDEFINED,
            format,
        }
    }

    pub fn from_raw(
        device: Arc<Device>,
        image: vk::Image,
        format: vk::Format,
        layout: vk::ImageLayout,
        image_type: vk::ImageType,
    ) -> Self {
        let view = unsafe {
            device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .components(vk::ComponentMapping::default())
                        .format(format)
                        .subresource_range(full_subresource_range(format_to_aspect(format)))
                        .view_type(image_type_to_view_type(image_type)),
                    None,
                )
                .unwrap()
        };
        Self {
            device,
            allocator: None,
            handle: image,
            view,
            allocation: None,
            layout,
            format,
        }
    }

    pub fn get_transition_barrier<'a>(
        &mut self,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        new_layout: vk::ImageLayout,
    ) -> vk::ImageMemoryBarrier2<'a> {
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(self.handle)
            .subresource_range(full_subresource_range(format_to_aspect(self.format)))
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(self.layout)
            .new_layout(new_layout);
        self.layout = new_layout;
        barrier
    }

    pub fn immediate_transition(
        &mut self,
        command_buffer: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        new_layout: vk::ImageLayout,
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .image(self.handle)
                        .subresource_range(full_subresource_range(format_to_aspect(self.format)))
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(self.layout)
                        .new_layout(new_layout),
                ]),
            );
        };

        self.layout = new_layout;
    }

    pub fn immediate_transition_from_undefined(
        &mut self,
        command_buffer: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        new_layout: vk::ImageLayout,
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .image(self.handle)
                        .subresource_range(full_subresource_range(format_to_aspect(self.format)))
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(dst_stage)
                        .dst_access_mask(dst_access)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(new_layout),
                ]),
            );
        };
        self.layout = new_layout;
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
        }

        if let (Some(allocator), Some(mut allocation)) =
            (self.allocator.as_ref(), self.allocation.take())
        {
            unsafe {
                allocator.destroy_image(self.handle, &mut allocation);
            };
        }
    }
}
