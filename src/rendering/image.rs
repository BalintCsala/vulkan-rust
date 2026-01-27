use std::sync::Arc;

use ash::vk;
use vk_mem::{Alloc, Allocation, AllocationCreateInfo, Allocator};

use crate::rendering::vulkan_utils::{
    format_to_aspect, full_subresource_range, image_type_to_view_type,
};

pub struct Image {
    device: ash::Device,
    allocator: Option<Arc<Allocator>>,
    pub handle: vk::Image,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
    layout: vk::ImageLayout,
    format: vk::Format,
}

impl Image {
    pub fn new(
        device: ash::Device,
        allocator: Arc<Allocator>,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        image_type: vk::ImageType,
        mip_levels: u32,
        array_layers: u32,
    ) -> Self {
        let (image, allocation) = unsafe {
            allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .extent(extent)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .mip_levels(mip_levels)
                        .image_type(image_type)
                        .format(format)
                        .usage(usage)
                        .array_layers(array_layers)
                        .samples(vk::SampleCountFlags::TYPE_1),
                    &AllocationCreateInfo::default(),
                )
                .unwrap()
        };

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
            allocator: Some(allocator),
            handle: image,
            allocation: Some(allocation),
            view,
            layout: vk::ImageLayout::UNDEFINED,
            format,
        }
    }

    pub fn from_raw(
        device: ash::Device,
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

    fn transition_impl(
        &mut self,
        command_buffer: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        old_layout: vk::ImageLayout,
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
                        .old_layout(old_layout)
                        .new_layout(new_layout),
                ]),
            );
        };
        self.layout = new_layout;
    }

    pub fn transition(
        &mut self,
        command_buffer: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        new_layout: vk::ImageLayout,
    ) {
        self.transition_impl(
            command_buffer,
            src_stage,
            src_access,
            dst_stage,
            dst_access,
            self.layout,
            new_layout,
        );
    }

    pub fn transition_from_undefined(
        &mut self,
        command_buffer: vk::CommandBuffer,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
        new_layout: vk::ImageLayout,
    ) {
        self.transition_impl(
            command_buffer,
            src_stage,
            src_access,
            dst_stage,
            dst_access,
            vk::ImageLayout::UNDEFINED,
            new_layout,
        );
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.view, None);
        }

        if let (Some(allocator), Some(allocation)) =
            (self.allocator.as_ref(), self.allocation.as_mut())
        {
            unsafe {
                allocator.destroy_image(self.handle, allocation);
            };
        }
    }
}
