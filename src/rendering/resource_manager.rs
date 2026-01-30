use std::sync::Arc;

use ash::vk;

use crate::rendering::{
    image::Image,
    wrappers::{allocator::Allocator, device::Device},
};

const IMAGE_BINDING: u32 = 0;
const SAMPLER_BINDING: u32 = 1;

const IMAGE_COUNT: u32 = 65536;
const SAMPLER_COUNT: u32 = 65536;

pub enum ImageSize {
    Fixed(u32, u32),
    Fixed3D(u32, u32, u32),
    Scaled(f32, f32),
    Dynamic(fn(u32, u32) -> (u32, u32)),
    Dynamic3D(fn(u32, u32) -> (u32, u32, u32)),
}

impl ImageSize {
    fn evaluate(&self, width: u32, height: u32) -> (vk::Extent3D, vk::ImageType) {
        match self {
            ImageSize::Fixed(width, height) => (
                vk::Extent3D::default()
                    .width(*width)
                    .height(*height)
                    .depth(1),
                vk::ImageType::TYPE_2D,
            ),
            ImageSize::Fixed3D(width, height, depth) => (
                vk::Extent3D::default()
                    .width(*width)
                    .height(*height)
                    .depth(*depth),
                vk::ImageType::TYPE_3D,
            ),
            ImageSize::Scaled(x_scale, y_scale) => (
                vk::Extent3D::default()
                    .width(((width as f32) * x_scale).ceil() as u32)
                    .height(((height as f32) * y_scale).ceil() as u32)
                    .depth(1),
                vk::ImageType::TYPE_2D,
            ),
            ImageSize::Dynamic(callback) => {
                let (width, height) = callback(width, height);
                (
                    vk::Extent3D::default().width(width).height(height).depth(1),
                    vk::ImageType::TYPE_2D,
                )
            }
            ImageSize::Dynamic3D(callback) => {
                let (width, height, depth) = callback(width, height);
                (
                    vk::Extent3D::default()
                        .width(width)
                        .height(height)
                        .depth(depth),
                    vk::ImageType::TYPE_3D,
                )
            }
        }
    }
}

pub struct ImageReference(i32);

pub struct ResourceManager {
    device: Arc<Device>,
    allocator: Arc<Allocator>,
    extent: vk::Extent2D,
    images: Vec<(ImageSize, vk::ImageUsageFlags, vk::Format, u32, u32, Image)>,
    descriptor_pool: vk::DescriptorPool,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
}

impl ResourceManager {
    pub fn new(device: Arc<Device>, allocator: Arc<Allocator>, extent: vk::Extent2D) -> Self {
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                                .descriptor_count(IMAGE_COUNT),
                            vk::DescriptorPoolSize::default()
                                .ty(vk::DescriptorType::SAMPLER)
                                .descriptor_count(SAMPLER_COUNT),
                        ])
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap()
        };

        let descriptor_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&[
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(IMAGE_BINDING)
                                .descriptor_count(IMAGE_COUNT)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(SAMPLER_BINDING)
                                .descriptor_count(SAMPLER_COUNT)
                                .descriptor_type(vk::DescriptorType::SAMPLER)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                        ])
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(&[
                                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                                ]),
                        ),
                    None,
                )
                .unwrap()
        };
        let descriptor_set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .set_layouts(&[descriptor_layout])
                        .descriptor_pool(descriptor_pool),
                )
                .unwrap()[0]
        };
        Self {
            device,
            allocator,
            extent,
            images: Vec::new(),
            descriptor_pool,
            descriptor_layout,
            descriptor_set,
        }
    }

    pub fn create_image(
        &mut self,
        size: ImageSize,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        array_layers: u32,
    ) -> ImageReference {
        let reference = self.images.len() as i32;

        let (extent, image_type) = size.evaluate(self.extent.width, self.extent.height);
        let image = Image::new(
            self.device.clone(),
            self.allocator.clone(),
            extent,
            format,
            usage | vk::ImageUsageFlags::SAMPLED,
            image_type,
            mip_levels,
            array_layers,
        );

        unsafe {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .dst_binding(reference as u32)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(image.view)])
                    .dst_set(self.descriptor_set)],
                &[],
            );
        };

        self.images
            .push((size, usage, format, mip_levels, array_layers, image));

        ImageReference(reference)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        for i in 0..self.images.len() {
            // TODO: Make image resize itself
            let (size, usage, format, mip_levels, array_layers, _) = &self.images[i];
            if matches!(size, ImageSize::Fixed(_, _)) || matches!(size, ImageSize::Fixed3D(_, _, _))
            {
                continue;
            }

            let (extent, image_type) = size.evaluate(width, height);
            self.images[i].5 = Image::new(
                self.device.clone(),
                self.allocator.clone(),
                extent,
                *format,
                *usage,
                image_type,
                *mip_levels,
                *array_layers,
            );
        }
    }

    pub fn get_image(&self, reference: &ImageReference) -> &Image {
        &self.images[reference.0 as usize].5
    }

    pub fn get_image_mut(&mut self, reference: &ImageReference) -> &mut Image {
        &mut self.images[reference.0 as usize].5
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        };
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        };
        self.images.clear();
    }
}
