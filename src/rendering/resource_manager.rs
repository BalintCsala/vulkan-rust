use std::sync::Arc;

use ash::{ext::debug_utils, vk};
use bevy::{ecs::resource::Resource, math::Mat4};

use crate::{
    assets::model::ModelData,
    rendering::{
        buffer::Buffer,
        command_cache::CommandCache,
        image::Image,
        vulkan_utils::format_to_aspect,
        wrappers::{allocator::Allocator, device::Device},
    },
};

const SAMPLED_IMAGE_BINDING: u32 = 0;
const STORAGE_IMAGE_BINDING: u32 = 1;
const SAMPLER_BINDING: u32 = 2;

const IMAGE_COUNT: u32 = 65536;
const SAMPLER_COUNT: u32 = 65536;

const MAX_MODEL_DATA_COUNT: usize = 16384;

const STAGING_BUFFER_SIZE: usize = 0x8000000; // 128MB

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

pub type ImageReference = i16;
pub type SamplerReference = u8;
pub type ModelReference = u16;
pub type InstanceReference = u16;

#[repr(C)]
struct InstanceData {
    model: [f32; 16],
    normal: [f32; 16],
    model_id: u32,
}

pub struct IndexData {
    pub index_buffer: Buffer,
    pub index_count: u32,
}

pub struct ImageInfo {
    size: ImageSize,
    usage: vk::ImageUsageFlags,
    format: vk::Format,
    mip_levels: u32,
    array_layers: u32,
    name: String,
    image: Image,
}

#[derive(Resource)]
pub struct ResourceManager {
    device: Arc<Device>,
    queue: vk::Queue,
    allocator: Arc<Allocator>,
    debug_utils_device: debug_utils::Device,
    command_cache: CommandCache,
    extent: vk::Extent2D,

    descriptor_pool: vk::DescriptorPool,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,

    images: Vec<ImageInfo>,
    samplers: Vec<vk::Sampler>,

    // TODO: Better suballocation strategy
    pub model_buffer: Buffer,
    pub index_data: Vec<IndexData>,
    next_model_ref: ModelReference,

    pub instance_buffer: Buffer,
    next_instance_ref: InstanceReference,

    staging_buffer: Buffer,
    staging_buffer_offset: usize,
    staging_fences: Vec<vk::Fence>,
}

impl ResourceManager {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Allocator>,
        debug_utils_device: debug_utils::Device,
        queue: vk::Queue,
        extent: vk::Extent2D,
    ) -> Self {
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
                                .ty(vk::DescriptorType::STORAGE_IMAGE)
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
                                .binding(SAMPLED_IMAGE_BINDING)
                                .descriptor_count(IMAGE_COUNT)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .stage_flags(vk::ShaderStageFlags::ALL),
                            vk::DescriptorSetLayoutBinding::default()
                                .binding(STORAGE_IMAGE_BINDING)
                                .descriptor_count(IMAGE_COUNT)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
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

        let model_buffer = Buffer::new(
            &device,
            allocator.clone(),
            &debug_utils_device,
            vk::BufferUsageFlags::empty(),
            (MAX_MODEL_DATA_COUNT * size_of::<ModelData>()) as u64,
            "Model data buffer",
        );

        let instance_buffer = Buffer::new(
            &device,
            allocator.clone(),
            &debug_utils_device,
            vk::BufferUsageFlags::empty(),
            (MAX_MODEL_DATA_COUNT * size_of::<ModelData>()) as u64,
            "Instance buffer",
        );

        let staging_buffer = Buffer::new(
            &device,
            allocator.clone(),
            &debug_utils_device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            STAGING_BUFFER_SIZE as u64,
            "Staging buffer",
        );

        let command_cache = CommandCache::new(device.clone());

        Self {
            device,
            allocator,
            debug_utils_device,
            queue,
            extent,
            images: Vec::new(),
            samplers: Vec::new(),

            model_buffer,
            index_data: Vec::new(),
            next_model_ref: 0,

            command_cache,
            staging_buffer,
            staging_buffer_offset: 0,
            staging_fences: Vec::new(),
            descriptor_pool,
            descriptor_layout,
            descriptor_set,
            instance_buffer,
            next_instance_ref: 0,
        }
    }

    fn write_image_to_descriptor(
        &self,
        usage: vk::ImageUsageFlags,
        reference: &ImageReference,
        image_info: &[vk::DescriptorImageInfo],
    ) {
        let writes: Vec<_> = [
            (
                vk::ImageUsageFlags::SAMPLED,
                vk::DescriptorType::SAMPLED_IMAGE,
                SAMPLED_IMAGE_BINDING,
            ),
            (
                vk::ImageUsageFlags::STORAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                STORAGE_IMAGE_BINDING,
            ),
        ]
        .iter()
        .filter_map(|&(descriptor_usage, descriptor_type, binding)| {
            if !usage.contains(descriptor_usage) {
                None
            } else {
                Some(
                    vk::WriteDescriptorSet::default()
                        .descriptor_count(1)
                        .descriptor_type(descriptor_type)
                        .dst_binding(binding)
                        .dst_array_element(*reference as u32)
                        .image_info(image_info)
                        .dst_set(self.descriptor_set),
                )
            }
        })
        .collect::<Vec<_>>();

        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            };
        }
    }

    pub fn register_raw_image(
        &mut self,
        image: Image,
        usage: vk::ImageUsageFlags,
        name: String,
    ) -> ImageReference {
        let reference = self.images.len() as i16;

        let descriptor_image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(image.view)];

        self.write_image_to_descriptor(usage, &reference, &descriptor_image_info);

        self.images.push(ImageInfo {
            size: ImageSize::Fixed(1, 1),
            usage,
            format: image.format,
            mip_levels: 1,
            array_layers: 1,
            image,
            name,
        });

        reference
    }

    pub fn update_raw_image(&mut self, image: Image, reference: &ImageReference) -> Image {
        let descriptor_image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(image.view)];

        self.write_image_to_descriptor(
            self.images[(*reference) as usize].usage,
            reference,
            &descriptor_image_info,
        );

        std::mem::replace(&mut self.images[(*reference) as usize].image, image)
    }

    pub fn create_empty_image(
        &mut self,
        size: ImageSize,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        array_layers: u32,
        name: String,
    ) -> ImageReference {
        let reference = self.images.len() as i16;

        let (extent, image_type) = size.evaluate(self.extent.width, self.extent.height);
        let image = Image::new(
            self.device.clone(),
            self.allocator.clone(),
            &self.debug_utils_device,
            extent,
            format,
            usage,
            image_type,
            mip_levels,
            array_layers,
            &name,
        );

        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(image.view)];

        self.write_image_to_descriptor(usage, &reference, &image_info);

        self.images.push(ImageInfo {
            size,
            usage,
            format,
            mip_levels,
            array_layers,
            image,
            name,
        });

        reference
    }

    fn dispatch_copy_from_staging(
        &mut self,
        command_buffer: vk::CommandBuffer,
        barriers: &[vk::ImageMemoryBarrier2],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default().image_memory_barriers(barriers),
            );
        };
        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap()
        };
        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();
        };
        unsafe {
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[command_buffer])],
                    fence,
                )
                .unwrap();
        };
        self.staging_fences.push(fence);
    }

    pub fn flush_staging(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&self.staging_fences, true, u64::MAX)
                .unwrap();
        };
        self.staging_fences.drain(..).for_each(|fence| {
            unsafe {
                self.device.destroy_fence(fence, None);
            };
        });

        self.staging_buffer_offset = 0;
    }

    pub fn upload_image_data<T>(&mut self, image_data: &[(ImageReference, &[T])]) {
        let mut command_buffer = self.command_cache.get_command_buffer();

        let initial_layout_transitions = image_data
            .iter()
            .map(|(reference, _)| {
                let image_info = &mut self.images[*reference as usize];
                image_info.image.get_transition_barrier(
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                )
            })
            .collect::<Vec<_>>();
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(initial_layout_transitions.as_slice()),
            );
        };

        let mut pending_barriers = Vec::new();
        for (reference, data) in image_data {
            let required_space = std::mem::size_of_val(*data);

            if STAGING_BUFFER_SIZE < required_space {
                panic!(
                    "Not enough space in staging buffer, required: {}, actual: {}",
                    required_space, STAGING_BUFFER_SIZE
                );
            }

            if STAGING_BUFFER_SIZE - self.staging_buffer_offset < required_space {
                if !pending_barriers.is_empty() {
                    self.dispatch_copy_from_staging(command_buffer, &pending_barriers);
                    println!("Staged {} images in one upload", pending_barriers.len());
                    pending_barriers.clear();
                    command_buffer = self.command_cache.get_command_buffer();
                }
                self.flush_staging();
            }

            self.staging_buffer.write(data, self.staging_buffer_offset);
            let image_info = &mut self.images[*reference as usize];

            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    self.staging_buffer.handle,
                    image_info.image.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(self.staging_buffer_offset as u64)
                        .image_extent(
                            image_info
                                .size
                                .evaluate(self.extent.width, self.extent.height)
                                .0,
                        )
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .base_array_layer(0)
                                .layer_count(1)
                                .mip_level(0)
                                .aspect_mask(format_to_aspect(image_info.format)),
                        )],
                );
            };

            pending_barriers.push(image_info.image.get_transition_barrier(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::ImageLayout::READ_ONLY_OPTIMAL,
            ));

            self.staging_buffer_offset += required_space;
            self.staging_buffer_offset = self.staging_buffer_offset.div_ceil(16) * 16;
        }

        self.dispatch_copy_from_staging(command_buffer, pending_barriers.as_slice());
        println!("Staged {} images in one upload", pending_barriers.len());
    }

    pub fn upload_model(
        &mut self,
        model: ModelData,
        index_buffer: Buffer,
        index_count: u32,
    ) -> ModelReference {
        let reference = self.next_model_ref;
        self.next_model_ref += 1;

        self.model_buffer
            .write(&[model], size_of::<ModelData>() * reference as usize);

        self.index_data.push(IndexData {
            index_buffer,
            index_count,
        });

        reference
    }

    pub fn create_instance(
        &mut self,
        model_matrix: &Mat4,
        model_ref: &ModelReference,
    ) -> InstanceReference {
        let reference = self.next_instance_ref;
        self.next_instance_ref += 1;

        let mut instance_data = InstanceData {
            model: [0.0; 16],
            normal: [0.0; 16],
            model_id: (*model_ref) as u32,
        };
        model_matrix.write_cols_to_slice(&mut instance_data.model);
        model_matrix
            .transpose()
            .inverse()
            .write_cols_to_slice(&mut instance_data.normal);

        self.instance_buffer.write(
            &[instance_data],
            size_of::<InstanceData>() * reference as usize,
        );

        reference
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (image_indices, image_infos): (Vec<_>, Vec<_>) = self
            .images
            .iter_mut()
            .enumerate()
            .filter_map(|(reference, image_info)| {
                // TODO: Make image resize itself
                if matches!(image_info.size, ImageSize::Fixed(_, _))
                    || matches!(image_info.size, ImageSize::Fixed3D(_, _, _))
                {
                    return None;
                }

                let (extent, image_type) = image_info.size.evaluate(width, height);
                image_info.image = Image::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    &self.debug_utils_device,
                    extent,
                    image_info.format,
                    image_info.usage,
                    image_type,
                    image_info.mip_levels,
                    image_info.array_layers,
                    &image_info.name,
                );
                Some((
                    reference,
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(image_info.image.view),
                ))
            })
            .unzip();

        let writes: Vec<_> = [
            (
                vk::ImageUsageFlags::SAMPLED,
                vk::DescriptorType::SAMPLED_IMAGE,
                SAMPLED_IMAGE_BINDING,
            ),
            (
                vk::ImageUsageFlags::STORAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                STORAGE_IMAGE_BINDING,
            ),
        ]
        .iter()
        .flat_map(|&(usage, descriptor_type, binding)| {
            let indices: Vec<_> = image_indices
                .iter()
                .enumerate()
                .filter_map(|(i, &reference)| {
                    if self.images[reference].usage.contains(usage) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();

            if indices.is_empty() {
                return Vec::new();
            }
            let mut last_end = 0;
            let mut writes = Vec::new();
            for i in 1..=indices.len() {
                if i != indices.len()
                    && image_indices[indices[i]] == image_indices[indices[i - 1]] + 1
                {
                    continue;
                }
                let dst_index = image_indices[indices[last_end]];
                let info_start = indices[last_end];
                let info_end = *indices.get(i).unwrap_or(&image_infos.len());
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .descriptor_count(1)
                        .descriptor_type(descriptor_type)
                        .dst_binding(binding)
                        .dst_array_element(dst_index as u32)
                        .image_info(&image_infos[info_start..info_end])
                        .dst_set(self.descriptor_set),
                );
                last_end = i;
            }
            writes
        })
        .collect();

        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        };
    }

    pub fn add_sampler(&mut self, sampler: vk::Sampler) -> SamplerReference {
        let reference = self.samplers.len() as SamplerReference;

        unsafe {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .dst_binding(SAMPLER_BINDING)
                    .dst_array_element(reference as u32)
                    .image_info(&[vk::DescriptorImageInfo::default().sampler(sampler)])
                    .dst_set(self.descriptor_set)],
                &[],
            );
        };
        self.samplers.push(sampler);

        reference
    }

    pub fn get_index_data(&self, model_ref: ModelReference) -> &IndexData {
        &self.index_data[model_ref as usize]
    }

    pub fn get_image(&self, reference: &ImageReference) -> &Image {
        &self.images[*reference as usize].image
    }

    pub fn get_image_mut(&mut self, reference: &ImageReference) -> &mut Image {
        &mut self.images[*reference as usize].image
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        println!("Dropped resource manager");
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
