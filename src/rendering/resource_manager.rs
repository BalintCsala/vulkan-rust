use std::{collections::HashMap, mem::offset_of, sync::Arc};

use ash::vk;
use bevy::{
    ecs::{entity::EntityIndex, resource::Resource},
    math::{Mat3, Mat4, Vec3},
};

use crate::{
    assets::model::{GpuModel, ModelRenderInfo},
    rendering::{
        command_cache::CommandCache,
        generated_pipelines::{MipmapPipelinePushConstants, create_mipmap_pipeline},
    },
};
use vulkan_utils::{
    complex_types::{buffer::Buffer, image::Image},
    pipeline_generator::pipeline_types::{ComputePipeline, Pipeline},
    utility_functions::{format_to_aspect, mip_level_subresource_range},
    wrappers::{
        allocator::Allocator, device::Device, fence::Fence, instance::Instance, sampler::Sampler,
    },
};

const SAMPLED_IMAGE_BINDING: u32 = 0;
const STORAGE_IMAGE_BINDING: u32 = 1;
const SAMPLER_BINDING: u32 = 2;

const IMAGE_COUNT: u32 = 65536;
const SAMPLER_COUNT: u32 = 65536;

const MAX_MODEL_DATA_COUNT: usize = 16384;

const STAGING_BUFFER_SIZE: usize = 0x8000000; // 128MB
const RT_INSTANCE_BUFFER_SIZE: u64 = 0x10000000; // 256MB

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
struct GpuInstance {
    model: [f32; 16],
    normal: [f32; 9],
    model_id: u32,
}

impl GpuInstance {
    pub fn new(model: Mat4, model_id: ModelReference) -> Self {
        let mut res = Self {
            model: [0.0; 16],
            normal: [0.0; 9],
            model_id: model_id.into(),
        };
        model.write_cols_to_slice(&mut res.model);
        Mat3::from_mat4(model)
            .inverse()
            .transpose()
            .write_cols_to_slice(&mut res.normal);
        res
    }
}

struct InstanceIndices {
    instance_id: InstanceReference,
    rt_instance_id: Option<InstanceReference>,
}

pub struct IndexData {
    pub index_buffer: Buffer,
    pub index_count: u32,
}

pub struct ImageInfo {
    size: ImageSize,
    usage: vk::ImageUsageFlags,
    array_layers: u32,
    name: String,
    image: Image,
}

struct PendingBlasBuild {
    position_device_address: vk::DeviceAddress,
    index_device_address: vk::DeviceAddress,
    position_count: u32,
    index_count: u32,
    reference: ModelReference,
    render_info: ModelRenderInfo,
}

struct RTInstance {
    transform: vk::TransformMatrixKHR,
    model_reference: ModelReference,
}

struct Limits {
    min_acceleration_structure_scratch_offset_alignment: u64,
}

#[derive(Resource)]
pub struct ResourceManager {
    device: Arc<Device>,
    allocator: Arc<Allocator>,

    command_cache: CommandCache,
    extent: vk::Extent2D,

    pub bindless_pipeline_layout: vk::PipelineLayout,

    descriptor_pool: vk::DescriptorPool,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,

    images: HashMap<ImageReference, ImageInfo>,
    images_by_name: HashMap<String, ImageReference>,
    next_image_reference: ImageReference,
    samplers: Vec<Sampler>,

    // TODO: Better suballocation strategy
    pub model_buffer: Buffer,
    pub index_data: Vec<IndexData>,
    next_model_ref: ModelReference,
    pub model_blases: Vec<vk::AccelerationStructureKHR>,
    pending_blas_builds: Vec<PendingBlasBuild>,
    pub tlas: vk::AccelerationStructureKHR,
    acceleration_structure_buffers: Vec<Buffer>,
    fence: Fence,

    pub instance_buffer: Buffer,
    rt_instance_buffer: Buffer,
    rt_instance_count: usize,
    instances: HashMap<EntityIndex, InstanceIndices>,

    staging_buffer: Buffer,

    mipmap_pipeline: ComputePipeline,

    limits: Limits,
}

impl ResourceManager {
    pub fn new(
        instance: &Arc<Instance>,
        device: Arc<Device>,
        physical_device: &vk::PhysicalDevice,
        allocator: Arc<Allocator>,
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
            vk::BufferUsageFlags::empty(),
            (MAX_MODEL_DATA_COUNT * size_of::<GpuModel>()) as u64,
            "Model data buffer",
            None,
        );

        let instance_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::empty(),
            (MAX_MODEL_DATA_COUNT * size_of::<GpuModel>()) as u64,
            "Instance buffer",
            None,
        );

        let staging_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            STAGING_BUFFER_SIZE as u64,
            "Staging buffer",
            None,
        );

        let command_cache = CommandCache::new(device.clone(), queue);

        let bindless_pipeline_layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .offset(0)
                            .size(256)])
                        .set_layouts(&[descriptor_layout]),
                    None,
                )
                .unwrap()
        };

        let mipmap_pipeline = create_mipmap_pipeline(device.clone(), bindless_pipeline_layout);

        let rt_instance_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            RT_INSTANCE_BUFFER_SIZE,
            "RT instance buffer",
            None,
        );

        let fence = Fence::new(device.clone(), &vk::FenceCreateInfo::default());

        let mut acceleration_structure_props =
            vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut props =
            vk::PhysicalDeviceProperties2::default().push_next(&mut acceleration_structure_props);

        unsafe { instance.get_physical_device_properties2(*physical_device, &mut props) };

        Self {
            device,
            allocator,

            extent,

            bindless_pipeline_layout,

            images: HashMap::new(),
            images_by_name: HashMap::new(),
            next_image_reference: 0,
            samplers: Vec::new(),

            model_buffer,
            index_data: Vec::new(),
            next_model_ref: 0,

            model_blases: Vec::new(),
            pending_blas_builds: Vec::new(),
            acceleration_structure_buffers: Vec::new(),
            tlas: vk::AccelerationStructureKHR::null(),

            rt_instance_buffer,
            rt_instance_count: 0,
            instance_buffer,
            instances: HashMap::new(),

            fence,

            command_cache,
            staging_buffer,
            descriptor_pool,
            descriptor_layout,
            descriptor_set,
            mipmap_pipeline,

            limits: Limits {
                min_acceleration_structure_scratch_offset_alignment: u64::from(
                    acceleration_structure_props
                        .min_acceleration_structure_scratch_offset_alignment,
                ),
            },
        }
    }

    fn write_images_to_descriptor(
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

    pub fn create_empty_image(
        &mut self,
        size: ImageSize,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        array_layers: u32,
        name: String,
    ) -> ImageReference {
        let reference = self.next_image_reference;
        self.next_image_reference += mip_levels as i16;

        let (extent, image_type) = size.evaluate(self.extent.width, self.extent.height);
        let mut image = Image::new(
            self.device.clone(),
            self.allocator.clone(),
            extent,
            format,
            usage,
            image_type,
            mip_levels,
            array_layers,
            &name,
        );

        self.command_cache
            .run_command(vk::Fence::null(), |&command_buffer| unsafe {
                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo::default().image_memory_barriers(&[image
                        .get_transition_barrier(
                            vk::PipelineStageFlags2::ALL_COMMANDS,
                            vk::AccessFlags2::NONE,
                            vk::PipelineStageFlags2::NONE,
                            vk::AccessFlags2::NONE,
                            vk::ImageLayout::GENERAL,
                        )]),
                );
            });

        let image_info: Vec<_> = (0..mip_levels)
            .map(|level| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(image.get_mip_view(level as usize))
            })
            .collect();

        self.write_images_to_descriptor(usage, &reference, &image_info);

        self.images.insert(
            reference,
            ImageInfo {
                size,
                usage,
                array_layers,
                image,
                name: name.clone(),
            },
        );
        self.images_by_name.insert(name, reference);

        reference
    }

    pub fn get_or_create_image<T>(
        &mut self,
        size: ImageSize,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        mip_levels: u32,
        array_layers: u32,
        name: String,
        fallback_contents: &[T],
    ) -> ImageReference {
        match self.get_image_reference_by_name(&name) {
            Some(image_ref) => image_ref,
            None => {
                let image_ref =
                    self.create_empty_image(size, format, usage, mip_levels, array_layers, name);
                self.upload_image_data(&mut vec![(image_ref, fallback_contents)]);
                image_ref
            }
        }
    }

    pub fn upload_image_data<T>(&mut self, image_data: &mut Vec<(ImageReference, &[T])>) {
        let mut mipmapped_image_references = Vec::new();

        while !image_data.is_empty() {
            self.fence.reset();

            self.command_cache
                .run_command(*self.fence, |&command_buffer| {
                    let mut staging_buffer_offset = 0;
                    while let Some((reference, data)) = image_data.pop() {
                        let required_space = std::mem::size_of_val(data);

                        if STAGING_BUFFER_SIZE < required_space {
                            panic!(
                                "Not enough space in staging buffer, required: {}, actual: {}",
                                required_space, STAGING_BUFFER_SIZE
                            );
                        }

                        if STAGING_BUFFER_SIZE - staging_buffer_offset < required_space {
                            image_data.push((reference, data));
                            return;
                        }

                        self.staging_buffer.write(data, staging_buffer_offset);

                        let image_info = &mut self.images.get_mut(&reference).unwrap();
                        unsafe {
                            self.device.cmd_copy_buffer_to_image(
                                command_buffer,
                                self.staging_buffer.handle,
                                image_info.image.handle,
                                vk::ImageLayout::GENERAL,
                                &[vk::BufferImageCopy::default()
                                    .buffer_offset(staging_buffer_offset as u64)
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
                                            .aspect_mask(format_to_aspect(image_info.image.format)),
                                    )],
                            );
                        };

                        staging_buffer_offset += required_space;
                        staging_buffer_offset = staging_buffer_offset.next_multiple_of(16);

                        if image_info.image.get_mip_count() > 1 {
                            mipmapped_image_references.push(reference);
                        }
                    }
                });

            self.fence.wait();
        }

        if !mipmapped_image_references.is_empty() {
            self.command_cache.run_command(vk::Fence::null(), |&command_buffer| {
            self.mipmap_pipeline.bind(command_buffer);
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.bindless_pipeline_layout,
                    0,
                    &[self.descriptor_set],
                    &[],
                );
            };

            for reference in mipmapped_image_references {
                let info = &mut self.images.get_mut(&reference).unwrap();
                let (extent, _) = info.size.evaluate(self.extent.width, self.extent.height);

                if info.usage.contains(vk::ImageUsageFlags::STORAGE) {
                    unsafe {
                        self.device.cmd_push_constants(
                            command_buffer,
                            self.bindless_pipeline_layout,
                            vk::ShaderStageFlags::ALL,
                            0,
                            bytemuck::bytes_of(&MipmapPipelinePushConstants {
                                base_image_id: reference as u32,
                                num_of_mips: info.image.get_mip_count(),
                            }),
                        );
                    };

                    unsafe {
                        self.device.cmd_dispatch(
                            command_buffer,
                            extent.width.div_ceil(32),
                            extent.height.div_ceil(32),
                            1,
                        );
                    };
                } else if info
                    .usage
                    .contains(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
                {
                    let mut src_mip_width = extent.width;
                    let mut src_mip_height = extent.height;
                    for mip_level in 0..info.image.get_mip_count() - 1 {
                        unsafe {
                            self.device.cmd_pipeline_barrier2(
                                command_buffer,
                                &vk::DependencyInfo::default().image_memory_barriers(&[
                                    vk::ImageMemoryBarrier2::default()
                                        .image(info.image.handle)
                                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                                        .old_layout(vk::ImageLayout::GENERAL)
                                        .new_layout(vk::ImageLayout::GENERAL)
                                        .subresource_range(mip_level_subresource_range(
                                            format_to_aspect(info.image.format),
                                            mip_level,
                                            1,
                                        )),
                                ]),
                            );
                        };

                        unsafe {
                            self.device.cmd_blit_image2(
                                command_buffer,
                                &vk::BlitImageInfo2::default()
                                    .src_image(info.image.handle)
                                    .src_image_layout(vk::ImageLayout::GENERAL)
                                    .dst_image(info.image.handle)
                                    .dst_image_layout(vk::ImageLayout::GENERAL)
                                    .regions(&[vk::ImageBlit2::default()
                                        .src_subresource(
                                            vk::ImageSubresourceLayers::default()
                                                .base_array_layer(0)
                                                .layer_count(info.array_layers)
                                                .aspect_mask(format_to_aspect(info.image.format))
                                                .mip_level(mip_level),
                                        )
                                        .src_offsets([
                                            vk::Offset3D { x: 0, y: 0, z: 0 },
                                            vk::Offset3D {
                                                x: src_mip_width as i32,
                                                y: src_mip_height as i32,
                                                z: 1,
                                            },
                                        ])
                                        .dst_subresource(
                                            vk::ImageSubresourceLayers::default()
                                                .base_array_layer(0)
                                                .layer_count(info.array_layers)
                                                .aspect_mask(format_to_aspect(info.image.format))
                                                .mip_level(mip_level + 1),
                                        )
                                        .dst_offsets([
                                            vk::Offset3D { x: 0, y: 0, z: 0 },
                                            vk::Offset3D {
                                                x: (src_mip_width / 2).max(1) as i32,
                                                y: (src_mip_height / 2).max(1) as i32,
                                                z: 1,
                                            },
                                        ])])
                                    .filter(vk::Filter::LINEAR),
                            );
                        }

                        src_mip_width /= 2;
                        src_mip_height /= 2;
                    }

                    unsafe {
                        self.device.cmd_pipeline_barrier2(
                            command_buffer,
                            &vk::DependencyInfo::default().memory_barriers(&[
                                vk::MemoryBarrier2::default()
                                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                            ]),
                        );
                    };
                } else {
                    panic!(
                        "Can't generate mipmaps without STORAGE or TRANSFER_DST | TRANSFER_SRC usages"
                    );
                }
            }
            });
        }
    }

    pub fn upload_model(
        &mut self,
        model: GpuModel,
        index_buffer: Buffer,
        index_count: u32,
        position_count: u32,
        render_info: ModelRenderInfo,
    ) -> ModelReference {
        let reference = self.next_model_ref;
        self.next_model_ref += 1;

        self.pending_blas_builds.push(PendingBlasBuild {
            position_device_address: model.positions,
            index_device_address: model.indices,
            position_count,
            index_count,
            reference,
            render_info,
        });

        self.model_buffer
            .write(&[model], size_of::<GpuModel>() * reference as usize);

        self.index_data.push(IndexData {
            index_buffer,
            index_count,
        });

        self.model_blases.push(vk::AccelerationStructureKHR::null());

        reference
    }

    fn flush_pending_blases(&mut self) {
        if self.pending_blas_builds.is_empty() {
            return;
        }

        let mut build_infos = Vec::new();
        let mut build_range_infos = Vec::new();

        let geometries: Vec<_> = self
            .pending_blas_builds
            .iter()
            .map(|build_data| {
                [vk::AccelerationStructureGeometryKHR::default()
                    .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                    .geometry(vk::AccelerationStructureGeometryDataKHR {
                        triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                            .vertex_format(vk::Format::R32G32B32_SFLOAT)
                            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: build_data.position_device_address,
                            })
                            .vertex_stride(size_of::<Vec3>() as u64)
                            .max_vertex(build_data.position_count - 1)
                            .index_data(vk::DeviceOrHostAddressConstKHR {
                                device_address: build_data.index_device_address,
                            })
                            .index_type(vk::IndexType::UINT32),
                    })
                    .flags(if build_data.render_info.opaque {
                        vk::GeometryFlagsKHR::OPAQUE
                    } else {
                        vk::GeometryFlagsKHR::empty()
                    })]
            })
            .collect();

        let build_ranges: Vec<_> = self
            .pending_blas_builds
            .iter()
            .map(|build_data| {
                [vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(build_data.index_count / 3)]
            })
            .collect();

        let mut scratch_buffers = Vec::new();

        self.pending_blas_builds
            .iter()
            .zip(geometries.iter())
            .zip(build_ranges.iter())
            .for_each(|((build_data, geometries), build_range)| {
                let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .geometries(geometries)
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE);

                let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
                unsafe {
                    self.device
                        .acceleration_structure
                        .get_acceleration_structure_build_sizes(
                            vk::AccelerationStructureBuildTypeKHR::DEVICE,
                            &build_info,
                            &[build_data.index_count / 3],
                            &mut size_info,
                        );
                };

                let blas_buffer = Buffer::new(
                    &self.device,
                    self.allocator.clone(),
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                    size_info.acceleration_structure_size,
                    &format!("BLAS buffer #{}", self.acceleration_structure_buffers.len()),
                    None,
                );

                let scratch_buffer = Buffer::new(
                    &self.device,
                    self.allocator.clone(),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    size_info.build_scratch_size,
                    "BLAS scratch buffer",
                    Some(
                        self.limits
                            .min_acceleration_structure_scratch_offset_alignment,
                    ),
                );

                build_info = build_info.scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.address,
                });

                let blas = unsafe {
                    self.device
                        .acceleration_structure
                        .create_acceleration_structure(
                            &vk::AccelerationStructureCreateInfoKHR::default()
                                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                                .buffer(blas_buffer.handle)
                                .size(size_info.acceleration_structure_size),
                            None,
                        )
                        .unwrap()
                };
                build_info = build_info.dst_acceleration_structure(blas);

                self.model_blases[build_data.reference as usize] = blas;
                self.acceleration_structure_buffers.push(blas_buffer);
                scratch_buffers.push(scratch_buffer);
                build_infos.push(build_info);
                build_range_infos.push(build_range.as_slice());
            });

        self.command_cache
            .run_command(*self.fence, |&command_buffer| {
                unsafe {
                    self.device
                        .acceleration_structure
                        .cmd_build_acceleration_structures(
                            command_buffer,
                            &build_infos,
                            &build_range_infos,
                        )
                };
            });

        self.pending_blas_builds.clear();
    }

    pub fn build_acceleration_structures(&mut self) {
        self.flush_pending_blases();

        let tlas_geometries = [vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: self.rt_instance_buffer.address,
                    }),
            })];

        let mut tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .dst_acceleration_structure(self.tlas)
            .geometries(&tlas_geometries)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE);

        let mut tlas_build_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.device
                .acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &tlas_build_info,
                    &[self.rt_instance_count as u32],
                    &mut tlas_build_sizes,
                );
        };

        let tlas_buffer = Buffer::new(
            &self.device,
            self.allocator.clone(),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            tlas_build_sizes.acceleration_structure_size,
            "TLAS Buffer",
            None,
        );

        self.tlas = unsafe {
            self.device
                .acceleration_structure
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                        .buffer(tlas_buffer.handle)
                        .size(tlas_build_sizes.acceleration_structure_size),
                    None,
                )
                .unwrap()
        };
        tlas_build_info = tlas_build_info.dst_acceleration_structure(self.tlas);
        self.acceleration_structure_buffers.push(tlas_buffer);

        let tlas_scratch_buffer = Buffer::new(
            &self.device,
            self.allocator.clone(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
            tlas_build_sizes.build_scratch_size,
            "TLAS Scratch Buffer",
            Some(
                self.limits
                    .min_acceleration_structure_scratch_offset_alignment,
            ),
        );
        tlas_build_info = tlas_build_info.scratch_data(vk::DeviceOrHostAddressKHR {
            device_address: tlas_scratch_buffer.address,
        });

        let tlas_build_range_info = [vk::AccelerationStructureBuildRangeInfoKHR::default()
            .first_vertex(0)
            .transform_offset(0)
            .primitive_offset(0)
            .primitive_count(u32::try_from(self.rt_instance_count).unwrap())];

        self.fence.reset();
        self.command_cache
            .run_command(*self.fence, |&command_buffer| {
                unsafe {
                    self.device.cmd_pipeline_barrier2(
                        command_buffer,
                        &vk::DependencyInfo::default().memory_barriers(&[
                            vk::MemoryBarrier2::default()
                                .src_stage_mask(
                                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                                )
                                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                                .dst_stage_mask(
                                    vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
                                )
                                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR),
                        ]),
                    );
                }

                unsafe {
                    self.device
                        .acceleration_structure
                        .cmd_build_acceleration_structures(
                            command_buffer,
                            &[tlas_build_info],
                            &[&tlas_build_range_info],
                        );
                }
            });

        self.fence.wait();
    }

    pub fn create_instance(
        &mut self,
        entity_id: EntityIndex,
        model_matrix: &Mat4,
        model_ref: &ModelReference,
    ) {
        self.flush_pending_blases();

        let instance_id = self.instances.len();
        let rt_instance_id = self.rt_instance_count;
        self.rt_instance_count += 1;

        self.instance_buffer.write(
            &[GpuInstance::new(*model_matrix, *model_ref)],
            size_of::<GpuInstance>() * instance_id,
        );

        let mut rt_instance = [0.0f32; 12];
        for i in 0..3 {
            let start_index = i * 4;
            model_matrix
                .row(i)
                .write_to_slice(&mut rt_instance[start_index..start_index + 4]);
        }
        let blas_address = unsafe {
            self.device
                .acceleration_structure
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(self.model_blases[usize::from(*model_ref)]),
                )
        };
        let rt_instance_data = [vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: rt_instance,
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(rt_instance_id as u32, 0xFF),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::empty().as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_address,
            },
        }];
        self.rt_instance_buffer.write(
            &rt_instance_data,
            rt_instance_id * size_of::<vk::AccelerationStructureInstanceKHR>(),
        );

        self.instances.insert(
            entity_id,
            InstanceIndices {
                instance_id: InstanceReference::try_from(instance_id).unwrap(),
                rt_instance_id: Some(InstanceReference::try_from(rt_instance_id).unwrap()),
            },
        );
    }

    pub fn update_instance(&mut self, entity_id: &EntityIndex, model_matrix: &Mat4) {
        let instance_id = self.get_entity_instance_id(entity_id);

        let mut model = [0.0; 16];
        model_matrix.write_cols_to_slice(&mut model);

        let mut normal = [0.0; 9];
        Mat3::from_mat4(*model_matrix)
            .inverse()
            .transpose()
            .write_cols_to_slice(&mut normal);

        self.instance_buffer.write(
            &[model],
            size_of::<GpuInstance>() * usize::from(instance_id) + offset_of!(GpuInstance, model),
        );
        self.instance_buffer.write(
            &[normal],
            size_of::<GpuInstance>() * usize::from(instance_id) + offset_of!(GpuInstance, normal),
        );
        // TODO: RT instance
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (image_indices, image_infos): (Vec<_>, Vec<_>) = self
            .images
            .iter_mut()
            .filter_map(|(reference, image_info)| {
                if matches!(image_info.size, ImageSize::Fixed(_, _))
                    || matches!(image_info.size, ImageSize::Fixed3D(_, _, _))
                {
                    return None;
                }

                let (extent, image_type) = image_info.size.evaluate(width, height);
                image_info.image = Image::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    extent,
                    image_info.image.format,
                    image_info.usage,
                    image_type,
                    image_info.image.get_mip_count(),
                    image_info.array_layers,
                    &image_info.name,
                );
                Some((
                    *reference,
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
                    if self.images[&reference].usage.contains(usage) {
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

    pub fn add_sampler(&mut self, sampler: Sampler) -> SamplerReference {
        let reference = self.samplers.len() as SamplerReference;

        unsafe {
            self.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .dst_binding(SAMPLER_BINDING)
                    .dst_array_element(reference as u32)
                    .image_info(&[vk::DescriptorImageInfo::default().sampler(*sampler)])
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
        &self.images[reference].image
    }

    pub fn get_image_reference_by_name(&self, name: &str) -> Option<ImageReference> {
        self.images_by_name.get(name).copied()
    }

    pub fn get_entity_instance_id(&self, entity_id: &EntityIndex) -> InstanceReference {
        self.instances[entity_id].instance_id
    }
}

impl Drop for ResourceManager {
    fn drop(&mut self) {
        println!("Dropped resource manager");
        unsafe {
            self.device.device_wait_idle().unwrap();
        };

        unsafe {
            self.device
                .destroy_pipeline_layout(self.bindless_pipeline_layout, None);
        };
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
        };
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        };

        self.samplers.clear();
        self.images.clear();

        self.acceleration_structure_buffers.clear();

        self.model_blases
            .drain(..)
            .for_each(|acceleration_structure| unsafe {
                self.device
                    .acceleration_structure
                    .destroy_acceleration_structure(acceleration_structure, None);
            });

        unsafe {
            self.device
                .acceleration_structure
                .destroy_acceleration_structure(self.tlas, None);
        };
    }
}
