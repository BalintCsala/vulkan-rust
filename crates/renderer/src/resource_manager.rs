use std::{collections::HashMap, sync::Arc};

use ash::vk;
use bevy::{
    ecs::{component::Component, resource::Resource},
    math::{Mat3, Mat4, Vec3},
};

use crate::{
    command_cache::CommandCache,
    generated_pipelines::{MipmapPipelinePushConstants, create_mipmap_pipeline},
    mesh::{GeometryType, GpuMesh, Mesh},
    vulkan_state::FRAMES_IN_FLIGHT,
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

const MAX_MESH_COUNT: usize = 16384;
const MATERIAL_BUFFER_SIZE: u64 = 1024 * 1024;

const MAX_INSTANCE_COUNT: usize = 65535;
const INSTANCE_BUFFER_STRIDE: usize = MAX_INSTANCE_COUNT * size_of::<GpuInstance>();

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

#[derive(Component, Clone, Copy)]
pub struct MeshReference {
    address: vk::DeviceAddress,
    id: usize,
}

#[derive(Component, Clone, Copy)]
pub struct MaterialReference {
    pub address: vk::DeviceAddress,
}

pub type InstanceReference = u16;

#[repr(C)]
struct GpuInstance {
    mesh: vk::DeviceAddress,
    material: vk::DeviceAddress,
    model: [f32; 16],
    normal: [f32; 9],
}

impl GpuInstance {
    pub fn new(model: Mat4, mesh: &MeshReference, material: &MaterialReference) -> Self {
        Self {
            mesh: mesh.address,
            material: material.address,
            model: model.to_cols_array(),
            normal: Mat3::from_mat4(model).inverse().transpose().to_cols_array(),
        }
    }
}

pub struct MeshInfo {
    pub index_buffer: Buffer,
    pub index_count: u32,
    blas: vk::AccelerationStructureKHR,
    _buffers: Vec<Buffer>,
}

pub struct ImageInfo {
    size: ImageSize,
    usage: vk::ImageUsageFlags,
    array_layers: u32,
    name: String,
    image: Image,
}

struct PendingBlasBuild {
    geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
    build_range_infos: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
    scratch_buffer: Buffer,
    blas: vk::AccelerationStructureKHR,
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
    pub mesh_buffer: Buffer,
    pub mesh_offset: usize,
    pub meshes: Vec<MeshInfo>,

    material_offset: usize,
    pub material_buffer: Buffer,

    _blas_buffers: Vec<Buffer>,
    pub tlas: vk::AccelerationStructureKHR,
    acceleration_structure_buffers: Vec<Buffer>,
    fence: Fence,

    pending_blas_builds: Vec<PendingBlasBuild>,

    instance_buffer: Buffer,
    rt_instance_buffer: Buffer,
    rt_instance_count: usize,

    staging_buffer: Buffer,

    mipmap_pipeline: ComputePipeline,

    limits: Limits,

    frame_index: u32,
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

        let mesh_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::empty(),
            (MAX_MESH_COUNT * size_of::<GpuMesh>()) as u64,
            "Mesh buffer",
            None,
        );

        let material_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::empty(),
            MATERIAL_BUFFER_SIZE,
            "Material buffer",
            None,
        );

        let instance_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::empty(),
            (INSTANCE_BUFFER_STRIDE * FRAMES_IN_FLIGHT) as u64,
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

        let fence = Fence::new(
            device.clone(),
            &vk::FenceCreateInfo::default(),
            "Resource manager fence",
        );

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

            mesh_buffer,
            mesh_offset: 0,
            meshes: Vec::new(),

            material_offset: 0,
            material_buffer,

            acceleration_structure_buffers: Vec::new(),

            pending_blas_builds: Vec::new(),

            tlas: vk::AccelerationStructureKHR::null(),

            rt_instance_buffer,
            rt_instance_count: 0,
            instance_buffer,
            _blas_buffers: Vec::new(),

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

            frame_index: 0,
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

    fn queue_blas_build(
        &mut self,
        positions: vk::DeviceAddress,
        position_count: u32,
        indices: vk::DeviceAddress,
        index_count: u32,
        opaque: bool,
    ) -> vk::AccelerationStructureKHR {
        let geometries = vec![
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: positions,
                        })
                        .vertex_stride(size_of::<Vec3>() as u64)
                        .max_vertex(position_count - 1)
                        .index_data(vk::DeviceOrHostAddressConstKHR {
                            device_address: indices,
                        })
                        .index_type(vk::IndexType::UINT32),
                })
                .flags(if opaque {
                    vk::GeometryFlagsKHR::OPAQUE
                } else {
                    vk::GeometryFlagsKHR::empty()
                }),
        ];

        let build_range_infos = vec![
            vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(index_count / 3),
        ];

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .geometries(&geometries)
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
                    &[index_count / 3],
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

        self.acceleration_structure_buffers.push(blas_buffer);
        self.pending_blas_builds.push(PendingBlasBuild {
            geometries,
            build_range_infos,
            scratch_buffer,
            blas,
        });

        blas
    }

    pub fn upload_mesh(&mut self, mesh: &impl Mesh) -> MeshReference {
        let indices = Buffer::from_data(
            &self.device,
            self.allocator.clone(),
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            mesh.indices(),
            &format!("{} index buffer", mesh.name()),
            None,
        );

        let positions = Buffer::from_data(
            &self.device,
            self.allocator.clone(),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            mesh.positions(),
            &format!("{} position buffer", mesh.name()),
            None,
        );

        let normals = mesh.normals().map(|normals| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                normals,
                &format!("{} normal buffer", mesh.name()),
                None,
            )
        });

        let tangents = mesh.tangents().map(|tangents| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                tangents,
                &format!("{} tangent buffer", mesh.name()),
                None,
            )
        });

        let texcoords_0 = mesh.texcoords_0().map(|texcoords_0| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                texcoords_0,
                &format!("{} texcoord_0 buffer", mesh.name()),
                None,
            )
        });

        let texcoords_1 = mesh.texcoords_1().map(|texcoords_1| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                texcoords_1,
                &format!("{} texcoord_1 buffer", mesh.name()),
                None,
            )
        });

        let colors = mesh.colors().map(|colors| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                colors,
                &format!("{} color buffer", mesh.name()),
                None,
            )
        });

        let joints = mesh.joints().map(|joints| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                joints,
                &format!("{} joint buffer", mesh.name()),
                None,
            )
        });

        let weights = mesh.weights().map(|weights| {
            Buffer::from_data(
                &self.device,
                self.allocator.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                weights,
                &format!("{} weight buffer", mesh.name()),
                None,
            )
        });

        let gpu_mesh = GpuMesh {
            indices: indices.address,
            positions: positions.address,
            normals: normals.as_ref().map_or(0, |normals| normals.address),
            tangents: tangents.as_ref().map_or(0, |tangents| tangents.address),
            texcoords_0: texcoords_0
                .as_ref()
                .map_or(0, |texcoords_0| texcoords_0.address),
            texcoords_1: texcoords_1
                .as_ref()
                .map_or(0, |texcoords_1| texcoords_1.address),
            colors: colors.as_ref().map_or(0, |colors| colors.address),
            joints: joints.as_ref().map_or(0, |joints| joints.address),
            weights: weights.as_ref().map_or(0, |weights| weights.address),
        };

        let index_count = u32::try_from(mesh.indices().len()).unwrap();

        let blas = self.queue_blas_build(
            positions.address,
            u32::try_from(mesh.positions().len()).unwrap(),
            indices.address,
            u32::try_from(mesh.indices().iter().len()).unwrap(),
            matches!(mesh.geometry_type(), GeometryType::Opaque),
        );

        let address = self.mesh_buffer.address + u64::try_from(self.mesh_offset).unwrap();
        self.mesh_buffer.write(&[gpu_mesh], self.mesh_offset);
        self.mesh_offset += size_of::<GpuMesh>();

        let id = self.meshes.len();

        self.meshes.push(MeshInfo {
            index_buffer: indices,
            index_count,
            blas,
            _buffers: [
                Some(positions),
                normals,
                tangents,
                texcoords_0,
                texcoords_1,
                colors,
                joints,
                weights,
            ]
            .into_iter()
            .flatten()
            .collect(),
        });

        MeshReference { address, id }
    }

    pub fn upload_material_data<T>(&mut self, material_data: &T) -> MaterialReference {
        let alignment = align_of::<T>();

        let offset = self.material_offset.next_multiple_of(alignment);

        self.material_buffer
            .write(std::slice::from_ref(material_data), offset);
        self.material_offset = offset + size_of::<T>();

        MaterialReference {
            address: self.material_buffer.address + u64::try_from(offset).unwrap(),
        }
    }

    fn build_pending_blases(&mut self) {
        if self.pending_blas_builds.is_empty() {
            return;
        }
        let mut build_infos = Vec::new();
        let mut build_range_infos = Vec::new();

        for pending_build in &self.pending_blas_builds {
            build_infos.push(
                vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .geometries(&pending_build.geometries)
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: pending_build.scratch_buffer.address,
                    })
                    .dst_acceleration_structure(pending_build.blas),
            );

            build_range_infos.push(pending_build.build_range_infos.as_slice());
        }

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
        self.build_pending_blases();

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

    pub fn create_instance(&mut self, model_matrix: Mat4, mesh: &MeshReference) {
        self.build_pending_blases();

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
                        .acceleration_structure(self.meshes[mesh.id].blas),
                )
        };
        let rt_instance_data = [vk::AccelerationStructureInstanceKHR {
            transform: vk::TransformMatrixKHR {
                matrix: rt_instance,
            },
            instance_custom_index_and_mask: vk::Packed24_8::new(
                u32::try_from(mesh.id).unwrap(),
                0xFF,
            ),
            instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                0,
                vk::GeometryInstanceFlagsKHR::empty().as_raw() as u8,
            ),
            acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                device_handle: blas_address,
            },
        }];

        let rt_instance_id = self.rt_instance_count;
        self.rt_instance_count += 1;
        self.rt_instance_buffer.write(
            &rt_instance_data,
            rt_instance_id * size_of::<vk::AccelerationStructureInstanceKHR>(),
        );
    }

    pub fn write_instances(&mut self, instances: &[(Mat4, &MeshReference, &MaterialReference)]) {
        let gpu_instances: Vec<_> = instances
            .iter()
            .map(|(model_matrix, mesh, material)| GpuInstance::new(*model_matrix, mesh, material))
            .collect();
        self.instance_buffer.write(
            &gpu_instances,
            usize::try_from(self.current_instance_buffer_offset()).unwrap(),
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

    pub fn add_sampler(&mut self, create_info: &vk::SamplerCreateInfo) -> SamplerReference {
        let reference = self.samplers.len() as SamplerReference;

        let sampler = Sampler::new(self.device.clone(), create_info);

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

    pub fn next_frame(&mut self) {
        self.frame_index = (self.frame_index + 1) % u32::try_from(FRAMES_IN_FLIGHT).unwrap();
    }

    pub fn get_index_buffer(&self, mesh: &MeshReference) -> &Buffer {
        &self.meshes[mesh.id].index_buffer
    }

    pub fn get_index_count(&self, mesh: &MeshReference) -> u32 {
        self.meshes[mesh.id].index_count
    }

    pub fn get_image(&self, image: &ImageReference) -> &Image {
        &self.images[image].image
    }

    pub fn get_image_reference_by_name(&self, name: &str) -> Option<ImageReference> {
        self.images_by_name.get(name).copied()
    }

    fn current_instance_buffer_offset(&self) -> u64 {
        u64::try_from(INSTANCE_BUFFER_STRIDE).unwrap() * u64::from(self.frame_index)
    }

    pub fn get_instance_buffer_address(&self) -> vk::DeviceAddress {
        self.instance_buffer.address + self.current_instance_buffer_offset()
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

        self.meshes.drain(..).for_each(|mesh| unsafe {
            self.device
                .acceleration_structure
                .destroy_acceleration_structure(mesh.blas, None);
        });

        unsafe {
            self.device
                .acceleration_structure
                .destroy_acceleration_structure(self.tlas, None);
        };
    }
}
