use std::{ffi::CString, fs::File, sync::Arc};

use ash::vk;

use crate::{
    complex_types::buffer::Buffer,
    pipeline_generator::{create_shader_module, types::PipelineDefinition},
    wrappers::{allocator::Allocator, device::Device, instance::Instance},
};

fn parse_definition(path: &str) -> PipelineDefinition {
    let file = File::open(path).unwrap();
    serde_json::from_reader(file).unwrap()
}

pub trait Pipeline {
    fn reload(&mut self);
    fn bind(&self, command_buffer: vk::CommandBuffer);
}

pub struct GraphicsPipeline {
    definition_path: String,
    device: Arc<Device>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl GraphicsPipeline {
    pub fn new(
        definition_path: String,
        device: Arc<Device>,
        pipeline_layout: vk::PipelineLayout,
    ) -> Self {
        let mut result = Self {
            definition_path,
            device,
            pipeline_layout,
            pipeline: vk::Pipeline::null(),
        };
        result.load_pipeline();
        result
    }

    fn load_pipeline(&mut self) {
        let definition = parse_definition(&self.definition_path);

        let (vertex, fragment, color_attachments, depth_attachment) = match definition.shader_info {
            super::types::ShaderInfo::Graphics {
                vertex,
                fragment,
                color_attachments,
                depth_attachment,
            } => (vertex, fragment, color_attachments, depth_attachment),
            _ => panic!("Wrong shader definition type"),
        };

        let module = create_shader_module(&self.device, &definition.shader_path);

        let vertex_entry = CString::new(vertex).unwrap();
        let fragment_entry = CString::new(fragment).unwrap();
        let depth_format = if let Some(depth_attachment) = &depth_attachment {
            depth_attachment.to_format()
        } else {
            vk::Format::UNDEFINED
        };
        self.pipeline = unsafe {
            self.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .layout(self.pipeline_layout)
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .name(&vertex_entry)
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .module(module),
                            vk::PipelineShaderStageCreateInfo::default()
                                .name(&fragment_entry)
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
                            &vk::PipelineColorBlendStateCreateInfo::default().attachments(
                                &color_attachments
                                    .iter()
                                    .map(|color_attachment| {
                                        vk::PipelineColorBlendAttachmentState::default()
                                            .color_write_mask(color_attachment.to_write_mask())
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::default()
                                .depth_write_enable(depth_attachment.is_some())
                                .depth_test_enable(depth_attachment.is_some())
                                .depth_compare_op(vk::CompareOp::LESS),
                        )
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(
                                    &color_attachments
                                        .iter()
                                        .map(|color_attachment| color_attachment.to_format())
                                        .collect::<Vec<_>>(),
                                )
                                .depth_attachment_format(depth_format),
                        )],
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            self.device.destroy_shader_module(module, None);
        };
    }

    pub fn draw_indexed(
        &self,
        command_buffer: vk::CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                command_buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
}

impl Pipeline for GraphicsPipeline {
    fn reload(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };

        self.load_pipeline();
    }

    fn bind(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
        };
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };
    }
}

pub struct ComputePipeline {
    definition_path: String,
    device: Arc<Device>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ComputePipeline {
    pub fn new(
        definition_path: String,
        device: Arc<Device>,
        pipeline_layout: vk::PipelineLayout,
    ) -> Self {
        let mut result = Self {
            definition_path,
            device,
            pipeline_layout,
            pipeline: vk::Pipeline::null(),
        };
        result.load_pipeline();
        result
    }

    fn load_pipeline(&mut self) {
        let definition = parse_definition(&self.definition_path);

        let entry = match definition.shader_info {
            super::types::ShaderInfo::Compute { entry } => entry,
            _ => panic!("Wrong shader definition type"),
        };

        let module = create_shader_module(&self.device, &definition.shader_path);

        let entry = CString::new(entry.clone()).unwrap();
        self.pipeline = unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(self.pipeline_layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .name(&entry)
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(module),
                        )],
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            self.device.destroy_shader_module(module, None);
        };
    }

    pub fn dispatch(&self, command_buffer: vk::CommandBuffer, width: u32, height: u32, depth: u32) {
        unsafe {
            self.device
                .cmd_dispatch(command_buffer, width, height, depth);
        }
    }
}

impl Pipeline for ComputePipeline {
    fn reload(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };

        self.load_pipeline();
    }

    fn bind(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        };
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };
    }
}

pub struct RaytracingPipeline {
    definition_path: String,
    instance: Arc<Instance>,
    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    allocator: Arc<Allocator>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    raygen_shader_binding_tables: vk::StridedDeviceAddressRegionKHR,
    miss_shader_binding_tables: vk::StridedDeviceAddressRegionKHR,
    hit_shader_binding_tables: vk::StridedDeviceAddressRegionKHR,
    sbt_buffer: Buffer,
}

impl RaytracingPipeline {
    pub fn new(
        definition_path: String,
        instance: Arc<Instance>,
        physical_device: vk::PhysicalDevice,
        device: Arc<Device>,
        allocator: Arc<Allocator>,
        pipeline_layout: vk::PipelineLayout,
    ) -> Self {
        let sbt_buffer = Buffer::new(
            &device,
            allocator.clone(),
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            1,
            "SBT Buffer",
        );
        let mut result = Self {
            definition_path,
            instance,
            physical_device,
            device,
            allocator,
            pipeline_layout,
            pipeline: vk::Pipeline::null(),
            raygen_shader_binding_tables: vk::StridedDeviceAddressRegionKHR::default(),
            miss_shader_binding_tables: vk::StridedDeviceAddressRegionKHR::default(),
            hit_shader_binding_tables: vk::StridedDeviceAddressRegionKHR::default(),
            sbt_buffer,
        };
        result.load_pipeline();
        result
    }

    fn load_pipeline(&mut self) {
        let definition = parse_definition(&self.definition_path);

        let (materials, raygen, miss) = match definition.shader_info {
            super::types::ShaderInfo::Raytracing {
                materials,
                raygen,
                miss,
            } => (materials, raygen, miss),
            _ => panic!("Wrong shader definition type"),
        };

        let mut stages = Vec::new();
        let mut groups = Vec::new();

        let main_module = create_shader_module(&self.device, &definition.shader_path);

        let raygen_entry = CString::new(raygen).unwrap();
        let raygen_stage_id = stages.len() as u32;
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(main_module)
                .name(&raygen_entry),
        );

        groups.push(
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(raygen_stage_id)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
        );

        let miss_entry = CString::new(miss.clone()).unwrap();
        let miss_stage_id = stages.len() as u32;
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(main_module)
                .name(&miss_entry),
        );

        groups.push(
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(miss_stage_id)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR),
        );

        let shader_data: Vec<_> = materials
            .iter()
            .map(|material| {
                (
                    create_shader_module(&self.device, &material.shader_path),
                    CString::new(material.closest_hit.clone()).unwrap(),
                    material
                        .any_hit
                        .as_ref()
                        .map(|any_hit| CString::new(any_hit.clone()).unwrap()),
                )
            })
            .collect();

        for (module, closest_hit_entry, any_hit_entry) in &shader_data {
            let closest_hit_stage_id = stages.len() as u32;
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                    .module(*module)
                    .name(closest_hit_entry),
            );

            let mut any_hit_stage_id = vk::SHADER_UNUSED_KHR;
            if let Some(any_hit_entry) = any_hit_entry {
                any_hit_stage_id = stages.len() as u32;
                stages.push(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                        .module(*module)
                        .name(any_hit_entry),
                );
            }

            groups.push(
                vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .intersection_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(closest_hit_stage_id)
                    .any_hit_shader(any_hit_stage_id),
            );
        }

        let hit_handle_count = (groups.len() - 2) as u32;

        self.pipeline = unsafe {
            self.device
                .ray_tracing
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoKHR::default()
                        .layout(self.pipeline_layout)
                        .stages(&stages)
                        .groups(&groups)
                        .max_pipeline_ray_recursion_depth(1)],
                    None,
                )
                .unwrap()[0]
        };

        let mut rt_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut rt_properties);
        unsafe {
            self.instance
                .get_physical_device_properties2(self.physical_device, &mut properties);
        };

        let total_group_count = hit_handle_count + 2;

        let align_up = |x: u32, alignment: u32| x.next_multiple_of(alignment);

        let handle_size = rt_properties.shader_group_handle_size;
        let handle_alignment = rt_properties.shader_group_handle_alignment;
        let base_alignment = rt_properties.shader_group_base_alignment;
        let handle_size_aligned = align_up(handle_size, handle_alignment);

        let raygen_size = handle_size_aligned;
        let miss_size = handle_size_aligned;
        let hit_size = hit_handle_count * handle_size_aligned;
        let callable_size = 0;

        let raygen_offset = 0;
        let miss_offset = align_up(raygen_offset + raygen_size, base_alignment);
        let hit_offset = align_up(miss_offset + miss_size, base_alignment);
        let callable_offset = align_up(hit_offset + hit_size, base_alignment);

        let sbt_size = callable_offset + callable_size;

        let sbt_handles = unsafe {
            self.device
                .ray_tracing
                .get_ray_tracing_shader_group_handles(
                    self.pipeline,
                    0,
                    total_group_count,
                    (total_group_count * handle_size) as usize,
                )
                .unwrap()
        };
        let mut sbt_data = vec![0u8; sbt_size as usize];

        {
            let src_start = 0;
            let src_end = src_start + handle_size as usize;

            let dst_start = raygen_offset as usize;
            let dst_end = dst_start + handle_size as usize;

            sbt_data[dst_start..dst_end].copy_from_slice(&sbt_handles[src_start..src_end]);
        }

        {
            let src_start = handle_size as usize;
            let src_end = src_start + handle_size as usize;

            let dst_start = miss_offset as usize;
            let dst_end = dst_start + handle_size as usize;

            sbt_data[dst_start..dst_end].copy_from_slice(&sbt_handles[src_start..src_end]);
        }

        for i in 0..hit_handle_count {
            let src_start = ((i + 2) * handle_size) as usize;
            let src_end = src_start + handle_size as usize;

            let dst_start = hit_offset as usize + (i * handle_size_aligned) as usize;
            let dst_end = dst_start + handle_size as usize;

            sbt_data[dst_start..dst_end].copy_from_slice(&sbt_handles[src_start..src_end]);
        }

        self.sbt_buffer = Buffer::new(
            &self.device,
            self.allocator.clone(),
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            sbt_size as u64,
            "SBT Buffer",
        );
        self.sbt_buffer.write(&sbt_data, 0);

        self.raygen_shader_binding_tables = self
            .raygen_shader_binding_tables
            .device_address(self.sbt_buffer.address + raygen_offset as u64)
            .stride(handle_size_aligned as u64)
            .size(raygen_size as u64);

        self.miss_shader_binding_tables = self
            .miss_shader_binding_tables
            .device_address(self.sbt_buffer.address + miss_offset as u64)
            .stride(handle_size_aligned as u64)
            .size(miss_size as u64);

        self.hit_shader_binding_tables = self
            .hit_shader_binding_tables
            .device_address(self.sbt_buffer.address + hit_offset as u64)
            .stride(handle_size_aligned as u64)
            .size(hit_size as u64);

        unsafe {
            self.device.destroy_shader_module(main_module, None);
            shader_data
                .iter()
                .for_each(|(module, _, _)| self.device.destroy_shader_module(*module, None));
        };
    }

    pub fn trace_rays(
        &self,
        command_buffer: vk::CommandBuffer,
        width: u32,
        height: u32,
        depth: u32,
    ) {
        unsafe {
            self.device.ray_tracing.cmd_trace_rays(
                command_buffer,
                &self.raygen_shader_binding_tables,
                &self.miss_shader_binding_tables,
                &self.hit_shader_binding_tables,
                &vk::StridedDeviceAddressRegionKHR::default(),
                width,
                height,
                depth,
            );
        }
    }
}

impl Pipeline for RaytracingPipeline {
    fn reload(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };

        self.load_pipeline();
    }

    fn bind(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
        };
    }
}

impl Drop for RaytracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };
    }
}
