use std::sync::Arc;

use ash::vk;
use bevy::{
    app::{Last, MainScheduleOrder, Plugin, PreStartup, Update},
    ecs::{
        component::Component,
        entity::Entity,
        query::{Added, Changed},
        resource::Resource,
        schedule::ScheduleLabel,
        system::{Commands, Query, Res, ResMut, Single},
    },
    input::{ButtonInput, keyboard::KeyCode},
    transform::components::{GlobalTransform, Transform},
    window::{RawHandleWrapperHolder, Window},
};
use vulkan_utils::{
    pipeline_generator::pipeline_types::{ComputePipeline, Pipeline, RaytracingPipeline},
    wrappers::device::Device,
};

use crate::{
    assets::model::Model,
    rendering::{
        components::camera::Camera,
        generated_pipelines::{
            DirectLightingPipelinePushConstants, TonemapPipelinePushConstants,
            create_direct_lighting_pipeline, create_tonemap_pipeline,
        },
        resource_manager::{ImageReference, ImageSize, ResourceManager},
        vulkan_state::VulkanState,
    },
};

#[derive(Component)]
struct Renderable;

#[derive(Resource)]
struct RendererState {
    device: Arc<Device>,
    direct_lighting_pipeline: RaytracingPipeline,
    tonemap_pipeline: ComputePipeline,

    gbuffer: ImageReference,
    direct_lighting: ImageReference,
    indirect_diffuse: ImageReference,
    output: ImageReference,

    frame: u32,
}

pub struct RendererPlugin;

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
struct PreRendering;

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
struct Rendering;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        let mut schedule_order = app.world_mut().resource_mut::<MainScheduleOrder>();
        schedule_order.insert_after(Last, PreRendering);
        schedule_order.insert_after(PreRendering, Rendering);

        app.add_systems(PreStartup, create_render_resources)
            .add_systems(Update, update_frame)
            .add_systems(PreRendering, on_new_renderables)
            .add_systems(Rendering, render);
    }
}

fn create_render_resources(
    mut commands: Commands,
    window: Single<&Window>,
    holder: Single<&RawHandleWrapperHolder>,
) {
    let wrapper = holder.0.lock().unwrap();
    let handles = (wrapper.as_ref()).expect("No window found");

    let vulkan_state = VulkanState::new(
        handles.get_display_handle(),
        handles.get_window_handle(),
        window.width() as u32,
        window.height() as u32,
    );

    let mut resource_manager = ResourceManager::new(
        vulkan_state.device.clone(),
        vulkan_state.allocator.clone(),
        vulkan_state.queue,
        vulkan_state.extent,
    );

    let direct_lighting_pipeline = create_direct_lighting_pipeline(
        vulkan_state.instance.clone(),
        vulkan_state.physical_device,
        vulkan_state.device.clone(),
        vulkan_state.allocator.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let tonemap_pipeline = create_tonemap_pipeline(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    commands.insert_resource(RendererState {
        device: vulkan_state.device.clone(),
        direct_lighting_pipeline,
        tonemap_pipeline,

        gbuffer: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R32G32B32A32_UINT,
            vk::ImageUsageFlags::STORAGE,
            1,
            1,
            "Gbuffer image".to_owned(),
        ),

        direct_lighting: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE,
            1,
            1,
            "Direct lighting".to_owned(),
        ),
        indirect_diffuse: resource_manager.create_empty_image(
            ImageSize::Scaled(0.5, 0.5),
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE,
            1,
            1,
            "Direct lighting".to_owned(),
        ),

        output: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
            1,
            "Output".to_owned(),
        ),

        frame: 0,
    });
    commands.insert_resource(resource_manager);
    commands.insert_resource(vulkan_state);
}

fn update_frame(
    keys: Res<ButtonInput<KeyCode>>,
    mut renderer_state: ResMut<RendererState>,
    camera: Query<&Camera, Changed<Transform>>,
) {
    renderer_state.frame += 1;
    if !camera.is_empty() {
        renderer_state.frame = 0;
    }

    if keys.just_pressed(KeyCode::KeyW) {
        renderer_state.frame = 0;
    }
}

fn on_new_renderables(
    vulkan_state: Res<VulkanState>,
    mut commands: Commands,
    renderables: Query<(Entity, &GlobalTransform, &Model), Added<Model>>,
    mut resource_manager: ResMut<ResourceManager>,
) {
    for (entity, transform, model) in renderables {
        resource_manager.create_instance(&transform.to_matrix(), &model.model_ref);
        commands.entity(entity).insert(Renderable);
    }

    resource_manager.build_acceleration_structures(&vulkan_state.device);
}

fn render(
    mut vulkan_state: ResMut<VulkanState>,
    renderer_state: Res<RendererState>,
    mut resource_manager: ResMut<ResourceManager>,
    camera: Single<(&Transform, &Camera)>,
    window: Single<&Window>,
) {
    let width = window.width() as u32;
    let height = window.height() as u32;
    if width != vulkan_state.extent.width || height != vulkan_state.extent.height {
        unsafe {
            vulkan_state.device.device_wait_idle().unwrap();
        };
        resource_manager.resize(width, height);
    }
    if !vulkan_state.start_frame(width, height) {
        return;
    }

    let (camera_transform, camera_data) = *camera;
    let view_projection = camera_data
        .projection_matrix(vulkan_state.extent.width, vulkan_state.extent.height)
        * camera_transform.to_matrix().inverse();

    let device = vulkan_state.device.clone();

    let command_buffer = vulkan_state.get_command_buffer();

    [
        vk::PipelineBindPoint::GRAPHICS,
        vk::PipelineBindPoint::COMPUTE,
        vk::PipelineBindPoint::RAY_TRACING_KHR,
    ]
    .iter()
    .for_each(|bind_point| unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            *bind_point,
            resource_manager.bindless_pipeline_layout,
            0,
            &[resource_manager.descriptor_set],
            &[],
        );
    });

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[
                resource_manager
                    .get_image_mut(&renderer_state.gbuffer)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::ImageLayout::GENERAL,
                    ),
                resource_manager
                    .get_image_mut(&renderer_state.direct_lighting)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::NONE,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::ImageLayout::GENERAL,
                    ),
            ]),
        );
    }

    // Path tracing
    renderer_state.direct_lighting_pipeline.bind(command_buffer);
    let mut direct_lighting_push_constants = DirectLightingPipelinePushConstants {
        acceleration_structure: unsafe {
            device
                .acceleration_structure
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(resource_manager.tlas),
                )
        },
        model_data: resource_manager.model_buffer.address,
        instance_data: resource_manager.instance_buffer.address,
        view_projection_inv: [0.0; 16],
        camera_position: [0.0; 3],
        gbuffer: renderer_state.gbuffer.into(),
        direct_lighting_output: renderer_state.direct_lighting.into(),
        frame: renderer_state.frame,
    };

    view_projection
        .inverse()
        .write_cols_to_slice(&mut direct_lighting_push_constants.view_projection_inv);
    camera_transform.translation.write_to_slice(
        direct_lighting_push_constants
            .camera_position
            .as_mut_slice(),
    );

    unsafe {
        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&direct_lighting_push_constants),
        );
    }

    renderer_state
        .direct_lighting_pipeline
        .trace_rays(command_buffer, width, height, 1);

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[
                resource_manager
                    .get_image_mut(&renderer_state.direct_lighting)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_READ,
                        vk::ImageLayout::GENERAL,
                    ),
                resource_manager
                    .get_image_mut(&renderer_state.output)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::BLIT,
                        vk::AccessFlags2::NONE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::ImageLayout::GENERAL,
                    ),
            ]),
        );
    };

    // Tonemapping
    renderer_state.tonemap_pipeline.bind(command_buffer);
    let tonemap_push_constants = TonemapPipelinePushConstants {
        gbuffer: renderer_state.gbuffer.into(),
        direct_lighting_output: renderer_state.direct_lighting.into(),
        indirect_diffuse_output: renderer_state.indirect_diffuse.into(),
        output: renderer_state.output.into(),
    };
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&tonemap_push_constants),
        );
    }
    renderer_state
        .tonemap_pipeline
        .dispatch(command_buffer, width / 8, height / 8, 1);

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[
                resource_manager
                    .get_image_mut(&renderer_state.output)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::PipelineStageFlags2::BLIT,
                        vk::AccessFlags2::TRANSFER_READ,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ),
                vulkan_state.current_image().get_transition_barrier(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ),
            ]),
        );
    };

    let source_image = resource_manager.get_image(&renderer_state.output);
    unsafe {
        device.cmd_blit_image(
            command_buffer,
            source_image.handle,
            source_image.layout,
            vulkan_state.current_image().handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit::default()
                .src_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: vulkan_state.extent.width as i32,
                        y: vulkan_state.extent.height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: vulkan_state.extent.width as i32,
                        y: vulkan_state.extent.height as i32,
                        z: 1,
                    },
                ])],
            vk::Filter::NEAREST,
        );
    }

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[vulkan_state
                .current_image()
                .get_transition_barrier(
                    vk::PipelineStageFlags2::BLIT,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )]),
        );
    };

    vulkan_state.end_frame(command_buffer);
}

impl Drop for RendererState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        };
    }
}
