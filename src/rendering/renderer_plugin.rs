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
use vulkan_utils::pipeline_generator::pipeline_types::{
    ComputePipeline, GraphicsPipeline, Pipeline, RaytracingPipeline,
};

use crate::{
    assets::model::Model,
    rendering::{
        components::camera::Camera,
        generated_pipelines::{
            MaterialsPipelinePushConstants, RaytracingPipelinePushConstants,
            VisibilityPipelinePushConstants, create_materials_pipeline, create_raytracing_pipeline,
            create_visibility_pipeline,
        },
        resource_manager::{ImageReference, ImageSize, InstanceReference, ResourceManager},
        vulkan_state::VulkanState,
    },
};

#[derive(Component)]
struct Renderable {
    instance_ref: InstanceReference,
}

#[derive(Resource)]
struct RendererState {
    visibility_pipeline: GraphicsPipeline,
    materials_pipeline: ComputePipeline,
    ray_tracing_pipeline: RaytracingPipeline,

    depth: ImageReference,
    visibility: ImageReference,

    base_color_output: ImageReference,
    normal_output: ImageReference,
    metallic_roughness_output: ImageReference,
    emissive_output: ImageReference,
    direct_lighting: ImageReference,

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

    let visibility_pipeline = create_visibility_pipeline(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let materials_pipeline = create_materials_pipeline(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let ray_tracing_pipeline = create_raytracing_pipeline(
        vulkan_state.instance.clone(),
        vulkan_state.physical_device,
        vulkan_state.device.clone(),
        vulkan_state.allocator.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    commands.insert_resource(RendererState {
        visibility_pipeline,
        materials_pipeline,
        ray_tracing_pipeline,

        depth: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            1,
            1,
            "Depth".to_owned(),
        ),
        visibility: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
            1,
            "Visibility".to_owned(),
        ),

        base_color_output: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
            1,
            1,
            "Base color material texture".to_owned(),
        ),
        normal_output: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R16G16B16A16_SNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            1,
            1,
            "Normal material texture".to_owned(),
        ),
        metallic_roughness_output: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            1,
            1,
            "Metallic-roughness material texture".to_owned(),
        ),
        emissive_output: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            1,
            1,
            "Emissive material texture".to_owned(),
        ),
        direct_lighting: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            1,
            1,
            "Direct lighting".to_owned(),
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
        let instance_ref =
            resource_manager.create_instance(&transform.to_matrix(), &model.model_ref);
        commands.entity(entity).insert(Renderable { instance_ref });
    }

    resource_manager.build_acceleration_structures(&vulkan_state.device);
}

fn render(
    mut vulkan_state: ResMut<VulkanState>,
    renderer_state: Res<RendererState>,
    renderables: Query<(&Model, &Renderable)>,
    mut resource_manager: ResMut<ResourceManager>,
    camera: Single<(&Transform, &Camera)>,
    window: Single<&Window>,
) {
    let width = window.width() as u32;
    let height = window.height() as u32;
    if width != vulkan_state.extent.width || height != vulkan_state.extent.height {
        resource_manager.resize(width, height);
    }
    if !vulkan_state.start_frame(width, height) {
        return;
    }

    // let light_direction = Vec3::new(-1.0, 4.0, 2.0).normalize();

    let command_buffer = vulkan_state.get_command_buffer();
    let camera_transform = camera.0;
    let camera_data = camera.1;
    let view_projection = camera_data
        .projection_matrix(vulkan_state.extent.width, vulkan_state.extent.height)
        * camera_transform.to_matrix().inverse();

    let device = vulkan_state.device.clone();

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[
                vulkan_state.current_image().get_transition_barrier(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ),
                resource_manager
                    .get_image_mut(&renderer_state.depth)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    ),
                resource_manager
                    .get_image_mut(&renderer_state.visibility)
                    .get_transition_barrier(
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    ),
            ]),
        );
    };

    unsafe {
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            resource_manager.bindless_pipeline_layout,
            0,
            &[resource_manager.descriptor_set],
            &[],
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            resource_manager.bindless_pipeline_layout,
            0,
            &[resource_manager.descriptor_set],
            &[],
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            resource_manager.bindless_pipeline_layout,
            0,
            &[resource_manager.descriptor_set],
            &[],
        );
    }

    let mut push_constants = VisibilityPipelinePushConstants {
        view_projection: [0.0; 16],
        model_data: resource_manager.model_buffer.address,
        instance_data: resource_manager.instance_buffer.address,
    };

    unsafe {
        renderer_state.visibility_pipeline.bind(command_buffer);

        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(vulkan_state.extent.width as f32)
                .height(vulkan_state.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)],
        );
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D::default()
                .offset(vk::Offset2D::default().x(0).y(0))
                .extent(vulkan_state.extent)],
        );

        device.cmd_begin_rendering(
            command_buffer,
            &vk::RenderingInfo::default()
                .layer_count(1)
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default().x(0).y(0))
                        .extent(vulkan_state.extent),
                )
                .color_attachments(&[vk::RenderingAttachmentInfo::default()
                    .image_view(resource_manager.get_image(&renderer_state.visibility).view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })])
                .depth_attachment(
                    &vk::RenderingAttachmentInfo::default()
                        .image_view(resource_manager.get_image(&renderer_state.depth).view)
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0),
                        }),
                ),
        );
        view_projection.write_cols_to_slice(push_constants.view_projection.as_mut_slice());

        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        for (model, renderable) in renderables {
            let index_data = resource_manager.get_index_data(model.model_ref);
            device.cmd_bind_index_buffer(
                command_buffer,
                index_data.index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            renderer_state.visibility_pipeline.draw_indexed(
                command_buffer,
                index_data.index_count,
                1,
                0,
                0,
                renderable.instance_ref as u32,
            );
        }

        device.cmd_end_rendering(command_buffer);

        // Materials
        renderer_state.materials_pipeline.bind(command_buffer);

        let mut materials_push_constants = MaterialsPipelinePushConstants {
            view_projection: [0.0; 16],
            model_data: resource_manager.model_buffer.address,
            instance_data: resource_manager.instance_buffer.address,
            resolution: [
                vulkan_state.extent.width as f32,
                vulkan_state.extent.height as f32,
            ],
            visibility_buffer_id: renderer_state.visibility.into(),
            base_color_output_id: renderer_state.base_color_output.into(),
            normal_output_id: renderer_state.normal_output.into(),
            metallic_roughness_output_id: renderer_state.metallic_roughness_output.into(),
            emissive_output_id: renderer_state.emissive_output.into(),
            _pad0: 0,
        };
        view_projection
            .write_cols_to_slice(materials_push_constants.view_projection.as_mut_slice());

        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&materials_push_constants),
        );

        renderer_state.materials_pipeline.dispatch(
            command_buffer,
            vulkan_state.extent.width.div_ceil(8),
            vulkan_state.extent.height.div_ceil(8),
            1,
        );

        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(
                &[
                    renderer_state.base_color_output,
                    renderer_state.normal_output,
                    renderer_state.normal_output,
                    renderer_state.emissive_output,
                ]
                .iter()
                .map(|id| {
                    resource_manager.get_image_mut(id).get_transition_barrier(
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_READ,
                        vk::ImageLayout::GENERAL,
                    )
                })
                .collect::<Vec<_>>(),
            ),
        );

        resource_manager
            .get_image_mut(&renderer_state.direct_lighting)
            .immediate_transition(
                command_buffer,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_WRITE,
                vk::ImageLayout::GENERAL,
            );

        renderer_state.ray_tracing_pipeline.bind(command_buffer);

        // Path tracing
        let mut raytracing_push_constants = RaytracingPipelinePushConstants {
            acceleration_structure: device
                .acceleration_structure
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(resource_manager.tlas),
                ),
            model_data: resource_manager.model_buffer.address,
            instance_data: resource_manager.instance_buffer.address,
            view_projection_inv: [0.0; 16],
            camera_position: [0.0; 3],
            output: renderer_state.direct_lighting.into(),
            resolution: [
                vulkan_state.extent.width as f32,
                vulkan_state.extent.height as f32,
            ],
            frame: renderer_state.frame,
            _pad0: 0,
        };
        view_projection
            .inverse()
            .write_cols_to_slice(&mut raytracing_push_constants.view_projection_inv);

        camera_transform
            .translation
            .write_to_slice(raytracing_push_constants.camera_position.as_mut_slice());

        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&raytracing_push_constants),
        );

        renderer_state
            .ray_tracing_pipeline
            .trace_rays(command_buffer, width, height, 1);

        let current_image = vulkan_state.current_image();
        current_image.immediate_transition(
            command_buffer,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        resource_manager
            .get_image_mut(&renderer_state.direct_lighting)
            .immediate_transition(
                command_buffer,
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

        let source_image = resource_manager.get_image(&renderer_state.direct_lighting);
        device.cmd_blit_image(
            command_buffer,
            source_image.handle,
            source_image.layout,
            current_image.handle,
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
    };
    resource_manager
        .get_image_mut(&renderer_state.direct_lighting)
        .immediate_transition(
            command_buffer,
            vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::NONE,
            vk::ImageLayout::GENERAL,
        );

    vulkan_state.current_image().immediate_transition(
        command_buffer,
        vk::PipelineStageFlags2::BLIT,
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::NONE,
        vk::AccessFlags2::NONE,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );

    vulkan_state.end_frame(command_buffer);
}
