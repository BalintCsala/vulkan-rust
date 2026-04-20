use ash::vk;
use bevy::{
    app::{Plugin, PostUpdate, PreStartup, PreUpdate},
    ecs::{
        component::Component,
        entity::Entity,
        query::Added,
        resource::Resource,
        system::{Commands, Query, Res, ResMut, Single},
    },
    math::Vec3,
    transform::components::Transform,
    window::{RawHandleWrapperHolder, Window},
};

use crate::{
    assets::model::Model,
    rendering::{
        components::camera::Camera,
        generated_pipelines::{
            DirectLightingPipeline, DirectLightingPipelinePushConstants, MaterialsPipeline,
            MaterialsPipelinePushConstants, Pipeline, VisibilityPipeline,
            VisibilityPipelinePushConstants,
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
    depth_buffer: ImageReference,

    visibility_buffer: ImageReference,

    visibility_pipeline: VisibilityPipeline,
    materials_pipeline: MaterialsPipeline,

    base_color_output: ImageReference,
    normal_output: ImageReference,
    metallic_roughness_output: ImageReference,
    emissive_output: ImageReference,
    direct_lighting: i16,
    direct_lighting_pipeline: DirectLightingPipeline,
}

pub struct RendererPlugin;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_systems(PreStartup, create_render_resources)
            .add_systems(PreUpdate, on_new_renderable)
            .add_systems(PostUpdate, render);
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
        vulkan_state.debug_utils_device.clone(),
        vulkan_state.queue,
        vulkan_state.extent,
    );

    let visibility_pipeline = VisibilityPipeline::new(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let materials_pipeline = MaterialsPipeline::new(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let direct_lighting_pipeline = DirectLightingPipeline::new(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    commands.insert_resource(RendererState {
        depth_buffer: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
            1,
            1,
            "Depth buffer".to_owned(),
        ),

        visibility_buffer: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
            1,
            "Visibility buffer".to_owned(),
        ),

        visibility_pipeline,
        materials_pipeline,
        direct_lighting_pipeline,

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
    });
    commands.insert_resource(resource_manager);
    commands.insert_resource(vulkan_state);
}

fn on_new_renderable(
    mut commands: Commands,
    renderables: Query<(Entity, &Transform, &Model), Added<Model>>,
    mut resource_manager: ResMut<ResourceManager>,
) {
    for (entity, transform, model) in renderables {
        let instance_ref =
            resource_manager.create_instance(&transform.to_matrix(), &model.model_ref);
        commands.entity(entity).insert(Renderable { instance_ref });
    }
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

    let command_buffer = vulkan_state.get_command_buffer();
    let camera_transform = camera.0;
    let camera_data = camera.1;
    let view_projection = camera_data
        .projection_matrix(vulkan_state.extent.width, vulkan_state.extent.height)
        * camera_transform.to_matrix().inverse();

    let device = vulkan_state.device.clone();

    vulkan_state
        .current_image()
        .immediate_transition_from_undefined(
            command_buffer,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );
    resource_manager
        .get_image_mut(&renderer_state.depth_buffer)
        .immediate_transition_from_undefined(
            command_buffer,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );
    resource_manager
        .get_image_mut(&renderer_state.visibility_buffer)
        .immediate_transition_from_undefined(
            command_buffer,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

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
    }

    unsafe {
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
                    .image_view(
                        resource_manager
                            .get_image(&renderer_state.visibility_buffer)
                            .view,
                    )
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
                        .image_view(
                            resource_manager
                                .get_image(&renderer_state.depth_buffer)
                                .view,
                        )
                        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0),
                        }),
                ),
        );
        let mut push_constants = VisibilityPipelinePushConstants {
            view_projection: [0.0; 16],
            model_data: resource_manager.model_buffer.address,
            instance_data: resource_manager.instance_buffer.address,
        };
        view_projection.write_cols_to_slice(push_constants.view_projection.as_mut_slice());

        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&push_constants),
        );

        renderer_state.visibility_pipeline.bind(command_buffer);

        for (model, renderable) in renderables {
            let index_data = resource_manager.get_index_data(model.model_ref);
            device.cmd_bind_index_buffer(
                command_buffer,
                index_data.index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            device.cmd_draw_indexed(
                command_buffer,
                index_data.index_count,
                1,
                0,
                0,
                renderable.instance_ref as u32,
            );
        }

        device.cmd_end_rendering(command_buffer);

        renderer_state.materials_pipeline.bind(command_buffer);

        let mut materials_push_constants = MaterialsPipelinePushConstants {
            view_projection: [0.0; 16],
            model_data: resource_manager.model_buffer.address,
            instance_data: resource_manager.instance_buffer.address,
            resolution: [
                vulkan_state.extent.width as f32,
                vulkan_state.extent.height as f32,
            ],
            visibility_buffer_id: renderer_state.visibility_buffer.into(),
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

        device.cmd_dispatch(
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

        renderer_state.direct_lighting_pipeline.bind(command_buffer);
        let mut direct_lighting_push_constants = DirectLightingPipelinePushConstants {
            view_projection_inverse: [0.0; 16],
            depth_texture_id: renderer_state.depth_buffer.into(),
            base_color_id: renderer_state.base_color_output.into(),
            normal_id: renderer_state.normal_output.into(),
            metallic_roughness_id: renderer_state.metallic_roughness_output.into(),
            emissive_id: renderer_state.emissive_output.into(),
            output_id: renderer_state.direct_lighting.into(),
            sun_direction: [0.0; 3],
            sun_light: [4.0, 4.0, 4.0],
            camera_position: [0.0; 3],
            resolution: [
                vulkan_state.extent.width as f32,
                vulkan_state.extent.height as f32,
            ],
        };

        view_projection.inverse().write_cols_to_slice(
            direct_lighting_push_constants
                .view_projection_inverse
                .as_mut_slice(),
        );
        Vec3::new(1.0, 3.0, 2.0)
            .normalize()
            .write_to_slice(direct_lighting_push_constants.sun_direction.as_mut_slice());
        camera_transform.translation.write_to_slice(
            direct_lighting_push_constants
                .camera_position
                .as_mut_slice(),
        );

        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&direct_lighting_push_constants),
        );
        device.cmd_dispatch(
            command_buffer,
            vulkan_state.extent.width.div_ceil(8),
            vulkan_state.extent.height.div_ceil(8),
            1,
        );

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
                vk::PipelineStageFlags2::COMPUTE_SHADER,
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
