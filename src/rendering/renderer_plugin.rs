use std::sync::Arc;

use ash::vk;
use bevy::{
    app::{Last, MainScheduleOrder, Plugin, PreStartup, Update},
    ecs::{
        entity::Entity,
        query::{Added, Changed, With},
        resource::Resource,
        schedule::{IntoScheduleConfigs, ScheduleLabel, SystemSet},
        system::{Commands, Query, Res, ResMut, Single},
    },
    input::{ButtonInput, keyboard::KeyCode},
    math::{Mat4, Vec3},
    transform::components::{GlobalTransform, Transform},
    window::{RawHandleWrapperHolder, Window},
};
use vulkan_utils::{
    pipeline_generator::pipeline_types::{
        ComputePipeline, GraphicsPipeline, Pipeline, RaytracingPipeline,
    },
    wrappers::device::Device,
};

use crate::rendering::{
    components::{camera::Camera, model::Model},
    generated_pipelines::{
        DirectLightingPipelinePushConstants, IndirectDiffusePipelinePushConstants,
        MaterialsPipelinePushConstants, TonemapPipelinePushConstants,
        VisibilityPipelinePushConstants, create_direct_lighting_pipeline,
        create_indirect_diffuse_pipeline, create_materials_pipeline, create_tonemap_pipeline,
        create_visibility_pipeline,
    },
    resource_manager::{ImageReference, ImageSize, ResourceManager},
    vulkan_state::VulkanState,
};

#[derive(Resource)]
struct RendererState {
    device: Arc<Device>,
    visibility_pipeline: GraphicsPipeline,
    materials_pipeline: ComputePipeline,
    direct_lighting_pipeline: RaytracingPipeline,
    indirect_diffuse_pipeline: RaytracingPipeline,
    tonemap_pipeline: GraphicsPipeline,

    visibility: ImageReference,
    gbuffer: ImageReference,
    direct_lighting: ImageReference,
    indirect_diffuse: ImageReference,

    rendering_active: bool,

    frame: u32,
}

#[derive(Resource)]
pub struct CommonRenderingResources {
    pub depth: ImageReference,
    pub view_projection: Mat4,
    pub camera_position: Vec3,
}

pub struct RendererPlugin;

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PreRendering;

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct Rendering;

#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash)]
pub struct PostRendering;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum PostStages {
    Opaque,
    Translucent,
    Final,
}

pub enum RenderLayer {
    Visibility,
    GBuffer,
    Opaque,
    Translucent,
    Debug,
}

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        let mut schedule_order = app.world_mut().resource_mut::<MainScheduleOrder>();
        schedule_order.insert_after(Last, PreRendering);
        schedule_order.insert_after(PreRendering, Rendering);
        schedule_order.insert_after(Rendering, PostRendering);

        app.add_systems(PreStartup, create_render_resources)
            .add_systems(Update, (reload_shaders, update_frame).chain())
            .add_systems(
                PreRendering,
                (on_new_renderables, on_changed_renderables, start_frame).chain(),
            )
            .add_systems(
                Rendering,
                (
                    visibility,
                    resolve_visibility,
                    gbuffer,
                    opaque,
                    translucent,
                    debug,
                )
                    .chain()
                    .run_if(is_rendering_active),
            )
            .add_systems(PostRendering, end_frame.run_if(is_rendering_active));

        app.configure_sets(
            Rendering,
            (
                PostStages::Opaque
                    .after(opaque)
                    .before(translucent)
                    .run_if(is_rendering_active),
                PostStages::Translucent
                    .after(translucent)
                    .before(debug)
                    .run_if(is_rendering_active),
                PostStages::Final.after(debug).run_if(is_rendering_active),
            ),
        );

        app.add_systems(
            Rendering,
            (
                tonemap.in_set(PostStages::Translucent),
                render_lighting.in_set(PostStages::Opaque),
            ),
        );
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
        &vulkan_state.instance,
        vulkan_state.device.clone(),
        &vulkan_state.physical_device,
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

    let indirect_diffuse_pipeline = create_indirect_diffuse_pipeline(
        vulkan_state.instance.clone(),
        vulkan_state.physical_device,
        vulkan_state.device.clone(),
        vulkan_state.allocator.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let visibility_pipeline = create_visibility_pipeline(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
        vulkan_state.surface_format.format,
    );

    let materials_pipeline = create_materials_pipeline(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
    );

    let tonemap_pipeline = create_tonemap_pipeline(
        vulkan_state.device.clone(),
        resource_manager.bindless_pipeline_layout,
        vulkan_state.surface_format.format,
    );

    commands.insert_resource(RendererState {
        device: vulkan_state.device.clone(),
        visibility_pipeline,
        materials_pipeline,
        direct_lighting_pipeline,
        indirect_diffuse_pipeline,
        tonemap_pipeline,

        gbuffer: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R32G32B32A32_UINT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            1,
            1,
            "Gbuffer".to_owned(),
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
            "Indirect diffuse".to_owned(),
        ),

        rendering_active: false,

        frame: 0,
    });

    commands.insert_resource(CommonRenderingResources {
        depth: resource_manager.create_empty_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            1,
            1,
            "Depth".to_owned(),
        ),
        view_projection: Mat4::default(),
        camera_position: Vec3::default(),
    });

    commands.insert_resource(resource_manager);
    commands.insert_resource(vulkan_state);
}

fn reload_shaders(
    keys: Res<ButtonInput<KeyCode>>,
    vulkan_state: Res<VulkanState>,
    mut renderer_state: ResMut<RendererState>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        unsafe {
            vulkan_state.device.device_wait_idle().unwrap();
        };

        renderer_state.materials_pipeline.reload();
        renderer_state.visibility_pipeline.reload();
        renderer_state.direct_lighting_pipeline.reload();
        renderer_state.indirect_diffuse_pipeline.reload();
        renderer_state.tonemap_pipeline.reload();
    }
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
    mut commands: Commands,
    renderables: Query<(Entity, &GlobalTransform, &Model), Added<Model>>,
    mut resource_manager: ResMut<ResourceManager>,
) {
    for (entity, transform, model) in renderables {
        resource_manager.create_instance(entity.index(), &transform.to_matrix(), &model.model_ref);
        commands.entity(entity).insert(());
    }

    resource_manager.build_acceleration_structures();
}

fn on_changed_renderables(
    renderables: Query<(Entity, &GlobalTransform), (With<Model>, Changed<GlobalTransform>)>,
    mut resource_manager: ResMut<ResourceManager>,
) {
    for (entity, transform) in renderables {
        resource_manager.update_instance(&entity.index(), &transform.to_matrix());
    }

    resource_manager.build_acceleration_structures();
}

fn start_frame(
    mut vulkan_state: ResMut<VulkanState>,
    mut resource_manager: ResMut<ResourceManager>,
    mut renderer_state: ResMut<RendererState>,
    window: Single<&Window>,
    camera: Single<(&Transform, &Camera)>,
    mut rendering_res: ResMut<CommonRenderingResources>,
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
        renderer_state.rendering_active = false;
        return;
    }

    let (camera_transform, camera_data) = *camera;

    rendering_res.view_projection = camera_data
        .projection_matrix(vulkan_state.extent.width, vulkan_state.extent.height)
        * camera_transform.to_matrix().inverse();
    rendering_res.camera_position = camera_transform.translation;

    renderer_state.rendering_active = true;

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
}

fn is_rendering_active(renderer_state: Res<RendererState>) -> bool {
    renderer_state.rendering_active
}

fn visibility(
    mut vulkan_state: ResMut<VulkanState>,
    resource_manager: Res<ResourceManager>,
    renderer_state: Res<RendererState>,
    renderables: Query<(Entity, &Model)>,
    rendering_res: Res<CommonRenderingResources>,
) {
    let device = vulkan_state.device.clone();
    let command_buffer = vulkan_state.get_command_buffer();

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default()
                .memory_barriers(&[
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        )
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE),
                    vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE),
                ])
                .image_memory_barriers(&[vulkan_state.current_image().get_transition_barrier(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::ImageLayout::GENERAL,
                )]),
        );
    }

    renderer_state.visibility_pipeline.bind(command_buffer);

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
    };

    unsafe {
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D::default()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(vulkan_state.extent)],
        );
    }

    unsafe {
        device.cmd_begin_rendering(
            command_buffer,
            &vk::RenderingInfo::default()
                .layer_count(1)
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D { x: 0, y: 0 })
                        .extent(vulkan_state.extent),
                )
                .color_attachments(&[vk::RenderingAttachmentInfo::default()
                    .image_view(resource_manager.get_image(&renderer_state.visibility).view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            uint32: [0, 0, 0, 0],
                        },
                    })])
                .depth_attachment(
                    &vk::RenderingAttachmentInfo::default()
                        .image_view(resource_manager.get_image(&rendering_res.depth).view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue::default().depth(1.0),
                        }),
                ),
        );
    };

    let mut visibility_push_constants = VisibilityPipelinePushConstants {
        view_projection: [0.0; 16],
        models: resource_manager.model_buffer.address,
        instance_data: resource_manager.instance_buffer.address,
    };
    rendering_res
        .view_projection
        .write_cols_to_slice(visibility_push_constants.view_projection.as_mut_slice());
    unsafe {
        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&visibility_push_constants),
        );
    };

    for (entity, model) in renderables {
        let index_data = resource_manager.get_index_data(model.model_ref);
        unsafe {
            device.cmd_bind_index_buffer(
                command_buffer,
                index_data.index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
        }
        renderer_state.visibility_pipeline.draw_indexed(
            command_buffer,
            index_data.index_count,
            1,
            0,
            0,
            resource_manager.get_entity_instance_id(&entity.index()) as u32,
        );
    }

    unsafe {
        device.cmd_end_rendering(command_buffer);
    };
}

fn resolve_visibility(
    vulkan_state: Res<VulkanState>,
    resource_manager: Res<ResourceManager>,
    renderer_state: Res<RendererState>,
    rendering_res: Res<CommonRenderingResources>,
) {
    let device = vulkan_state.device.clone();
    let command_buffer = vulkan_state.get_command_buffer();

    renderer_state.materials_pipeline.bind(command_buffer);

    let mut materials_push_constants = MaterialsPipelinePushConstants {
        view_projection: [0.0; 16],
        models: resource_manager.model_buffer.address,
        instances: resource_manager.instance_buffer.address,
        resolution: [
            vulkan_state.extent.width as f32,
            vulkan_state.extent.height as f32,
        ],
        visibility_buffer_id: renderer_state.visibility.into(),
        gbuffer_id: renderer_state.gbuffer.into(),
    };
    rendering_res
        .view_projection
        .write_cols_to_slice(materials_push_constants.view_projection.as_mut_slice());

    unsafe {
        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&materials_push_constants),
        );
    };

    renderer_state.materials_pipeline.dispatch(
        command_buffer,
        vulkan_state.extent.width.div_ceil(8),
        vulkan_state.extent.height.div_ceil(8),
        1,
    );
}

fn gbuffer() {}

fn opaque() {}

fn translucent() {}

fn render_lighting(
    vulkan_state: Res<VulkanState>,
    resource_manager: Res<ResourceManager>,
    renderer_state: Res<RendererState>,
    rendering_res: Res<CommonRenderingResources>,
) {
    let device = vulkan_state.device.clone();
    let command_buffer = vulkan_state.get_command_buffer();

    let sun_direction = Vec3::new(-1.0, 4.0, 2.0).normalize();

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)]),
        );
    }

    // Direct Lighting
    let tlas_address = unsafe {
        device
            .acceleration_structure
            .get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(resource_manager.tlas),
            )
    };

    renderer_state.direct_lighting_pipeline.bind(command_buffer);
    let mut direct_lighting_push_constants = DirectLightingPipelinePushConstants {
        tlas: tlas_address,
        models: resource_manager.model_buffer.address,
        instances: resource_manager.instance_buffer.address,
        view_projection_inv: [0.0; 16],
        camera_position: [0.0; 3],
        depth_texture_id: rendering_res.depth.into(),
        gbuffer_id: renderer_state.gbuffer.into(),
        direct_lighting_output_id: renderer_state.direct_lighting.into(),
        frame: renderer_state.frame,
        _pad0: 0,
    };

    rendering_res
        .view_projection
        .inverse()
        .write_cols_to_slice(&mut direct_lighting_push_constants.view_projection_inv);
    rendering_res.camera_position.write_to_slice(
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

    renderer_state.direct_lighting_pipeline.trace_rays(
        command_buffer,
        vulkan_state.extent.width,
        vulkan_state.extent.height,
        1,
    );

    // Indirect diffuse
    renderer_state
        .indirect_diffuse_pipeline
        .bind(command_buffer);

    let mut indirect_diffuse_push_constants = IndirectDiffusePipelinePushConstants {
        tlas: tlas_address,
        models: resource_manager.model_buffer.address,
        instances: resource_manager.instance_buffer.address,
        view_projection_inv: [0.0; 16],
        sun_dir: [0.0; 3],
        depth_id: rendering_res.depth.into(),
        gbuffer_id: renderer_state.gbuffer.into(),
        output_id: renderer_state.indirect_diffuse.into(),
        frame: renderer_state.frame,
        _pad0: 0,
    };

    sun_direction.write_to_slice(&mut indirect_diffuse_push_constants.sun_dir);
    rendering_res
        .view_projection
        .inverse()
        .write_cols_to_slice(&mut indirect_diffuse_push_constants.view_projection_inv);

    unsafe {
        device.cmd_push_constants(
            command_buffer,
            resource_manager.bindless_pipeline_layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&indirect_diffuse_push_constants),
        );
    }

    // renderer_state.indirect_diffuse_pipeline.trace_rays(
    //     command_buffer,
    //     width.div_ceil(2),
    //     height.div_ceil(2),
    //     1,
    // );
}

fn debug() {}

fn tonemap(
    mut vulkan_state: ResMut<VulkanState>,
    resource_manager: Res<ResourceManager>,
    renderer_state: Res<RendererState>,
) {
    let device = vulkan_state.device.clone();
    let command_buffer = vulkan_state.get_command_buffer();

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().memory_barriers(&[
                vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ),
                vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            ]),
        );
    };

    renderer_state.tonemap_pipeline.bind(command_buffer);

    unsafe {
        device.cmd_begin_rendering(
            command_buffer,
            &vk::RenderingInfo::default()
                .layer_count(1)
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D { x: 0, y: 0 })
                        .extent(vulkan_state.extent),
                )
                .color_attachments(&[vk::RenderingAttachmentInfo::default()
                    .image_view(vulkan_state.current_image().view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    })]),
        );
    };

    let tonemap_push_constants = TonemapPipelinePushConstants {
        gbuffer_id: renderer_state.gbuffer.into(),
        direct_lighting_output_id: renderer_state.direct_lighting.into(),
        indirect_diffuse_output_id: renderer_state.indirect_diffuse.into(),
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
        .draw(command_buffer, 3, 1, 0, 0);

    unsafe {
        device.cmd_end_rendering(command_buffer);
    };
}

fn end_frame(mut vulkan_state: ResMut<VulkanState>) {
    let device = vulkan_state.device.clone();
    let command_buffer = vulkan_state.get_command_buffer();

    unsafe {
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo::default().image_memory_barriers(&[vulkan_state
                .current_image()
                .get_transition_barrier(
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::NONE,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                )]),
        );
    };

    vulkan_state.end_frame();
}

impl Drop for RendererState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        };
    }
}
