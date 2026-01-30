use std::{fs::File, sync::Arc, time::Instant};

use ash::vk;
use bevy::{
    MinimalPlugins,
    a11y::AccessibilityPlugin,
    app::{App, Startup, Update},
    ecs::{
        message::MessageReader,
        query::With,
        resource::Resource,
        schedule::IntoScheduleConfigs,
        system::{Commands, Res, ResMut, Single},
    },
    input::{
        ButtonInput, InputPlugin,
        keyboard::KeyCode,
        mouse::{MouseButton, MouseMotion},
    },
    math::{Quat, Vec3},
    time::Time,
    transform::components::Transform,
    window::{RawHandleWrapperHolder, Window, WindowPlugin, WindowResolution},
    winit::WinitPlugin,
};
use bytemuck::{Pod, Zeroable};

use crate::{
    assets::{asset_manager::AssetManager, gltf::Gltf},
    rendering::{
        components::camera::{Camera, CameraResolution},
        resource_manager::{ImageReference, ImageSize},
        shader::Shader,
        vulkan_state::{self, VulkanState},
        wrappers::device::Device,
    },
};

mod assets;
mod rendering;

const CAMERA_SPEED: f32 = 3.0;
const CAMERA_SENSITIVITY: f32 = 0.003;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PushConstant {
    model_view_projection: [f32; 16],
    model_data: vk::DeviceAddress,
}

#[derive(Resource)]
struct RenderingRes {
    device: Arc<Device>,
    depth_buffer: ImageReference,
    intermediary: ImageReference,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    gltf: Gltf,
    last_frame_time: Instant,
    frame: u32,
}

impl Drop for RenderingRes {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
        };
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        };
    }
}

fn main() {
    App::new()
        .add_plugins(MinimalPlugins)
        .add_plugins((
            InputPlugin,
            AccessibilityPlugin,
            WinitPlugin::default(),
            WindowPlugin {
                primary_window: Some(Window {
                    title: "Vulkan Rust".into(),
                    resolution: WindowResolution::new(1024, 768),
                    visible: true,
                    present_mode: bevy::window::PresentMode::Immediate,
                    ..Default::default()
                }),
                ..Default::default()
            },
        ))
        .add_systems(Startup, (init_rendering, setup_scene))
        .add_systems(Update, ((keyboard_input, mouse_input), render).chain())
        .run();
}

fn setup_scene(mut commands: Commands) {
    commands.spawn((
        Transform {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            ..Default::default()
        },
        Camera {
            fov_y: 90.0f32.to_radians(),
            z_near: 0.1,
            z_far: 100.0,
            resolution: CameraResolution::Scale(1.0),
        },
    ));
}

fn init_rendering(
    mut commands: Commands,
    window: Single<&Window>,
    holder: Single<&RawHandleWrapperHolder>,
) {
    let wrapper = holder.0.lock().unwrap();
    let handles = (wrapper.as_ref()).expect("No window found");

    let mut vulkan_state = VulkanState::new(
        handles.get_display_handle(),
        handles.get_window_handle(),
        window.width() as u32,
        window.height() as u32,
    );

    let mut asset_manager = AssetManager::new(
        &vulkan_state.device,
        &vulkan_state.allocator,
        &vulkan_state.debug_utils_device,
    );

    let shader = Shader::new(vulkan_state.device.clone(), "spv/test.spv").unwrap();

    let pipeline_layout = unsafe {
        vulkan_state
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
                        .offset(0)
                        .size(size_of::<PushConstant>() as u32)])
                    .set_layouts(&[vulkan_state.resource_manager.descriptor_layout]),
                None,
            )
            .unwrap()
    };

    let pipeline = unsafe {
        *vulkan_state
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .layout(pipeline_layout)
                    .stages(&[
                        vk::PipelineShaderStageCreateInfo::default()
                            .name(c"vs")
                            .stage(vk::ShaderStageFlags::VERTEX)
                            .module(shader.module),
                        vk::PipelineShaderStageCreateInfo::default()
                            .name(c"fs")
                            .stage(vk::ShaderStageFlags::FRAGMENT)
                            .module(shader.module),
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
                        &vk::PipelineColorBlendStateCreateInfo::default()
                            .attachments(&[vk::PipelineColorBlendAttachmentState::default()
                                .color_write_mask(vk::ColorComponentFlags::RGBA)]),
                    )
                    .depth_stencil_state(
                        &vk::PipelineDepthStencilStateCreateInfo::default()
                            .depth_write_enable(true)
                            .depth_test_enable(true)
                            .depth_compare_op(vk::CompareOp::LESS),
                    )
                    .push_next(
                        &mut vk::PipelineRenderingCreateInfo::default()
                            .color_attachment_formats(&[vulkan_state.surface_format.format])
                            .depth_attachment_format(vk::Format::D32_SFLOAT),
                    )],
                None,
            )
            .unwrap()
            .first()
            .unwrap()
    };

    commands.insert_resource(RenderingRes {
        device: vulkan_state.device.clone(),
        depth_buffer: vulkan_state.resource_manager.create_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            1,
            1,
        ),
        intermediary: vulkan_state.resource_manager.create_image(
            ImageSize::Scaled(1.0, 1.0),
            vk::Format::R16G16B16_SFLOAT,
            vk::ImageUsageFlags::STORAGE,
            1,
            1,
        ),
        gltf: Gltf::from_glb(
            &vulkan_state.device,
            &vulkan_state.allocator,
            &vulkan_state.debug_utils_device,
            &mut asset_manager,
            &mut File::open("assets/DragonAttenuation.glb").unwrap(),
        )
        .unwrap(),
        pipeline_layout,
        pipeline,
        last_frame_time: Instant::now(),
        frame: 0,
    });

    commands.insert_resource(asset_manager);
    commands.insert_resource(vulkan_state);
}

fn keyboard_input(
    mut camera: Single<&mut Transform, With<Camera>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let x = camera.local_x();
    let y = camera.local_y();
    let z = camera.local_z();
    let delta = time.delta_secs() * CAMERA_SPEED;
    if keys.pressed(KeyCode::KeyA) {
        camera.translation -= x * delta;
    }
    if keys.pressed(KeyCode::KeyD) {
        camera.translation += x * delta;
    }
    if keys.pressed(KeyCode::KeyW) {
        camera.translation -= z * delta;
    }
    if keys.pressed(KeyCode::KeyS) {
        camera.translation += z * delta;
    }
    if keys.pressed(KeyCode::KeyQ) {
        camera.translation -= y * delta;
    }
    if keys.pressed(KeyCode::KeyE) {
        camera.translation += y * delta;
    }
}

fn mouse_input(
    mut camera: Single<&mut Transform, With<Camera>>,
    keys: Res<ButtonInput<MouseButton>>,
    mut mouse_motion: MessageReader<MouseMotion>,
) {
    if !keys.pressed(MouseButton::Left) {
        return;
    }
    for motion in mouse_motion.read() {
        camera.rotate_y(-motion.delta.x * CAMERA_SENSITIVITY);
        camera.rotate_local_x(-motion.delta.y * CAMERA_SENSITIVITY);
    }
}

fn render(
    mut vk_state: ResMut<VulkanState>,
    mut rendering_res: ResMut<RenderingRes>,
    camera: Single<(&Transform, &Camera)>,
    mut window: Single<&mut Window>,
) {
    if !vk_state.start_frame(window.width() as u32, window.height() as u32) {
        return;
    }

    let command_buffer = vk_state.get_command_buffer();
    let camera_transform = camera.0;
    let camera_data = camera.1;
    let view_projection = camera_data
        .projection_matrix(vk_state.extent.width, vk_state.extent.height)
        * camera_transform.to_matrix().inverse();

    let device = vk_state.device.clone();

    vk_state.current_image().transition_from_undefined(
        command_buffer,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::NONE,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    );
    vk_state
        .resource_manager
        .get_image_mut(&rendering_res.depth_buffer)
        .transition_from_undefined(
            command_buffer,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
        );

    unsafe {
        device.cmd_set_viewport(
            command_buffer,
            0,
            &[vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(vk_state.extent.width as f32)
                .height(vk_state.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0)],
        );
        device.cmd_set_scissor(
            command_buffer,
            0,
            &[vk::Rect2D::default()
                .offset(vk::Offset2D::default().x(0).y(0))
                .extent(vk_state.extent)],
        );

        device.cmd_begin_rendering(
            command_buffer,
            &vk::RenderingInfo::default()
                .layer_count(1)
                .render_area(
                    vk::Rect2D::default()
                        .offset(vk::Offset2D::default().x(0).y(0))
                        .extent(vk_state.extent),
                )
                .color_attachments(&[vk::RenderingAttachmentInfo::default()
                    .image_view(vk_state.current_image().view)
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
                            vk_state
                                .resource_manager
                                .get_image_mut(&rendering_res.depth_buffer)
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

        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            rendering_res.pipeline_layout,
            0,
            &[vk_state.resource_manager.descriptor_set],
            &[],
        );

        let node_ids = match &rendering_res.gltf.scenes {
            Some(scenes) => {
                let scene_id = rendering_res.gltf.scene.unwrap_or(0) as usize;
                &scenes[scene_id].nodes
            }
            None => &(0..rendering_res.gltf.nodes.len() as i32).collect(),
        };

        for node_id in node_ids {
            let node = &rendering_res.gltf.nodes[*node_id as usize];
            let mesh_id = match node.mesh {
                Some(mesh) => mesh,
                None => continue,
            };
            for primitive_id in &rendering_res.gltf.meshes[mesh_id as usize].primitives {
                let primitive = &rendering_res.gltf.primitives[*primitive_id];
                let model = node.model_matrix();
                let mut push_constants = PushConstant {
                    model_view_projection: [0.0; 16],
                    model_data: primitive.model_data,
                };
                (view_projection * model)
                    .write_cols_to_slice(push_constants.model_view_projection.as_mut_slice());
                device.cmd_push_constants(
                    command_buffer,
                    rendering_res.pipeline_layout,
                    vk::ShaderStageFlags::ALL_GRAPHICS,
                    0,
                    bytemuck::bytes_of(&push_constants),
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    rendering_res.pipeline,
                );

                device.cmd_bind_index_buffer(
                    command_buffer,
                    primitive.indices.handle,
                    0,
                    primitive.index_type,
                );
                device.cmd_draw_indexed(command_buffer, primitive.indices_count, 1, 0, 0, 0);
            }
        }
        device.cmd_end_rendering(command_buffer);
    };

    vk_state.current_image().transition(
        command_buffer,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        vk::AccessFlags2::NONE,
        vk::ImageLayout::PRESENT_SRC_KHR,
    );

    vk_state.end_frame(command_buffer);

    if rendering_res.frame.is_multiple_of(100) {
        let elapsed = rendering_res.last_frame_time.elapsed().as_secs_f32();
        window.title = format!("Vulkan Rust, FPS: {}", 100.0 / elapsed);
        rendering_res.last_frame_time = Instant::now();
    }
    rendering_res.frame += 1
}
