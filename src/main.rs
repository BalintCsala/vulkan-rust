use std::{
    collections::VecDeque,
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
    time::Instant,
};

use bevy::{
    MinimalPlugins,
    a11y::AccessibilityPlugin,
    app::{App, Startup, Update},
    ecs::{
        message::MessageReader,
        query::With,
        resource::Resource,
        system::{Commands, Res, ResMut, Single},
    },
    input::{
        ButtonInput, InputPlugin,
        keyboard::KeyCode,
        mouse::{MouseButton, MouseMotion},
    },
    math::{Mat4, Quat, Vec3},
    time::Time,
    transform::components::Transform,
    window::{Window, WindowPlugin, WindowResolution},
    winit::WinitPlugin,
};

use crate::{
    assets::gltf::Gltf,
    rendering::{
        components::camera::{Camera, CameraResolution},
        renderer_plugin::RendererPlugin,
        resource_manager::ResourceManager,
        vulkan_state::VulkanState,
        wrappers::device::Device,
    },
};

mod assets;
mod rendering;

const CAMERA_SPEED: f32 = 3.0;
const CAMERA_SENSITIVITY: f32 = 0.003;

#[derive(Resource)]
struct RenderingRes {
    device: Arc<Device>,
    last_frame_time: Instant,
    frame: u32,
}

impl Drop for RenderingRes {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
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
            RendererPlugin,
        ))
        .add_systems(Startup, (init_rendering, setup_scene))
        .add_systems(Update, (keyboard_input, mouse_input, update))
        .run();
}

fn setup_scene(mut commands: Commands) {
    commands.spawn((
        Transform {
            translation: Vec3::new(0.0, 0.0, 3.0),
            rotation: Quat::IDENTITY,
            ..Default::default()
        },
        Camera {
            fov_y: 90.0f32.to_radians(),
            z_near: 0.1,
            z_far: 200.0,
            resolution: CameraResolution::Scale(1.0),
        },
    ));
}

fn init_rendering(
    mut commands: Commands,
    vulkan_state: Res<VulkanState>,
    mut resource_manager: ResMut<ResourceManager>,
) {
    let scene_list = File::open("./assets/scene.txt").unwrap();
    for line in BufReader::new(scene_list).lines().map_while(Result::ok) {
        let gltf = Gltf::from_glb(
            &vulkan_state.device,
            &vulkan_state.allocator,
            &vulkan_state.debug_utils_device,
            &mut resource_manager,
            &mut File::open(format!("./assets/{line}")).unwrap(),
        )
        .unwrap();

        let scenes = gltf.scenes.as_ref().unwrap();
        let scene = &scenes[gltf.scene.unwrap_or(0)];

        let mut remaining_nodes: VecDeque<_> = scene
            .nodes
            .iter()
            .map(|node_id| (node_id, Mat4::IDENTITY))
            .collect();
        while let Some((node_id, parent_transform)) = remaining_nodes.pop_front() {
            let node = &gltf.nodes[*node_id];
            let transform = parent_transform * node.model_matrix();
            if let Some(children) = &node.children {
                remaining_nodes.extend(children.iter().map(|node_id| (node_id, transform)));
            }
            let mesh_id = match node.mesh {
                Some(mesh) => mesh,
                None => continue,
            };
            // TODO: Add a parent entity for the node
            for primitive_id in &gltf.meshes[mesh_id].primitives {
                let primitive = &gltf.primitives[*primitive_id];
                commands.spawn((Transform::from_matrix(transform), primitive.clone()));
            }
        }
    }

    commands.insert_resource(RenderingRes {
        device: vulkan_state.device.clone(),
        last_frame_time: Instant::now(),
        frame: 0,
    });
}

fn update(mut rendering_res: ResMut<RenderingRes>, mut window: Single<&mut Window>) {
    if rendering_res.frame.is_multiple_of(100) {
        let elapsed = rendering_res.last_frame_time.elapsed().as_secs_f32();
        window.title = format!("Vulkan Rust, FPS: {}", 100.0 / elapsed);
        rendering_res.last_frame_time = Instant::now();
    }
    rendering_res.frame += 1
}

fn keyboard_input(
    mut camera: Single<&mut Transform, With<Camera>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let x = camera.local_x();
    let y = camera.local_y();
    let z = camera.local_z();
    let mut speed = time.delta_secs() * CAMERA_SPEED;

    if keys.pressed(KeyCode::ShiftLeft) | keys.pressed(KeyCode::ControlLeft) {
        speed *= 2.0;
    }

    if keys.pressed(KeyCode::KeyA) {
        camera.translation -= x * speed;
    }
    if keys.pressed(KeyCode::KeyD) {
        camera.translation += x * speed;
    }
    if keys.pressed(KeyCode::KeyW) {
        camera.translation -= z * speed;
    }
    if keys.pressed(KeyCode::KeyS) {
        camera.translation += z * speed;
    }
    if keys.pressed(KeyCode::KeyQ) {
        camera.translation -= y * speed;
    }
    if keys.pressed(KeyCode::KeyE) {
        camera.translation += y * speed;
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
