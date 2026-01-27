use bevy::{ecs::component::Component, math::Mat4};

pub enum CameraResolution {
    Scale(f32),
    Exact(u32, u32),
}

#[derive(Component)]
pub struct Camera {
    pub fov_y: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub resolution: CameraResolution,
}

impl Camera {
    pub fn projection_matrix(&self, image_width: u32, image_height: u32) -> Mat4 {
        let (width, height) = match self.resolution {
            CameraResolution::Scale(scale) => {
                ((image_width as f32 * scale), (image_height as f32 * scale))
            }
            CameraResolution::Exact(width, height) => (width as f32, height as f32),
        };

        Mat4::perspective_rh_vk(self.fov_y, width / height, self.z_near, self.z_far)
    }
}
