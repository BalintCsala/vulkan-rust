use bevy::{ecs::bundle::Bundle, transform::components::Transform};

use crate::rendering::components::model::Model;

#[derive(Bundle)]
pub struct Renderable {
    pub model: Model,
    pub transform: Transform,
}
