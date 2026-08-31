use bevy::{ecs::bundle::Bundle, transform::components::Transform};

use crate::resource_manager::{MaterialReference, MeshReference};

#[derive(Bundle)]
pub struct Renderable {
    pub mesh: MeshReference,
    pub material: MaterialReference,
    pub transform: Transform,
}
