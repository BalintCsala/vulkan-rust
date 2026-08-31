use bevy::ecs::component::Component;

use crate::rendering::resource_manager::ModelReference;

#[derive(Component, Clone)]
pub struct Model {
    pub model_ref: ModelReference,
}
