# Vulkan renderer

Written in Rust, using the entity component system of Bevy and Vulkan 1.3.

## Current state

- Able to load GLTF models
- Renders objects to a visibility buffer
- Bindless system, dynamic rendering and extensive use of buffer device addresses

## Future plans

- Three-phase rendering: Visibility buffer, material buffer, lighting evaluation
- Fully GPU driven culling and rendering
