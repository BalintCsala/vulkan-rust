pub mod types;

use std::{
    io::{Read, Seek},
    str::Utf8Error,
    sync::Arc,
};

use ash::{ext::debug_utils, vk};
use bevy::{
    ecs::system::Commands,
    math::{Mat4, Quat, Vec3},
};
use vk_mem::Allocator;

use crate::{
    assets::{
        asset_manager::AssetManager,
        model::{Model, ModelData},
    },
    rendering::buffer::Buffer,
};

#[derive(Debug)]
pub enum Error {
    IO(std::io::ErrorKind),
    Utf8,
    EndOfFile,
    InvalidFileType,
    InvalidVersion,
    MissingJSONChunk,
    MissingBinChunk,
    JSONParse,
    NoSceneNodes,
    MissingRequiredAttribute,
    InvalidAttribute,
    InvalidIndexType,
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        match value.kind() {
            std::io::ErrorKind::UnexpectedEof => Self::EndOfFile,
            kind => Self::IO(kind),
        }
    }
}

impl From<Utf8Error> for Error {
    fn from(_: Utf8Error) -> Self {
        Error::Utf8
    }
}

impl From<serde_json::Error> for Error {
    fn from(_: serde_json::Error) -> Self {
        Error::JSONParse
    }
}

impl types::Node {
    pub fn model_matrix(&self) -> Mat4 {
        if let Some(matrix) = self.matrix {
            return Mat4::from_cols_slice(&matrix);
        }
        Mat4::from_scale_rotation_translation(
            Vec3::from_slice(&self.scale),
            Quat::from_slice(&self.rotation),
            Vec3::from_slice(&self.translation),
        )
    }

    pub fn spawn(&self, mut commands: Commands) {}
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub struct Mesh {
    pub primitives: Vec<usize>,
}

pub struct Gltf {
    _buffers: Vec<Buffer>,
    pub primitives: Vec<Model>,
    pub meshes: Vec<Mesh>,
    pub scene: Option<u32>,
    pub scenes: Option<Vec<types::Scene>>,
    pub nodes: Vec<types::Node>,
}

impl Gltf {
    pub fn from_glb<R: Read + Seek>(
        device: &ash::Device,
        allocator: &Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
        asset_manager: &mut AssetManager,
        reader: &mut R,
    ) -> Result<Self, Error> {
        let magic = read_u32(reader)?;
        if magic != 0x46546C67 {
            return Err(Error::InvalidFileType);
        }

        let version = read_u32(reader)?;
        if version != 2 {
            return Err(Error::InvalidVersion);
        }

        let _length = read_u32(reader);

        let mut info: Option<types::Info> = None;
        let mut bin_content = None;

        loop {
            let chunk_length = match read_u32(reader) {
                Ok(chunk_length) => chunk_length,
                Err(Error::EndOfFile) => break,
                Err(e) => Err(e)?,
            };
            let chunk_type = read_u32(reader)?;
            let mut chunk_data = vec![0u8; chunk_length as usize];
            reader.read_exact(&mut chunk_data)?;
            match chunk_type {
                0x4E4F534A => {
                    // JSON
                    info = Some(serde_json::from_slice(chunk_data.as_slice())?);
                }
                0x004E4942 => {
                    // BIN
                    bin_content = Some(chunk_data);
                }
                _ => continue,
            }
        }

        let info = match info {
            Some(info) => info,
            None => return Err(Error::MissingJSONChunk),
        };

        let bin = match bin_content {
            Some(bin) => bin,
            None => return Err(Error::MissingBinChunk),
        };

        let mut meshes = Vec::new();
        let mut primitives = Vec::new();
        let mut buffers = Vec::new();
        if let Some(gltf_meshes) = &info.meshes {
            for (i, mesh) in gltf_meshes.iter().enumerate() {
                let mut mesh_primitives = Vec::new();
                for primitive in &mesh.primitives {
                    mesh_primitives.push(primitives.len());
                    primitives.push(Self::load_primitive(
                        device,
                        allocator,
                        debug_utils_device,
                        asset_manager,
                        &info,
                        primitive,
                        &bin,
                        &mut buffers,
                        mesh.name.as_ref().unwrap_or(&format!("Mesh {i}")),
                    )?);
                }
                meshes.push(Mesh {
                    primitives: mesh_primitives,
                });
            }
        }

        let nodes = match info.nodes {
            Some(nodes) => nodes,
            None => return Err(Error::NoSceneNodes),
        };

        Ok(Self {
            _buffers: buffers,
            primitives,
            meshes,
            scene: info.scene,
            scenes: info.scenes,
            nodes,
        })
    }

    fn load_primitive(
        device: &ash::Device,
        allocator: &Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
        asset_manager: &mut AssetManager,
        info: &types::Info,
        primitive: &types::Primitive,
        bin: &[u8],
        buffers: &mut Vec<Buffer>,
        mesh_name: &str,
    ) -> Result<Model, Error> {
        let mut positions = None;
        let mut normals = None;
        let mut tangents = None;
        let mut texcoords_0 = None;
        let mut texcoords_1 = None;
        let mut colors = None;
        let mut joints = None;
        let mut weights = None;

        let mut positions_count = 0;

        for (name, accessor_id) in &primitive.attributes {
            let primitive_name = format!("{mesh_name} {name}");
            let accessor = &info.accessors[*accessor_id as usize];
            let buffer = Self::load_buffer(
                device,
                allocator,
                debug_utils_device,
                info,
                bin,
                accessor,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                &primitive_name,
            )?;
            match name.as_str() {
                "POSITION" => {
                    positions = Some(buffer.address);
                    positions_count = accessor.count;
                }
                "NORMAL" => normals = Some(buffer.address),
                "TANGENT" => tangents = Some(buffer.address),
                "TEXCOORD_0" => texcoords_0 = Some(buffer.address),
                "TEXCOORD_1" => texcoords_1 = Some(buffer.address),
                "COLOR_0" => colors = Some(buffer.address),
                "JOINTS_0" => joints = Some(buffer.address),
                "WEIGHTS_0" => weights = Some(buffer.address),
                _ => println!("Unhandled gltf attribute {name}"),
            }
            buffers.push(buffer);
        }

        let positions = match positions {
            Some(positions) => positions,
            _ => return Err(Error::MissingRequiredAttribute),
        };
        let indices_name = format!("{mesh_name} indices");
        let (indices, indices_count, index_type) = match primitive.indices {
            Some(indices) => {
                let accessor = &info.accessors[indices as usize];
                println!("count: {}", accessor.count);
                (
                    Self::load_buffer(
                        device,
                        allocator,
                        debug_utils_device,
                        info,
                        bin,
                        accessor,
                        vk::BufferUsageFlags::INDEX_BUFFER,
                        &indices_name,
                    )?,
                    accessor.count,
                    match accessor.component_type {
                        5123 => vk::IndexType::UINT16,
                        5125 => vk::IndexType::UINT32,
                        _ => Err(Error::InvalidIndexType)?,
                    },
                )
            }
            None => {
                let mut indices = Buffer::new(
                    device.clone(),
                    allocator.clone(),
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    positions_count as u64,
                );
                indices.set_name(debug_utils_device, &indices_name);
                indices.write(Vec::from_iter(0..positions_count as u32).as_slice(), 0);
                (indices, positions_count, vk::IndexType::UINT32)
            }
        };

        let model_data = asset_manager.upload_model_data(ModelData {
            positions,
            normals: normals.unwrap_or(0),
            tangents: tangents.unwrap_or(0),
            texcoords_0: texcoords_0.unwrap_or(0),
            texcoords_1: texcoords_1.unwrap_or(0),
            colors: colors.unwrap_or(0),
            joints: joints.unwrap_or(0),
            weights: weights.unwrap_or(0),
        });
        Ok(Model {
            model_data,
            indices,
            indices_count: indices_count as u32,
            index_type,
        })
    }

    fn load_buffer(
        device: &ash::Device,
        allocator: &Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
        info: &types::Info,
        bin: &[u8],
        accessor: &types::Accessor,
        buffer_usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<Buffer, Error> {
        let component_byte_size: usize = match accessor.component_type {
            5120 => 1, // Signed byte
            5121 => 1, // Unsigned byte
            5122 => 2, // Signed short
            5123 => 2, // Unsigned short
            5125 => 4, // Unsigned int
            5126 => 4, // Float
            _ => return Err(Error::InvalidAttribute),
        };

        let components_per_element = match accessor.element_type.as_str() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT2" => 4,
            "MAT3" => 9,
            "MAT4" => 16,
            _ => return Err(Error::InvalidAttribute),
        };

        let bytes_per_element = components_per_element * component_byte_size;
        let byte_length = (accessor.count as usize) * bytes_per_element;

        let data = match accessor.buffer_view {
            Some(buffer_view_id) => {
                let buffer_view = &info.buffer_views.as_ref().unwrap()[buffer_view_id as usize];
                let offset = (accessor.byte_offset + buffer_view.byte_offset) as usize;
                let stride = match buffer_view.byte_stride {
                    Some(stride) => stride as usize,
                    None => components_per_element * component_byte_size,
                };

                if stride == bytes_per_element {
                    bin[offset..offset + byte_length].to_vec()
                } else {
                    let mut data = vec![0u8; byte_length];
                    for element_id in 0..accessor.count as usize {
                        let src_start = offset + element_id * stride;
                        let src_end = src_start + bytes_per_element;
                        let dst_start = element_id * bytes_per_element;
                        let dst_end = dst_start + bytes_per_element;
                        data[dst_start..dst_end].copy_from_slice(&bin[src_start..src_end]);
                    }
                    data
                }
            }
            None => vec![0u8; byte_length],
        };

        let mut buffer = Buffer::new(
            device.clone(),
            allocator.clone(),
            buffer_usage,
            byte_length as u64,
        );
        buffer.set_name(debug_utils_device, name);
        buffer.write(&data, 0);
        Ok(buffer)
    }
}
