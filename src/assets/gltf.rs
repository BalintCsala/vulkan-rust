pub mod types;

use std::{
    collections::HashMap,
    io::{Cursor, Read, Seek},
    str::Utf8Error,
    sync::{Arc, Mutex},
    thread,
};

use ash::{ext::debug_utils, vk};
use bevy::{
    math::{Mat4, Quat, Vec3},
    platform::collections::HashSet,
};
use image::{ImageError, ImageFormat, ImageReader};

use crate::{
    assets::model::{Model, ModelData},
    rendering::{
        buffer::Buffer,
        resource_manager::{ImageReference, ResourceManager},
        wrappers::{allocator::Allocator, device::Device},
    },
};

#[derive(Debug)]
pub enum Error {
    IO,
    Utf8,
    EndOfFile,
    InvalidFileType,
    UnsupportedVersion,
    MissingJSONChunk,
    MissingBinChunk,
    JSONParse(serde_json::Error),
    NoSceneNodes,
    MissingRequiredAttribute,
    InvalidAttribute,
    InvalidIndexType,
    ImageError,
}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        match value.kind() {
            std::io::ErrorKind::UnexpectedEof => Self::EndOfFile,
            _ => Self::IO,
        }
    }
}

impl From<Utf8Error> for Error {
    fn from(_: Utf8Error) -> Self {
        Error::Utf8
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::JSONParse(err)
    }
}

impl From<ImageError> for Error {
    fn from(_value: ImageError) -> Self {
        Error::ImageError
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
    pub scene: Option<usize>,
    pub scenes: Option<Vec<types::Scene>>,
    pub nodes: Vec<types::Node>,
}

impl Gltf {
    pub fn from_glb<R: Read + Seek>(
        device: &Arc<Device>,
        allocator: &Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
        resource_manager: &mut ResourceManager,
        reader: &mut R,
    ) -> Result<Self, Error> {
        let magic = read_u32(reader)?;
        if magic != 0x46546C67 {
            return Err(Error::InvalidFileType);
        }

        let version = read_u32(reader)?;
        if version != 2 {
            return Err(Error::UnsupportedVersion);
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
                    let result = serde_json::from_slice(chunk_data.as_slice());
                    if let Err(err) = &result {
                        eprintln!("Json parse error:");
                        eprintln!(
                            "{}",
                            String::from_utf8(
                                chunk_data[err.column() - 15..err.column() + 50].to_vec()
                            )
                            .unwrap()
                        );
                        eprintln!("{}^", " ".repeat(15));
                    }
                    info = Some(result?);
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

        let mut srgb_textures = HashSet::new();
        if let Some(nodes) = &info.nodes
            && let Some(meshes) = &info.meshes
            && let Some(materials) = &info.materials
        {
            for node in nodes {
                if let Some(mesh) = node.mesh {
                    for primitive in &meshes[mesh].primitives {
                        if let Some(material) = primitive.material
                            && let Some(pbr_metallic_roughness) =
                                &materials[material].pbr_metallic_roughness
                            && let Some(base_color_texture) =
                                &pbr_metallic_roughness.base_color_texture
                        {
                            srgb_textures.insert(base_color_texture.index);
                        }
                    }
                }
            }
        }

        let mut image_lookup = HashMap::new();

        if let Some(textures) = &info.textures
            && let Some(images) = &info.images
            && let Some(buffer_views) = &info.buffer_views
        {
            let decoded_images = Arc::new(Mutex::new(Vec::new()));
            let bin = Arc::new(&bin);
            thread::scope(|scope| {
                println!("Loading textures...");
                textures.iter().for_each(|texture| {
                    let decoded_images = decoded_images.clone();
                    let bin = bin.clone();
                    scope.spawn(move || {
                        if let Some(source) = texture.source {
                            let image = &images[source];
                            let buffer_view = &buffer_views[image.buffer_view.unwrap()];
                            let offset = buffer_view.byte_offset;
                            let data = bin[offset..offset + buffer_view.byte_length].to_vec();
                            let mime_type = image.mime_type.as_ref().unwrap();
                            let format = match mime_type.as_str() {
                                "image/jpeg" => ImageFormat::Jpeg,
                                "image/png" => ImageFormat::Png,
                                _ => panic!("Unrecognized image format: {}", mime_type),
                            };
                            let mut img = ImageReader::new(Cursor::new(data));
                            img.set_format(format);
                            let img = img.decode().expect("Failed to decode image").into_rgba8();
                            decoded_images.lock().unwrap().push((source, img));
                        }
                    });
                });
                println!("Finished loading textures");
            });

            let decoded_images = decoded_images.lock().unwrap();

            let uploads = decoded_images
                .iter()
                .enumerate()
                .map(|(i, (source, img))| {
                    let reference = resource_manager.create_empty_image(
                        crate::rendering::resource_manager::ImageSize::Fixed(
                            img.width(),
                            img.height(),
                        ),
                        if srgb_textures.contains(source) {
                            vk::Format::R8G8B8A8_SRGB
                        } else {
                            vk::Format::R8G8B8A8_UNORM
                        },
                        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                        1,
                        1,
                        format!("Gltf texture #{i}"),
                    );
                    image_lookup.insert(*source, reference);
                    (reference, img.iter().as_slice())
                })
                .collect::<Vec<_>>();

            resource_manager.upload_image_data(uploads.as_slice());
            resource_manager.flush_staging();
            println!("Finished uploading images");
        }

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
                        resource_manager,
                        &image_lookup,
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
        device: &Arc<Device>,
        allocator: &Arc<Allocator>,
        debug_utils_device: &debug_utils::Device,
        resource_manager: &mut ResourceManager,
        image_lookup: &HashMap<usize, ImageReference>,
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
            let accessor = &info.accessors[*accessor_id];
            let data = Self::load_accessor_data(info, bin, accessor)?;

            let mut buffer = Buffer::new(
                device,
                allocator.clone(),
                debug_utils_device,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                data.len() as u64,
                &format!("{mesh_name} {name}"),
            );
            buffer.write(&data, 0);

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
        let (indices, index_count) = match primitive.indices {
            Some(indices) => {
                let accessor = &info.accessors[indices];
                let mut data = Self::load_accessor_data(info, bin, accessor)?;
                if accessor.component_type == 5123 {
                    // UINT16
                    let mut inflated = Vec::with_capacity(data.len() * 2);
                    data.chunks(2).for_each(|pair| {
                        inflated.extend_from_slice(pair);
                        inflated.push(0);
                        inflated.push(0);
                    });
                    data = inflated;
                }

                let mut indices = Buffer::new(
                    device,
                    allocator.clone(),
                    debug_utils_device,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
                    data.len() as u64,
                    &indices_name,
                );
                indices.write(&data, 0);

                (indices, accessor.count)
            }
            None => {
                let mut indices = Buffer::new(
                    device,
                    allocator.clone(),
                    debug_utils_device,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    positions_count as u64,
                    &indices_name,
                );
                indices.write(Vec::from_iter(0..positions_count as u32).as_slice(), 0);
                (indices, positions_count)
            }
        };

        let (base_color_texture_id, base_color_texcoord_id, base_color_sampler_id) = match primitive
            .material
        {
            Some(material) => match &info.materials.as_ref().unwrap()[material]
                .pbr_metallic_roughness
            {
                Some(pbr_metallic_roughness) => match &pbr_metallic_roughness.base_color_texture {
                    Some(base_color_texture) => (
                        *image_lookup.get(&base_color_texture.index).unwrap_or(&-1),
                        base_color_texture.tex_coord,
                        0,
                    ),
                    None => (-1, 0, 0),
                },
                None => (-1, 0, 0),
            },
            None => (-1, 0, 0),
        };

        let model_ref = resource_manager.upload_model(
            ModelData {
                positions,
                indices: indices.address,
                normals: normals.unwrap_or(0),
                tangents: tangents.unwrap_or(0),
                texcoords_0: texcoords_0.unwrap_or(0),
                texcoords_1: texcoords_1.unwrap_or(0),
                colors: colors.unwrap_or(0),
                joints: joints.unwrap_or(0),
                weights: weights.unwrap_or(0),
                base_color_texture_id,
                base_color_texcoord_id,
                base_color_sampler_id,
            },
            indices,
            index_count as u32,
        );
        Ok(Model { model_ref })
    }

    fn load_accessor_data(
        info: &types::Info,
        bin: &[u8],
        accessor: &types::Accessor,
    ) -> Result<Vec<u8>, Error> {
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
        let byte_length = accessor.count * bytes_per_element;

        Ok(match accessor.buffer_view {
            Some(buffer_view_id) => {
                let buffer_view = &info.buffer_views.as_ref().unwrap()[buffer_view_id];
                let offset = accessor.byte_offset + buffer_view.byte_offset;
                let stride = match buffer_view.byte_stride {
                    Some(stride) => stride,
                    None => components_per_element * component_byte_size,
                };

                if stride == bytes_per_element {
                    bin[offset..offset + byte_length].to_vec()
                } else {
                    let mut data = vec![0u8; byte_length];
                    for element_id in 0..accessor.count {
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
        })
    }
}
