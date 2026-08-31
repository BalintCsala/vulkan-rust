pub mod types;

use anyhow::{Result, anyhow};
use bytemuck::Pod;
use std::{
    collections::HashMap,
    io::{Cursor, Read, Seek},
    sync::{Arc, Mutex},
    thread,
};

use ash::vk;
use bevy::{
    math::{Mat4, Quat, U16Vec4, Vec2, Vec3, Vec4},
    platform::collections::HashSet,
};
use image::{ImageFormat, ImageReader};

use crate::gltf::types::AlphaMode;
use renderer::{
    materials::standard_material::StandardMaterialData,
    mesh::{GeometryType, Mesh},
    resource_manager::{
        ImageReference, ImageSize, MaterialReference, MeshReference, ResourceManager,
        SamplerReference,
    },
};

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

fn read_u32<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn bytes_to_vec<T: Pod>(data: Vec<u8>) -> Vec<T> {
    data.chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

pub struct GltfMesh {
    indices: Vec<u32>,
    positions: Vec<Vec3>,
    normals: Option<Vec<Vec3>>,
    tangents: Option<Vec<Vec4>>,
    texcoords_0: Option<Vec<Vec2>>,
    texcoords_1: Option<Vec<Vec2>>,
    colors: Option<Vec<Vec4>>,
    joints: Option<Vec<U16Vec4>>,
    weights: Option<Vec<Vec4>>,
    geometry_type: GeometryType,
    name: String,
}

pub struct GltfModel {
    mesh: GltfMesh,
    material: StandardMaterialData,
}

impl GltfModel {
    pub fn upload(
        &self,
        resource_manager: &mut ResourceManager,
    ) -> (MeshReference, MaterialReference) {
        let mesh = resource_manager.upload_mesh(&self.mesh);
        let material = resource_manager.upload_material_data(&self.material);
        (mesh, material)
    }
}

impl Mesh for GltfMesh {
    fn indices(&self) -> &Vec<u32> {
        &self.indices
    }

    fn positions(&self) -> &Vec<Vec3> {
        &self.positions
    }

    fn normals(&self) -> Option<&Vec<Vec3>> {
        self.normals.as_ref()
    }

    fn tangents(&self) -> Option<&Vec<Vec4>> {
        self.tangents.as_ref()
    }

    fn texcoords_0(&self) -> Option<&Vec<Vec2>> {
        self.texcoords_0.as_ref()
    }

    fn texcoords_1(&self) -> Option<&Vec<Vec2>> {
        self.texcoords_1.as_ref()
    }

    fn colors(&self) -> Option<&Vec<Vec4>> {
        self.colors.as_ref()
    }

    fn joints(&self) -> Option<&Vec<U16Vec4>> {
        self.joints.as_ref()
    }

    fn weights(&self) -> Option<&Vec<Vec4>> {
        self.weights.as_ref()
    }

    fn geometry_type(&self) -> &GeometryType {
        &self.geometry_type
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub struct Gltf {
    pub primitives: Vec<Vec<GltfModel>>,
    pub scene: Option<usize>,
    pub scenes: Option<Vec<types::Scene>>,
    pub nodes: Vec<types::Node>,
}

impl Gltf {
    pub fn from_glb<R: Read + Seek>(
        resource_manager: &mut ResourceManager,
        reader: &mut R,
    ) -> Result<Self> {
        let magic = read_u32(reader)?;
        if magic != 0x46546C67 {
            return Err(anyhow!("Invalid file type for GLB file"));
        }

        let version = read_u32(reader)?;
        if version != 2 {
            return Err(anyhow!("Unsupported GLTF version"));
        }

        let _length = read_u32(reader);

        let mut info: Option<types::Info> = None;
        let mut bin_content = None;

        loop {
            let chunk_length = match read_u32(reader) {
                Ok(chunk_length) => chunk_length,
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
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
            None => return Err(anyhow!("Missing JSON chunk from GLB file")),
        };

        let bin = match bin_content {
            Some(bin) => bin,
            None => return Err(anyhow!("Missing BIN chunk from GLB file")),
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

        let default_sampler_ref = resource_manager.add_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .min_lod(0.0)
                .max_lod(6.0),
        );

        let mut sampler_lookup = HashMap::new();
        if let Some(samplers) = &info.samplers {
            samplers
                .iter()
                .enumerate()
                .for_each(|(sampler_id, sampler)| {
                    let mag_filter = match sampler.mag_filter {
                        Some(9728) => vk::Filter::NEAREST,
                        Some(9729) => vk::Filter::LINEAR,
                        None => vk::Filter::LINEAR,
                        _ => panic!(
                            "Unhandled mag_filter value: {}",
                            sampler.mag_filter.unwrap()
                        ),
                    };
                    let (min_filter, mipmap_mode) = match sampler.min_filter {
                        Some(9728) => (vk::Filter::NEAREST, vk::SamplerMipmapMode::LINEAR),
                        Some(9729) => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR),
                        Some(9984) => (vk::Filter::NEAREST, vk::SamplerMipmapMode::NEAREST),
                        Some(9985) => (vk::Filter::LINEAR, vk::SamplerMipmapMode::NEAREST),
                        Some(9986) => (vk::Filter::NEAREST, vk::SamplerMipmapMode::LINEAR),
                        Some(9987) => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR),
                        None => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR),
                        _ => panic!(
                            "Unhandled min_filter value: {}",
                            sampler.min_filter.unwrap()
                        ),
                    };
                    let address_mode_u = match sampler.wrap_s {
                        33071 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        33648 => vk::SamplerAddressMode::MIRRORED_REPEAT,
                        10497 => vk::SamplerAddressMode::REPEAT,
                        _ => panic!("Unhandled wrap_s value: {}", sampler.wrap_s),
                    };
                    let address_mode_v = match sampler.wrap_t {
                        33071 => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        33648 => vk::SamplerAddressMode::MIRRORED_REPEAT,
                        10497 => vk::SamplerAddressMode::REPEAT,
                        _ => panic!("Unhandled wrap_s value: {}", sampler.wrap_t),
                    };

                    let sampler_ref = resource_manager.add_sampler(
                        &vk::SamplerCreateInfo::default()
                            .mag_filter(mag_filter)
                            .min_filter(min_filter)
                            .mipmap_mode(mipmap_mode)
                            .address_mode_u(address_mode_u)
                            .address_mode_v(address_mode_v)
                            .min_lod(0.0)
                            .max_lod(6.0),
                    );
                    sampler_lookup.insert(sampler_id, sampler_ref);
                });
        }

        let mut texture_lookup = HashMap::new();

        if let Some(textures) = &info.textures
            && let Some(images) = &info.images
            && let Some(buffer_views) = &info.buffer_views
        {
            let decoded_images = Arc::new(Mutex::new(Vec::new()));
            let bin = Arc::new(&bin);
            thread::scope(|scope| {
                println!("Loading textures...");
                textures
                    .iter()
                    .enumerate()
                    .for_each(|(texture_id, texture)| {
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
                                let img =
                                    img.decode().expect("Failed to decode image").into_rgba8();
                                decoded_images.lock().unwrap().push((
                                    texture_id,
                                    texture.sampler,
                                    img,
                                    texture
                                        .name
                                        .clone()
                                        .unwrap_or(format!("Gltf texture {texture_id}").to_owned()),
                                ));
                            }
                        });
                    });
                println!("Finished loading textures");
            });

            let decoded_images = decoded_images.lock().unwrap();

            let mut uploads = decoded_images
                .iter()
                .map(|(texture_id, sampler_id, img, name)| {
                    let image_ref = resource_manager.create_empty_image(
                        ImageSize::Fixed(img.width(), img.height()),
                        if srgb_textures.contains(texture_id) {
                            vk::Format::R8G8B8A8_SRGB
                        } else {
                            vk::Format::R8G8B8A8_UNORM
                        },
                        if srgb_textures.contains(texture_id) {
                            vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::TRANSFER_SRC
                        } else {
                            vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_DST
                        },
                        u32::min(img.width(), img.height()).ilog2() + 1,
                        1,
                        name.clone()
                    );
                    texture_lookup.insert(
                        *texture_id,
                        (
                            image_ref,
                            *match sampler_id {
                                Some(sampler_id) => sampler_lookup
                                    .get(sampler_id)
                                    .expect("GLTF file references non-existent sampler with id {sampler_id}"),
                                None => &default_sampler_ref,
                            },
                        ),
                    );
                    (image_ref, img.iter().as_slice())
                })
                .collect();

            resource_manager.upload_image_data(&mut uploads);
            println!("Finished uploading images");
        }

        let mut primitives = Vec::new();
        if let Some(gltf_meshes) = &info.meshes {
            for mesh in gltf_meshes.iter() {
                let mut models = Vec::new();
                for (i, primitive) in mesh.primitives.iter().enumerate() {
                    let (mesh, material) = Self::load_primitive(
                        &texture_lookup,
                        &info,
                        primitive,
                        &bin,
                        format!(
                            "{} primitive #{i}",
                            mesh.name.as_ref().unwrap_or(&"Mesh".to_owned())
                        ),
                    )?;
                    models.push(GltfModel { mesh, material });
                }
                primitives.push(models);
            }
        }

        let nodes = match info.nodes {
            Some(nodes) => nodes,
            None => return Err(anyhow!("No scene nodes in GLTF file")),
        };

        Ok(Self {
            primitives,
            scene: info.scene,
            scenes: info.scenes,
            nodes,
        })
    }

    fn load_primitive(
        texture_lookup: &HashMap<usize, (ImageReference, SamplerReference)>,
        info: &types::Info,
        primitive: &types::Primitive,
        bin: &[u8],
        name: String,
    ) -> Result<(GltfMesh, StandardMaterialData)> {
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
            match name.as_str() {
                "POSITION" => {
                    positions = Some(bytes_to_vec(data));
                    positions_count = accessor.count;
                }
                "NORMAL" => normals = Some(bytes_to_vec(data)),
                "TANGENT" => tangents = Some(bytes_to_vec(data)),
                "TEXCOORD_0" => texcoords_0 = Some(bytes_to_vec(data)),
                "TEXCOORD_1" => texcoords_1 = Some(bytes_to_vec(data)),
                "COLOR_0" => colors = Some(bytes_to_vec(data)),
                "JOINTS_0" => joints = Some(bytes_to_vec(data)),
                "WEIGHTS_0" => weights = Some(bytes_to_vec(data)),
                _ => println!("Unhandled gltf attribute {name}"),
            };
        }

        let Some(positions) = positions else {
            return Err(anyhow!("Missing required attribute"));
        };

        let indices = match primitive.indices {
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

                bytes_to_vec(data)
            }
            None => Vec::from_iter(0..positions_count as u32),
        };

        let (base_color_texture, base_color_texcoord_id, base_color_sampler) = (|| {
            let texture = info.materials.as_ref()?[primitive.material?]
                .pbr_metallic_roughness
                .as_ref()?
                .base_color_texture
                .as_ref()?;

            let (image_ref, sampler_ref) = texture_lookup.get(&texture.index)?;
            Some((*image_ref, texture.tex_coord, *sampler_ref))
        })()
        .unwrap_or((-1, 0, 0));

        let base_color_factor = (|| {
            Some(
                info.materials.as_ref()?[primitive.material?]
                    .pbr_metallic_roughness
                    .as_ref()?
                    .base_color_factor,
            )
        })()
        .unwrap_or([1.0, 1.0, 1.0, 1.0]);

        let (normal_texture, normal_texcoord_id, normal_sampler) = (|| {
            let texture = info.materials.as_ref()?[primitive.material?]
                .normal_texture
                .as_ref()?;
            let (image_ref, sampler_ref) = texture_lookup.get(&texture.index)?;
            Some((*image_ref, texture.tex_coord, *sampler_ref))
        })()
        .unwrap_or((-1, 0, 0));

        let (
            metallic_roughness_texture,
            metallic_roughness_texcoord_id,
            metallic_roughness_sampler,
        ) = (|| {
            let texture = info.materials.as_ref()?[primitive.material?]
                .pbr_metallic_roughness
                .as_ref()?
                .metallic_roughness_texture
                .as_ref()?;
            let (image_ref, sampler_ref) = texture_lookup.get(&texture.index)?;
            Some((*image_ref, texture.tex_coord, *sampler_ref))
        })()
        .unwrap_or((-1, 0, 0));

        let (emissive_texture, emissive_texcoord_id, emissive_sampler) = (|| {
            let texture = info.materials.as_ref()?[primitive.material?]
                .emissive_texture
                .as_ref()?;
            let (image_ref, sampler_ref) = texture_lookup.get(&texture.index)?;
            Some((*image_ref, texture.tex_coord, *sampler_ref))
        })()
        .unwrap_or((-1, 0, 0));

        let emissive_factor =
            (|| Some(info.materials.as_ref()?[primitive.material?].emissive_factor))()
                .unwrap_or([0.0, 0.0, 0.0]);

        let geometry_type = (|| {
            Some(
                match info.materials.as_ref()?[primitive.material?].alpha_mode {
                    AlphaMode::Opaque => GeometryType::Opaque,
                    AlphaMode::Mask => GeometryType::Cutout,
                    AlphaMode::Blend => GeometryType::Translucent,
                },
            )
        })()
        .unwrap_or(GeometryType::Opaque);

        Ok((
            GltfMesh {
                positions,
                indices,
                normals,
                tangents,
                texcoords_0,
                texcoords_1,
                colors,
                joints,
                weights,
                geometry_type,
                name,
            },
            StandardMaterialData {
                base_color_factor,
                base_color_texture,
                base_color_texcoord_id,
                base_color_sampler,

                normal_texture,
                normal_texcoord_id,
                normal_sampler,

                metallic_roughness_texture,
                metallic_roughness_texcoord_id,
                metallic_roughness_sampler,

                emissive_texture,
                emissive_texcoord_id,
                emissive_sampler,
                emissive_factor,
            },
        ))
    }

    fn load_accessor_data(
        info: &types::Info,
        bin: &[u8],
        accessor: &types::Accessor,
    ) -> Result<Vec<u8>> {
        let component_byte_size: usize = match accessor.component_type {
            5120 => 1, // Signed byte
            5121 => 1, // Unsigned byte
            5122 => 2, // Signed short
            5123 => 2, // Unsigned short
            5125 => 4, // Unsigned int
            5126 => 4, // Float
            _ => return Err(anyhow!("Invalid accessor component type")),
        };

        let components_per_element = match accessor.element_type.as_str() {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT2" => 4,
            "MAT3" => 9,
            "MAT4" => 16,
            _ => return Err(anyhow!("Invalid accessor element type")),
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
