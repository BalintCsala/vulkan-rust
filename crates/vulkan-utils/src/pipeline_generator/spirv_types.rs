use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};

pub fn to_snake_case(input: &str) -> String {
    let mut name = String::new();

    for character in input.chars() {
        match character {
            'a'..='z' => name.push(character),
            'A'..='Z' => {
                if !name.is_empty() {
                    name.push('_');
                }
                name.push(character.to_ascii_lowercase());
            }
            _ => panic!("Invalid character in name '{}': {}", name, character),
        }
    }

    name
}

fn create_padding(size: u32, name: &str) -> TokenStream {
    let ident = format_ident!("{}", name);
    match size {
        1 => quote! { pub #ident: u8 },
        2 => quote! { pub #ident: u16 },
        4 => quote! { pub #ident: u32 },
        _ => panic!("Can't pad by {} bytes", size),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Field {
    name: String,
    #[serde(rename = "type")]
    ty: VariableType,
}

impl Field {
    pub fn to_code(&self) -> TokenStream {
        let name = format_ident!("{}", to_snake_case(&self.name));
        let ty = self.ty.to_code(None);
        quote! { pub #name : #ty }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub(crate) enum ScalarType {
    Float32,
    Uint32,
    Int32,
    Uint64,
}

impl ScalarType {
    fn to_code(&self) -> TokenStream {
        match self {
            ScalarType::Float32 => quote! { f32 },
            ScalarType::Uint32 => quote! { u32 },
            ScalarType::Int32 => quote! { i32 },
            ScalarType::Uint64 => quote! { u64 },
        }
    }

    fn size(&self) -> u32 {
        match self {
            ScalarType::Float32 => 4,
            ScalarType::Uint32 => 4,
            ScalarType::Int32 => 4,
            ScalarType::Uint64 => 8,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub(crate) enum VariableType {
    #[serde(rename_all = "camelCase")]
    Scalar {
        scalar_type: ScalarType,
    },
    #[serde(rename_all = "camelCase")]
    Pointer {
        value_type: String,
    },
    #[serde(rename_all = "camelCase")]
    Array {
        element_count: usize,
        element_type: Box<VariableType>,
    },
    #[serde(rename_all = "camelCase")]
    Vector {
        element_count: usize,
        element_type: Box<VariableType>,
    },
    #[serde(rename_all = "camelCase")]
    Matrix {
        row_count: usize,
        column_count: usize,
        element_type: Box<VariableType>,
    },
    Struct {
        name: String,
        fields: Vec<Field>,
    },
    #[serde(rename_all = "camelCase")]
    ConstantBuffer {
        element_type: Box<VariableType>,
    },
    #[serde(rename = "DynamicResource")]
    DynamicResource,
}

impl VariableType {
    pub fn to_code(&self, name_override: Option<String>) -> TokenStream {
        match self {
            VariableType::Scalar { scalar_type } => scalar_type.to_code(),
            VariableType::Pointer { value_type: _ } => quote! { vk::DeviceAddress },
            VariableType::Array {
                element_count,
                element_type,
            } => {
                let element_type = element_type.to_code(None);
                quote! { [ #element_type; #element_count ] }
            }
            VariableType::Vector {
                element_count,
                element_type,
            } => {
                let element_type = element_type.to_code(None);
                quote! { [ #element_type; #element_count ] }
            }
            VariableType::Matrix {
                row_count,
                column_count,
                element_type,
            } => {
                let element_type = element_type.to_code(None);
                let total_count = column_count * row_count;
                quote! { [ #element_type; #total_count ] }
            }
            VariableType::Struct { fields, name } => {
                let name = format_ident!("{}", name_override.as_ref().unwrap_or(name));
                let mut fields_code = Vec::new();
                let mut offset = 0u32;
                let mut pad_count = 0;
                let mut alignment = 0;
                for field in fields {
                    let required_offset = offset.next_multiple_of(field.ty.alignment());
                    if required_offset != offset {
                        fields_code.push(create_padding(
                            required_offset - offset,
                            &format!("_pad{}", pad_count),
                        ));
                        pad_count += 1;
                    }
                    fields_code.push(field.to_code());
                    offset = required_offset + field.ty.size();
                    alignment = alignment.max(field.ty.alignment());
                }
                let padded_size = offset.next_multiple_of(alignment);
                if padded_size != offset {
                    fields_code.push(create_padding(
                        padded_size - offset,
                        &format!("_pad{}", pad_count),
                    ));
                }

                quote! {
                    struct #name {
                        #(#fields_code),*
                    }
                }
            }
            VariableType::ConstantBuffer { element_type: _ } => {
                panic!("Cannot emit constant buffer as a type")
            }
            VariableType::DynamicResource => {
                panic!("Cannot emit dynamic resource as a type")
            }
        }
    }

    pub fn alignment(&self) -> u32 {
        match self {
            VariableType::Scalar { scalar_type } => scalar_type.size(),
            VariableType::Pointer { value_type: _ } => 8,
            VariableType::Array {
                element_count: _,
                element_type,
            } => element_type.alignment(),
            VariableType::Vector {
                element_count: _,
                element_type,
            } => element_type.alignment(),
            VariableType::Matrix {
                row_count: _,
                column_count: _,
                element_type,
            } => element_type.alignment(),
            VariableType::Struct { name: _, fields } => fields
                .iter()
                .map(|field| field.ty.alignment())
                .max()
                .unwrap_or(0),
            VariableType::ConstantBuffer { element_type: _ } => 0,
            VariableType::DynamicResource => 0,
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            VariableType::Scalar { scalar_type } => scalar_type.size(),
            VariableType::Pointer { value_type: _ } => 8,
            VariableType::Array {
                element_count,
                element_type,
            } => *element_count as u32 * element_type.size(),
            VariableType::Vector {
                element_count,
                element_type,
            } => *element_count as u32 * element_type.size(),
            VariableType::Matrix {
                row_count,
                column_count,
                element_type,
            } => *row_count as u32 * *column_count as u32 * element_type.size(),
            VariableType::Struct { name: _, fields } => {
                let mut size = 0u32;
                for field in fields {
                    size = size.next_multiple_of(field.ty.alignment());
                    size += field.ty.size();
                }
                size
            }
            VariableType::ConstantBuffer { element_type: _ } => 0,
            VariableType::DynamicResource => 0,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub(crate) enum ParameterBinding {
    PushConstantBuffer,
    DescriptorTableSlot,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct Parameter {
    name: String,
    binding: ParameterBinding,
    #[serde(rename = "type")]
    pub ty: VariableType,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct SpirvReflection {
    parameters: Vec<Parameter>,
}

impl SpirvReflection {
    pub fn get_push_constants_type(&self) -> Option<&VariableType> {
        if let VariableType::ConstantBuffer { element_type } = &self
            .parameters
            .iter()
            .find(|parameter| matches!(parameter.binding, ParameterBinding::PushConstantBuffer))?
            .ty
        {
            Some(element_type)
        } else {
            panic!("Invalid push constant definition")
        }
    }
}
