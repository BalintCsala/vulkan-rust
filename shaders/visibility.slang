import lib.bindless;
import lib.forward;
import lib.materials.standard_material;

struct PushConstants {
    float4x4 viewProjection;
    InstanceData<StandardMaterial> *instances;
};

[vk::push_constant]
PushConstants pushConstants;

struct VSOut {
    float4 position : SV_Position;
    float2 uv;
    nointerpolation int textureId;
    nointerpolation int samplerId;
    nointerpolation int instanceId;
    nointerpolation uint vid;
};

[shader("vertex")]
VSOut vs(uint id: SV_VertexID, uint instanceId: SV_VulkanInstanceID) {
    VSOut output;
    let instance = pushConstants.instances[instanceId];
    let mesh    = instance.mesh;
    let material = instance.material;
    output.vid   = id;

    output.position = float4(mesh->positions[id], 1.0);
    output.position = mul(instance.modelMatrix, output.position);
    output.position = mul(pushConstants.viewProjection, output.position);

    output.textureId = material->baseColorTextureId;
    output.samplerId = material->baseColorSamplerId;
    float2 *texcoords = nullptr;
   switch (material->baseColorTexcoordId) {
        case 0:
            texcoords = mesh->texcoords0;
            break;
        case 1:
            texcoords = mesh->texcoords1;
            break;
    }
    if (texcoords != nullptr) {
        output.uv = texcoords[id];
    }

    output.instanceId = instanceId;

    return output;
}

[shader("fragment")]
uint fs(VSOut input, uint triangleId: SV_PrimitiveID) {
    if (input.textureId >= 0) {
        var texture = getSampledImage<Texture2D<float4>>(input.textureId);
        var sampler = getSampler<SamplerState>(input.samplerId);

        var color = texture.Sample(sampler, input.uv);
        if (color.a < 0.1) {
            discard;
        }
    }
    return encodeUniqueId(input.instanceId, triangleId);
}



