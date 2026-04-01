import lib.bindless;
import lib.forward;

struct PushConstants {
    float4x4 viewProjection;
    ModelData *models;
    InstanceData *instanceData;
};

[vk::push_constant]
PushConstants pushConstants;

struct VSOut {
    float4 position : SV_Position;
    float2 uv;
    nointerpolation uint textureId;
    nointerpolation uint samplerId;
    nointerpolation uint instanceId;
    nointerpolation uint vid;
};

[shader("vertex")]
VSOut vs(uint id: SV_VertexID, uint instanceId: SV_VulkanInstanceID) {
    VSOut output;
    var instance = pushConstants.instanceData[instanceId];
    var model    = pushConstants.models[instance.modelId];
    output.vid   = id;

    output.position = float4(model.positions[id], 1.0);
    output.position = mul(instance.modelMatrix, output.position);
    output.position = mul(pushConstants.viewProjection, output.position);

    output.textureId = model.baseColorTextureId;
    output.samplerId = model.baseColorSamplerId;
    float2 *texcoords;
    switch (model.baseColorTexcoordId) {
        case 0:
            texcoords = model.texcoords0;
            break;
        case 1:
            texcoords = model.texcoords1;
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
    if (input.textureId != 0xFFFF) {
        var texture = getSampledImage<Texture2D<float4>>(input.textureId);
        var sampler = getSampler<SamplerState>(input.samplerId);

        var color = texture.Sample(sampler, input.uv);
        if (color.a < 0.1) {
            discard;
        }
    }
    return encodeUniqueId(input.instanceId, triangleId);
}



