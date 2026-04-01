import lib.bindless;
import lib.forward;
import lib.barycentrics;

struct PushConstants {
    float4x4 viewProjection;
    ModelData *models;
    InstanceData *instances;
    float2 resolution;
    int visibilityBufferId;
    int outputId;
};

[vk::push_constant]
PushConstants pushConstants;

// Temp
float hash1(uint x) {
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return (float) (x & 0xFFFFFF) / 16777215.0;
}

float3 hashColor(uint seed) {
    float r = hash1(seed);
    float g = hash1(seed + 0x9e3779b9);
    float b = hash1(seed + 0x243f6a88);

    return float3(r, g, b);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void cs(uint2 gid: SV_DispatchThreadID) {
    var visibilityBuffer = getStorageImage<RWTexture2D<uint>>(pushConstants.visibilityBufferId);
    var output           = getStorageImage<RWTexture2D<float4>>(pushConstants.outputId);

    var uniqueId = visibilityBuffer[gid];
    if (uniqueId == 0) {
        return;
    }

    var instanceId = getInstanceIdFromUniqueId(uniqueId);
    var triangleId = getTriangleIdFromUniqueId(uniqueId);

    var instance = pushConstants.instances[instanceId];
    var model    = pushConstants.models[instance.modelId];

    var index0 = model.indices[triangleId * 3 + 0];
    var index1 = model.indices[triangleId * 3 + 1];
    var index2 = model.indices[triangleId * 3 + 2];

    var position0 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(model.positions[index0], 1.0)));
    var position1 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(model.positions[index1], 1.0)));
    var position2 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(model.positions[index2], 1.0)));

    float2 ndc = (float2(gid) + 0.5) / pushConstants.resolution * 2.0 - 1.0;

    Barycentrics barycentrics =
        Barycentrics.fromScreenPoints(position0, position1, position2, ndc, pushConstants.resolution);

    var texture = getSampledImage<Texture2D<float4>>(model.baseColorTextureId);
    var sampler = getSampler<SamplerState>(0);

    var uv0 = model.texcoords0[index0];
    var uv1 = model.texcoords0[index1];
    var uv2 = model.texcoords0[index2];

    var interpolated = barycentrics.interpolate(uv0, uv1, uv2);

    output[gid] = texture.SampleGrad(sampler, interpolated.value, interpolated.ddx, interpolated.ddy);
}



