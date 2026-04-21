import lib.bindless;
import lib.forward;
import lib.barycentrics;
import lib.encoding;

struct PushConstants {
    float4x4 viewProjection;
    ModelData *models;
    InstanceData *instances;
    float2 resolution;
    int visibilityBufferId;
    int baseColorOutputId;
    int normalOutputId;
    int metallicRoughnessOutputId;
    int emissiveOutputId;
};

[vk::push_constant]
PushConstants pushConstants;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs(uint2 gid: SV_DispatchThreadID) {
    var visibilityBuffer = getStorageImage<RWTexture2D<uint>>(pushConstants.visibilityBufferId);

    var uniqueId = visibilityBuffer[gid];
    if (uniqueId == 0) {
        return;
    }

    var baseColorOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.baseColorOutputId);
    var normalOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.normalOutputId);
    var metallicRoughnessOutput = getStorageImage<RWTexture2D<float2>>(pushConstants.metallicRoughnessOutputId);
    var emissiveOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.emissiveOutputId);

    var instanceId = getInstanceIdFromUniqueId(uniqueId);
    var triangleId = getTriangleIdFromUniqueId(uniqueId);

    var instance = pushConstants.instances[instanceId];
    var model = pushConstants.models[instance.modelId];

    var index0 = model.indices[triangleId * 3 + 0];
    var index1 = model.indices[triangleId * 3 + 1];
    var index2 = model.indices[triangleId * 3 + 2];

    var position0 = model.positions[index0];
    var position1 = model.positions[index1];
    var position2 = model.positions[index2];

    var clip0 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(position0, 1.0)));
    var clip1 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(position1, 1.0)));
    var clip2 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(position2, 1.0)));

    float2 ndc = (float2(gid) + 0.5) / pushConstants.resolution * 2.0 - 1.0;

    Barycentrics barycentrics = Barycentrics.fromScreenPoints(clip0, clip1, clip2, ndc, pushConstants.resolution);

    var baseColorTexture = getSampledImage<Texture2D<float4>>(model.baseColorTextureId);
    var normalTexture = getSampledImage<Texture2D<float4>>(model.normalTextureId);
    var metallicRoughnessTexture = getSampledImage<Texture2D<float4>>(model.metallicRoughnessTextureId);
    var emissiveTexture = getSampledImage<Texture2D<float4>>(model.emissiveTextureId);

    var baseColorSampler = getSampler<SamplerState>(model.baseColorSamplerId);
    var normalSampler = getSampler<SamplerState>(model.normalSamplerId);
    var metallicRoughnessSampler = getSampler<SamplerState>(model.metallicRoughnessSamplerId);
    var emissiveSampler = getSampler<SamplerState>(model.emissiveSamplerId);

    var fragmentData = model.getFragmentData(index0, index1, index2, position0, position1, position2, barycentrics);
    fragmentData.tbn = mul(instance.normalMatrix, fragmentData.tbn);

    var baseColorUV = fragmentData.uvs[model.baseColorTexcoordId];
    baseColorOutput[gid] =
        baseColorTexture.SampleGrad(baseColorSampler, baseColorUV.value, baseColorUV.ddx, baseColorUV.ddy);

    var normalUV = fragmentData.uvs[model.normalTexcoordId];
    var normal = normalTexture.SampleGrad(normalSampler, normalUV.value, normalUV.ddx, normalUV.ddy).xyz * 2.0 - 1.0;
    normal = normalize(mul(normal, fragmentData.tbn));
    normalOutput[gid] = float4(octahedron::encode(normalize(fragmentData.tbn[2])), octahedron::encode(normal));

    var metallicRoughnessUV = fragmentData.uvs[model.metallicRoughnessTexcoordId];
    metallicRoughnessOutput[gid] =
        metallicRoughnessTexture
            .SampleGrad(
                metallicRoughnessSampler, metallicRoughnessUV.value, metallicRoughnessUV.ddx, metallicRoughnessUV.ddy)
            .bg;

    var emissiveUV = fragmentData.uvs[model.emissiveTexcoordId];
    emissiveOutput[gid] =
        float4(emissiveTexture.SampleGrad(emissiveSampler, emissiveUV.value, emissiveUV.ddx, emissiveUV.ddy).xyz *
                   model.emissiveFactor,
               1.0);
}



