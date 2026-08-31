import lib.bindless;
import lib.forward;
import lib.barycentrics;
import lib.packing;
import lib.material;

struct PushConstants {
    float4x4 viewProjection;
    ModelData *models;
    InstanceData *instances;
    float2 resolution;
    int visibilityBufferId;
    int gbufferId;
};

[vk::push_constant]
PushConstants pushConstants;

[shader("compute")]
[numthreads(8, 8, 1)]
void cs(uint2 gid: SV_DispatchThreadID) {
    var visibilityBuffer = getStorageImage<RWTexture2D<uint>>(pushConstants.visibilityBufferId);
    var gbuffer = getStorageImage<RWTexture2D<uint4>>(pushConstants.gbufferId);

    var uniqueId = visibilityBuffer[gid];
    if (uniqueId == 0) {
        gbuffer[gid] = 0;
        return;
    }

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

    Material material;

    var fragmentData = model.getFragmentData(index0, index1, index2, position0, position1, position2, barycentrics);

    if (model.baseColorTextureId >= 0) {
        var baseColorTexture = getSampledImage<Texture2D<float4>>(model.baseColorTextureId);
        var baseColorSampler = getSampler<SamplerState>(model.baseColorSamplerId);

        var baseColorUV = fragmentData.uvs[model.baseColorTexcoordId];
        material.baseColor =
            baseColorTexture.SampleGrad(baseColorSampler, baseColorUV.value, baseColorUV.ddx, baseColorUV.ddy).rgb * model.baseColorFactor.rgb;
    } else {
        material.baseColor = model.baseColorFactor.rgb;
    }

    if (model.normalTextureId >= 0) {
        var normalTexture = getSampledImage<Texture2D<float4>>(model.normalTextureId);
        var normalSampler = getSampler<SamplerState>(model.normalSamplerId);

        var normalUV = fragmentData.uvs[model.normalTexcoordId];
        material.normal = normalTexture.SampleGrad(normalSampler, normalUV.value, normalUV.ddx, normalUV.ddy).xyz * 2.0 - 1.0;
        material.normal = mul(material.normal, fragmentData.tbn);
        material.normal = mul(instance.normalMatrix, material.normal);
    } else {
        material.normal = fragmentData.tbn[2];
    }

    if (model.metallicRoughnessTextureId >= 0) {
        var metallicRoughnessTexture = getSampledImage<Texture2D<float4>>(model.metallicRoughnessTextureId);
        var metallicRoughnessSampler = getSampler<SamplerState>(model.metallicRoughnessSamplerId);
        var metallicRoughnessUV = fragmentData.uvs[model.metallicRoughnessTexcoordId];
        float2 metallicRoughness = 
            metallicRoughnessTexture
                .SampleGrad(
                    metallicRoughnessSampler, metallicRoughnessUV.value, metallicRoughnessUV.ddx, metallicRoughnessUV.ddy)
                .bg;
        material.metallic = metallicRoughness.x;
        material.roughness = metallicRoughness.y;
    } else {
        material.metallic = 1.0;
        material.roughness = 1.0;
    }

    if (model.emissiveTextureId >= 0) {
        var emissiveUV = fragmentData.uvs[model.emissiveTexcoordId];
        var emissiveTexture = getSampledImage<Texture2D<float4>>(model.emissiveTextureId);
        var emissiveSampler = getSampler<SamplerState>(model.emissiveSamplerId);

        material.emissive =
            emissiveTexture.SampleGrad(emissiveSampler, emissiveUV.value, emissiveUV.ddx, emissiveUV.ddy).xyz *
                       model.emissiveFactor;
    } else {
        material.emissive = model.emissiveFactor;
    }

    gbuffer[gid] = material.pack();
}



