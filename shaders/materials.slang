import lib.bindless;
import lib.forward;
import lib.barycentrics;
import lib.packing;
import lib.material;
import lib.materials.standard_material;

struct PushConstants {
    float4x4 viewProjection;
    InstanceData<StandardMaterial> *instances;
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
    var mesh = instance.mesh;

    var index0 = mesh->indices[triangleId * 3 + 0];
    var index1 = mesh->indices[triangleId * 3 + 1];
    var index2 = mesh->indices[triangleId * 3 + 2];

    var position0 = mesh->positions[index0];
    var position1 = mesh->positions[index1];
    var position2 = mesh->positions[index2];

    var clip0 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(position0, 1.0)));
    var clip1 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(position1, 1.0)));
    var clip2 = mul(pushConstants.viewProjection, mul(instance.modelMatrix, float4(position2, 1.0)));

    float2 ndc = (float2(gid) + 0.5) / pushConstants.resolution * 2.0 - 1.0;

    Barycentrics barycentrics = Barycentrics.fromScreenPoints(clip0, clip1, clip2, ndc, pushConstants.resolution);

    Material material;

    var fragmentData = mesh->getFragmentData(index0, index1, index2, position0, position1, position2, barycentrics);

    if (instance.material->baseColorTextureId >= 0) {
        var baseColorTexture = getSampledImage<Texture2D<float4>>(instance.material->baseColorTextureId);
        var baseColorSampler = getSampler<SamplerState>(instance.material->baseColorSamplerId);

        var baseColorUV = fragmentData.uvs[instance.material->baseColorTexcoordId];
        material.baseColor =
            baseColorTexture.SampleGrad(baseColorSampler, baseColorUV.value, baseColorUV.ddx, baseColorUV.ddy).rgb * instance.material->baseColorFactor.rgb;
    } else {
        material.baseColor = instance.material->baseColorFactor.rgb;
    }

    if (instance.material->normalTextureId >= 0) {
        var normalTexture = getSampledImage<Texture2D<float4>>(instance.material->normalTextureId);
        var normalSampler = getSampler<SamplerState>(instance.material->normalSamplerId);

        var normalUV = fragmentData.uvs[instance.material->normalTexcoordId];
        material.normal = normalTexture.SampleGrad(normalSampler, normalUV.value, normalUV.ddx, normalUV.ddy).xyz * 2.0 - 1.0;
        material.normal = mul(material.normal, fragmentData.tbn);
        material.normal = mul(instance.normalMatrix, material.normal);
    } else {
        material.normal = fragmentData.tbn[2];
    }

    if (instance.material->metallicRoughnessTextureId >= 0) {
        var metallicRoughnessTexture = getSampledImage<Texture2D<float4>>(instance.material->metallicRoughnessTextureId);
        var metallicRoughnessSampler = getSampler<SamplerState>(instance.material->metallicRoughnessSamplerId);
        var metallicRoughnessUV = fragmentData.uvs[instance.material->metallicRoughnessTexcoordId];
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

    if (instance.material->emissiveTextureId >= 0) {
        var emissiveUV = fragmentData.uvs[instance.material->emissiveTexcoordId];
        var emissiveTexture = getSampledImage<Texture2D<float4>>(instance.material->emissiveTextureId);
        var emissiveSampler = getSampler<SamplerState>(instance.material->emissiveSamplerId);

        material.emissive =
            emissiveTexture.SampleGrad(emissiveSampler, emissiveUV.value, emissiveUV.ddx, emissiveUV.ddy).xyz *
                       instance.material->emissiveFactor;
    } else {
        material.emissive = instance.material->emissiveFactor;
    }

    gbuffer[gid] = material.pack();
}



