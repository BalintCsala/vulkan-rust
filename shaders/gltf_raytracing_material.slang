import lib.forward;
import lib.bindless;
import lib.material;
import lib.random;

struct PushConstants {
    uint64_t tlas;
    ModelData *models;
    InstanceData *instances;
    float4x4 viewProjectionInv;
    float3 cameraPosition;
    float2 resolution;
    int outputId;
}

[vk::push_constant]
PushConstants pushConstants;

[raypayload] 
struct Payload {
    float4 color : read(caller) : write(caller, closesthit, anyhit, miss);
    float3 rayOrigin : read(caller) : write(caller, closesthit);
    float3 rayDirection : read(caller) : write(caller, closesthit);
    float3 pdf : read(caller) : write(caller, closesthit);
    float3 radiance : read(caller) : write(caller, closesthit);
    float2 uv : read(closesthit, anyhit) : write(closesthit, anyhit);
    bool hit : read(caller) : write(caller, closesthit, miss);
    Random random : read(caller, closesthit) : write(caller, closesthit);
}

[shader("closesthit")]
void chit(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
    payload.hit = true;

    Material material;
    material.baseColor = payload.color.rgb;

    var instanceIndex = InstanceIndex();
    var triangleId = PrimitiveIndex();
    var barycentrics = attr.barycentrics;

    var instance = pushConstants.instances[instanceIndex];
    var model = pushConstants.models[instance.modelId];

    // var normalTexture = getSampledImage<Texture2D<float4>>(model.normalTextureId);
    // var normalSampler = getSampler<SamplerState>(model.normalSamplerId);

    var index0 = model.indices[triangleId * 3 + 0];
    var index1 = model.indices[triangleId * 3 + 1];
    var index2 = model.indices[triangleId * 3 + 2];

    var normal0 = model.normals[index0];
    var normal1 = model.normals[index1];
    var normal2 = model.normals[index2];
    material.normal = normal1 * barycentrics.x + normal2 * barycentrics.y + normal0 * (1.0 - barycentrics.x - barycentrics.y);

    var metallicRoughnessTexture = getSampledImage<Texture2D<float4>>(model.metallicRoughnessTextureId);
    var metallicRoughnessSampler = getSampler<SamplerState>(model.metallicRoughnessSamplerId);

    float2 metallicRoughness = metallicRoughnessTexture.SampleLevel(metallicRoughnessSampler, payload.uv, 0).bg;
    material.metallic = metallicRoughness.x;
    material.roughness = metallicRoughness.y;

    var emissiveTexture = getSampledImage<Texture2D<float4>>(model.emissiveTextureId);
    var emissiveSampler = getSampler<SamplerState>(model.emissiveSamplerId);
    material.emissive = emissiveTexture.SampleLevel(emissiveSampler, payload.uv, 0).xyz * model.emissiveFactor;

    material.f0 = 0.04;

    payload.rayOrigin = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.rayDirection = material.sampleCosineWeightedHemisphere(payload.random);

    payload.pdf = material.baseColor;
    payload.radiance = material.emissive;
}

[shader("anyhit")]
void ahit(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
    var instanceIndex = InstanceIndex();
    var triangleId = PrimitiveIndex();
    var barycentrics = attr.barycentrics;

    var instance = pushConstants.instances[instanceIndex];
    var model = pushConstants.models[instance.modelId];

    var index0 = model.indices[triangleId * 3 + 0];
    var index1 = model.indices[triangleId * 3 + 1];
    var index2 = model.indices[triangleId * 3 + 2];

    var uv0 = model.texcoords0[index0];
    var uv1 = model.texcoords0[index1];
    var uv2 = model.texcoords0[index2];

    payload.uv = uv1 * barycentrics.x + uv2 * barycentrics.y + uv0 * (1.0 - barycentrics.x - barycentrics.y);

    var baseColorTexture = getSampledImage<Texture2D<float4>>(model.baseColorTextureId);
    var baseColorSampler = getSampler<SamplerState>(model.baseColorSamplerId);

    payload.color = baseColorTexture.SampleLevel(baseColorSampler, payload.uv, 0);
    if (payload.color.a < 0.5) {
        IgnoreHit();
    }
}
