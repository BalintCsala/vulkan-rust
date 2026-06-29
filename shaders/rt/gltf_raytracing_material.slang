import lib.forward;
import lib.bindless;
import lib.material;
import common;

[vk::push_constant]
RTPushConstantsBase pushConstants;

[shader("closesthit")]
void chit(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
    uint instanceIndex = InstanceIndex();
    uint triangleId = PrimitiveIndex();
    float3 barycentrics = float3(1.0 - dot(attr.barycentrics.xy, 1.0), attr.barycentrics.xy);

    InstanceData instance = pushConstants.instances[instanceIndex];
    ModelData model = pushConstants.models[instance.modelId];

    // var normalTexture = getSampledImage<Texture2D<float4>>(model.normalTextureId);
    // var normalSampler = getSampler<SamplerState>(model.normalSamplerId);

    uint index0 = model.indices[triangleId * 3 + 0];
    uint index1 = model.indices[triangleId * 3 + 1];
    uint index2 = model.indices[triangleId * 3 + 2];

    float2 uv0 = model.texcoords0[index0];
    float2 uv1 = model.texcoords0[index1];
    float2 uv2 = model.texcoords0[index2];
    float2 uv = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;

    Material material;

    var baseColorTexture = getSampledImage<Texture2D<float4>>(model.baseColorTextureId);
    var baseColorSampler = getSampler<SamplerState>(model.baseColorSamplerId);
    material.baseColor = baseColorTexture.SampleLevel(baseColorSampler, uv, 0).rgb;

    float3 normal0 = model.normals[index0];
    float3 normal1 = model.normals[index1];
    float3 normal2 = model.normals[index2];
    material.normal = normal0 * barycentrics.x  +normal1 * barycentrics.y + normal2 * barycentrics.z;
    material.normal = normalize(mul(ObjectToWorld3x4(), float4(material.normal, 0.0)));

    if (model.metallicRoughnessTextureId >= 0) {
        var metallicRoughnessTexture = getSampledImage<Texture2D<float4>>(model.metallicRoughnessTextureId);
        var metallicRoughnessSampler = getSampler<SamplerState>(model.metallicRoughnessSamplerId);
        float2 metallicRoughness = metallicRoughnessTexture.SampleLevel(metallicRoughnessSampler, uv, 0).bg;
        material.metallic = metallicRoughness.x;
        material.roughness = metallicRoughness.y;
    } else {
        material.metallic = 1.0;
        material.roughness = 1.0;
    }

    if (model.emissiveTextureId >= 0) {
        var emissiveTexture = getSampledImage<Texture2D<float4>>(model.emissiveTextureId);
        var emissiveSampler = getSampler<SamplerState>(model.emissiveSamplerId);
        material.emissive = emissiveTexture.SampleLevel(emissiveSampler, uv, 0).xyz * model.emissiveFactor;
    } else {
        material.emissive = model.emissiveFactor;
    }

    payload.packedMaterial = material.pack();
    payload.dist = RayTCurrent();
}

[shader("anyhit")]
void ahit(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
    uint instanceIndex = InstanceIndex();
    uint triangleId = PrimitiveIndex();
    float3 barycentrics = float3(1.0 - dot(attr.barycentrics.xy, 1.0), attr.barycentrics.xy);

    InstanceData instance = pushConstants.instances[instanceIndex];
    ModelData model = pushConstants.models[instance.modelId];

    uint index0 = model.indices[triangleId * 3 + 0];
    uint index1 = model.indices[triangleId * 3 + 1];
    uint index2 = model.indices[triangleId * 3 + 2];

    float2 uv0 = model.texcoords0[index0];
    float2 uv1 = model.texcoords0[index1];
    float2 uv2 = model.texcoords0[index2];
    float2 uv = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;

    var baseColorTexture = getSampledImage<Texture2D<float4>>(model.baseColorTextureId);
    var baseColorSampler = getSampler<SamplerState>(model.baseColorSamplerId);
    float4 baseColor = baseColorTexture.SampleLevel(baseColorSampler, uv, 0);
    if (baseColor.a < 0.5) {
        IgnoreHit();
    }

    payload.dist = RayTCurrent();
}
