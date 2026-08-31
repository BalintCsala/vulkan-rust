import lib.forward;
import lib.bindless;
import lib.material;
import lib.materials.standard_material;
import common;

[vk::push_constant]
RTPushConstantsBase<StandardMaterial> pushConstants;

[shader("closesthit")]
void chit(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
    uint triangleId = PrimitiveIndex();
    float3 barycentrics = float3(1.0 - dot(attr.barycentrics.xy, 1.0), attr.barycentrics.xy);

    let instance = pushConstants.instances[InstanceIndex()];
    let mesh = instance.mesh;

    // var normalTexture = getSampledImage<Texture2D<float4>>(mesh->normalTextureId);
    // var normalSampler = getSampler<SamplerState>(mesh->normalSamplerId);

    uint index0 = mesh->indices[triangleId * 3 + 0];
    uint index1 = mesh->indices[triangleId * 3 + 1];
    uint index2 = mesh->indices[triangleId * 3 + 2];

    float2 uv0 = mesh->texcoords0[index0];
    float2 uv1 = mesh->texcoords0[index1];
    float2 uv2 = mesh->texcoords0[index2];
    float2 uv = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;

    Material material;

    var baseColorTexture = getSampledImage<Texture2D<float4>>(instance.material->baseColorTextureId);
    var baseColorSampler = getSampler<SamplerState>(instance.material->baseColorSamplerId);
    material.baseColor = baseColorTexture.SampleLevel(baseColorSampler, uv, 0).rgb;

    float3 normal0 = mesh->normals[index0];
    float3 normal1 = mesh->normals[index1];
    float3 normal2 = mesh->normals[index2];
    material.normal = normal0 * barycentrics.x  +normal1 * barycentrics.y + normal2 * barycentrics.z;
    material.normal = normalize(mul(ObjectToWorld3x4(), float4(material.normal, 0.0)));

    if (instance.material->metallicRoughnessTextureId >= 0) {
        var metallicRoughnessTexture = getSampledImage<Texture2D<float4>>(instance.material->metallicRoughnessTextureId);
        var metallicRoughnessSampler = getSampler<SamplerState>(instance.material->metallicRoughnessSamplerId);
        float2 metallicRoughness = metallicRoughnessTexture.SampleLevel(metallicRoughnessSampler, uv, 0).bg;
        material.metallic = metallicRoughness.x;
        material.roughness = metallicRoughness.y;
    } else {
        material.metallic = 1.0;
        material.roughness = 1.0;
    }

    if (instance.material->emissiveTextureId >= 0) {
        var emissiveTexture = getSampledImage<Texture2D<float4>>(instance.material->emissiveTextureId);
        var emissiveSampler = getSampler<SamplerState>(instance.material->emissiveSamplerId);
        material.emissive = emissiveTexture.SampleLevel(emissiveSampler, uv, 0).xyz * instance.material->emissiveFactor;
    } else {
        material.emissive = instance.material->emissiveFactor;
    }

    payload.packedMaterial = material.pack();
    payload.dist = RayTCurrent();
}

[shader("anyhit")]
void ahit(inout Payload payload, BuiltInTriangleIntersectionAttributes attr) {
    uint triangleId = PrimitiveIndex();
    float3 barycentrics = float3(1.0 - dot(attr.barycentrics.xy, 1.0), attr.barycentrics.xy);

    let instance = pushConstants.instances[InstanceIndex()];
    let mesh = instance.mesh;

    uint index0 = mesh->indices[triangleId * 3 + 0];
    uint index1 = mesh->indices[triangleId * 3 + 1];
    uint index2 = mesh->indices[triangleId * 3 + 2];

    float2 uv0 = mesh->texcoords0[index0];
    float2 uv1 = mesh->texcoords0[index1];
    float2 uv2 = mesh->texcoords0[index2];
    float2 uv = barycentrics.x * uv0 + barycentrics.y * uv1 + barycentrics.z * uv2;

    var baseColorTexture = getSampledImage<Texture2D<float4>>(instance.material->baseColorTextureId);
    var baseColorSampler = getSampler<SamplerState>(instance.material->baseColorSamplerId);
    float4 baseColor = baseColorTexture.SampleLevel(baseColorSampler, uv, 0);
    if (baseColor.a < 0.5) {
        IgnoreHit();
    }

    payload.dist = RayTCurrent();
}
