import lib.bindless;
import lib.encoding;
import lib.material;
import lib.tonemap;

struct PushConstants {
    float4x4 viewProjectionInverse;
    int depthTextureId;
    int baseColorId;
    int normalId;
    int metallicRoughnessId;
    int emissiveId;
    int outputId;
    float3 sunDirection;
    float3 sunLight;
    float3 cameraPosition;
    float2 resolution;
}

[vk::push_constant]
PushConstants pushConstants;

[shader("compute")]
[numthreads(8, 8)]
void cs(uint2 gid: SV_DispatchThreadID) {
    var outputTexture = getStorageImage<RWTexture2D<float4>>(pushConstants.outputId);
    var depthTexture = getStorageImage<RWTexture2D<float>>(pushConstants.depthTextureId);
    float depth = depthTexture[gid];
    if (depth == 1.0) {
        outputTexture[gid] = float4(0, 0, 0, 1);
        return;
    }

    var baseColorTexture = getStorageImage<RWTexture2D<float4>>(pushConstants.baseColorId);
    var normalTexture = getStorageImage<RWTexture2D<float4>>(pushConstants.normalId);
    var metallicRoughnessTexture = getStorageImage<RWTexture2D<float2>>(pushConstants.metallicRoughnessId);
    var emissiveTexture = getStorageImage<RWTexture2D<float4>>(pushConstants.emissiveId);

    Material material;
    material.f0 = 0.04;

    material.baseColor = pow(baseColorTexture[gid].rgb, 2.2);

    var packedNormal = normalTexture[gid];
    material.normal = octahedron::decode(packedNormal.xy);

    var metallicRoughness = metallicRoughnessTexture[gid];
    material.metallic = metallicRoughness.r;
    material.roughness = metallicRoughness.g;

    material.emissive = emissiveTexture[gid].rgb;

    float3 screen = float3((gid + 0.5) / pushConstants.resolution * 2.0 - 1.0, depth);
    float4 temp = mul(pushConstants.viewProjectionInverse, float4(screen, 1.0));
    float3 world = temp.xyz / temp.w;

    float3 view = normalize(pushConstants.cameraPosition - world);

    float3 radiance = material.brdf(view, pushConstants.sunDirection) * pushConstants.sunLight;
    outputTexture[gid] = float4(tonemap(radiance), 1.0);
}



