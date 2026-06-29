import lib.bindless;
import lib.tonemap;

struct PushConstants {
    int gbufferId;
    int directLightingOutputId;
    int indirectDiffuseOutputId;
}

struct VertexOut {
    float4 position : SV_Position;
};

[vk::push_constant]
PushConstants pushConstants;

[shader("vertex")]
VertexOut vs(uint index : SV_VertexID) {
    VertexOut o;
    switch (index) {
        case 0:
            o.position = float4(-1.0, -1.0, 0.0, 1.0);
            break;
        case 1:
            o.position = float4(-1.0, 3.0, 0.0, 1.0);
            break;
        case 2:
            o.position = float4(3.0, -1.0, 0.0, 1.0);
            break;
    }
    return o;
}

[shader("fragment")]
float4 fs(VertexOut i) {
    uint2 pixel = uint2(i.position.xy);
    var gbuffer = getStorageImage<RWTexture2D<uint4>>(pushConstants.gbufferId);
    var directLightingOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.directLightingOutputId);
    var indirectDiffuseOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.indirectDiffuseOutputId);

    uint4 gbufferData = gbuffer[pixel];
    float3 radiance;
    if (all(gbufferData == 0u)) {
        // Sky
        radiance = pow(float3(0.45, 0.61, 0.84), 2.2) * 0.05;
    } else {
        radiance = directLightingOutput[pixel].rgb + indirectDiffuseOutput[pixel / 2].rgb;
    }

    return float4(tonemap(radiance * 5.0), 1.0);
}
