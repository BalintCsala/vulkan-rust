import lib.bindless;
import lib.tonemap;

struct PushConstants {
    int gbufferId;
    int directLightingOutputId;
    int indirectDiffuseOutputId;
    int outputId;
}

[vk::push_constant]
PushConstants pushConstants;

[shader("compute")]
[numthreads(8, 8)]
void cs(uint2 pixel: SV_DispatchThreadID) {
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

    var outputImage = getStorageImage<RWTexture2D<float4>>(pushConstants.outputId);
    outputImage[pixel] = float4(tonemap(radiance * 5.0), 1.0);
}



