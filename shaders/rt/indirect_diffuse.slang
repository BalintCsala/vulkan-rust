import lib.bindless;
import lib.forward;
import lib.random;
import lib.material;
import common;

static uint MAX_PT_DEPTH = 5;

struct PushConstants {
    uint64_t tlas;
    ModelData *models;
    InstanceData *instances;
    float4x4 viewProjectionInv;
    float3 cameraPosition;
    int gbufferId;
    int directLightingOutputId;
    int indirectDiffuseOutputId;
    uint frame;
}

[vk::push_constant]
PushConstants pushConstants;

[shader("raygeneration")]
void rgen() {
    var gbuffer = getStorageImage<RWTexture2D<uint4>>(pushConstants.gbufferId);
    var directLightingOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.directLightingOutputId);
    var indirectDiffuseOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.indirectDiffuseOutputId);
    var tlas = RaytracingAccelerationStructure(pushConstants.tlas);

    uint2 pixel = DispatchRaysIndex().xy;
    Random random = Random(pixel, DispatchRaysDimensions().xy, pushConstants.frame);

    float2 screenPos = (float2(pixel) + random.uniform2()) / DispatchRaysDimensions().xy;
    float4 temp = mul(pushConstants.viewProjectionInv, float4(screenPos * 2.0 - 1.0, 1.0, 1.0));
    float3 worldPos = temp.xyz / temp.w;

    float3 throughput = 1.0;
    float3 radiance = 0.0;

    RayDesc ray;
    ray.Origin = pushConstants.cameraPosition;
    ray.Direction = normalize(worldPos - ray.Origin);
    ray.TMin = 0.01;
    ray.TMax = 256.0;
    
    const float3 LIGHT_DIR = normalize(float3(1,3,2));

    [unroll]
    for (int i = 0; i < MAX_PT_DEPTH; i++) {
        Payload payload;
        TraceRay(tlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);

        if (payload.dist < 0.0) {
            radiance += throughput * pow(float3(0.45, 0.61, 0.84), 2.2) * 0.05;
            break;
        }
        Material material = payload.material.unpack();
        radiance += throughput * material.emissive;

        ray.Origin += ray.Direction * payload.dist;

        {
            // Sun
            Payload sunPayload;
            ray.Direction = LIGHT_DIR;
            TraceRay(
                tlas, 
                RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 
                0xFF, 
                0, 0, 0, 
                ray, 
                sunPayload
            );

            if (sunPayload.dist < 0.0) {
                radiance += throughput * max(dot(material.normal, LIGHT_DIR), 0.0) * material.baseColor * 0.1;
            }
        }

        throughput *= material.baseColor;

        if (i != MAX_PT_DEPTH - 1) {
            ray.Direction = material.sampleCosineWeightedHemisphere(random);
        }

        // if (i > 2) {
        //     float russianRouletteProb = dot(throughput, 1.0 / 3.0);
        //     if (random.uniform() > russianRouletteProb) {
        //         break;
        //     }
        //     throughput /= russianRouletteProb;
        // }
    }

    float reprojectionWeight = max(1.0 / float(pushConstants.frame + 1), 1.0 / 800.0);    
    indirectDiffuseOutput[pixel].rgb = lerp(indirectDiffuseOutput[pixel].rgb, radiance, reprojectionWeight);
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.dist = -1.0;
}
