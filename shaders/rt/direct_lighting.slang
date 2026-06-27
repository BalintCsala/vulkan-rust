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
    uint frame;
}

[vk::push_constant]
PushConstants pushConstants;

static float3 LIGHT_DIR = normalize(float3(1,3,2));

[shader("raygeneration")]
void rgen() {
    var gbuffer = getStorageImage<RWTexture2D<uint4>>(pushConstants.gbufferId);
    var directLightingOutput = getStorageImage<RWTexture2D<float4>>(pushConstants.directLightingOutputId);
    var tlas = RaytracingAccelerationStructure(pushConstants.tlas);

    uint2 pixel = DispatchRaysIndex().xy;
    Random random = Random(pixel, DispatchRaysDimensions().xy, pushConstants.frame);

    float2 screenPos = (float2(pixel) + 0.5) / DispatchRaysDimensions().xy;
    float4 temp = mul(pushConstants.viewProjectionInv, float4(screenPos * 2.0 - 1.0, 1.0, 1.0));
    float3 worldPos = temp.xyz / temp.w;

    float3 rayDir = normalize(worldPos - pushConstants.cameraPosition);

    RayDesc ray;
    ray.Origin = pushConstants.cameraPosition;
    ray.Direction = rayDir;
    ray.TMin = 0.01;
    ray.TMax = 256.0;
    
    Payload directPayload;
    TraceRay(tlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, directPayload);
    if (directPayload.dist < 0) {
        gbuffer[pixel] = 0u;
        return;
    }

    gbuffer[pixel] = directPayload.material.toRaw();
    Material material = directPayload.material.unpack();

    Payload sunPayload;
    ray.Origin += ray.Direction * directPayload.dist;
    ray.Direction = LIGHT_DIR;
    TraceRay(
        tlas, 
        RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 
        0xFF, 
        0, 0, 0, 
        ray, 
        sunPayload
    );

    float3 radiance = material.emissive;
    if (sunPayload.dist < 0) {
        radiance += material.brdf(-rayDir, LIGHT_DIR) * 1.0;
    }

    directLightingOutput[pixel] = float4(radiance, directPayload.dist);
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.dist = -1.0;
}
