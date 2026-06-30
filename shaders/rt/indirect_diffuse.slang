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
    float3 sunDir;
    int depthId;
    int gbufferId;
    int outputId;
    uint frame;
}

[vk::push_constant]
PushConstants pushConstants;

float3 pathtrace(RaytracingAccelerationStructure tlas, float3 worldPos, float3 rayDir) {
    float3 radiance = 0.0;

    RayDesc ray;
    ray.Origin = worldPos;
    ray.Direction = rayDir;
    ray.TMin = 0.01;
    ray.TMax = 256.0;
    
    Payload payload;
    TraceRay(tlas, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);

    if (payload.dist < 0.0) {
        return pow(float3(0.45, 0.61, 0.84), 2.2) * 0.5;
    }

    Material material = Material.unpack(payload.packedMaterial);
    radiance += material.emissive * 10.0;

    if (material.metallic == 1.0) {
        return radiance;
    }

    float NdotL = dot(material.normal, pushConstants.sunDir);
    if (NdotL > 0.0) {
        // Sun
        Payload sunPayload;

        RayDesc sunRay;
        sunRay.Origin = ray.Origin + ray.Direction * payload.dist;
        sunRay.Direction = pushConstants.sunDir;
        sunRay.TMin = 0.01;
        sunRay.TMax = 256.0;
        TraceRay(
            tlas, 
            RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 
            0xFF, 
            0, 0, 0, 
            sunRay, 
            sunPayload
        );

        if (sunPayload.dist < 0.0) {
            radiance += material.brdf(-ray.Direction, sunRay.Direction) * max(NdotL, 0.0);
        }
    }
    return radiance;
}

[shader("raygeneration")]
void rgen() {
    let depthTex = getSampledImage<Texture2D<float>>(pushConstants.depthId);
    let gbuffer = getStorageImage<RWTexture2D<uint4>>(pushConstants.gbufferId);
    let output = getStorageImage<RWTexture2D<float4>>(pushConstants.outputId);
    let tlas = RaytracingAccelerationStructure(pushConstants.tlas);

    uint2 pixel = DispatchRaysIndex().xy;
    Random random = Random(pixel, DispatchRaysDimensions().xy, pushConstants.frame);

    float depth = depthTex[pixel * 2];
    if (depth == 1.0) {
        output[pixel] = 0.0;
        return;
    }
    Material primaryMaterial = Material.unpack(gbuffer[pixel * 2]);
    if (primaryMaterial.metallic == 1.0) {
        output[pixel] = 0.0; 
        return;
    }

    float2 screenPos = (float2(pixel) + random.uniform2()) / DispatchRaysDimensions().xy;
    float4 temp = mul(pushConstants.viewProjectionInv, float4(screenPos * 2.0 - 1.0, depth, 1.0));
    float3 worldPos = temp.xyz / temp.w;
    float3 rayDir = primaryMaterial.sampleCosineWeightedHemisphere(random);

    float3 radiance = pathtrace(tlas, worldPos, rayDir);

    float reprojectionWeight = max(1.0 / float(pushConstants.frame + 1), 1.0 / 800.0);
    output[pixel].rgb = lerp(output[pixel].rgb, radiance, reprojectionWeight);
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.dist = -1.0;
}
