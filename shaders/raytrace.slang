import lib.bindless;
import lib.forward;
import lib.random;

struct PushConstants {
    uint64_t tlas;
    ModelData *models;
    InstanceData *instances;
    float4x4 viewProjectionInv;
    float3 cameraPosition;
    float2 resolution;
    int outputId;
    uint frame;
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

[shader("raygeneration")]
void rgen() {
    var outputImage = getStorageImage<RWTexture2D<float4>>(pushConstants.outputId);
    var tlas = RaytracingAccelerationStructure(pushConstants.tlas);
    uint2 pixel = DispatchRaysIndex().xy;


    float2 screenPos = (float2(pixel) + 0.5) / pushConstants.resolution;
    float4 temp = mul(pushConstants.viewProjectionInv, float4(screenPos * 2.0 - 1.0, 1.0, 1.0));
    float3 worldPos = temp.xyz / temp.w;

    float3 throughput = 1.0;
    float3 radiance = 0.0;

    RayDesc ray;
    ray.Origin = pushConstants.cameraPosition;
    ray.Direction = normalize(worldPos - ray.Origin);
    ray.TMin = 0.01;
    ray.TMax = 100.0;

    Random random = Random(pixel, uint2(pushConstants.resolution), pushConstants.frame);

    for (int i = 0; i < 5; i++) {
        Payload payload = Payload(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, random);
        TraceRay(tlas, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);


        if (!payload.hit) {
            break;
        }

        radiance += throughput * payload.radiance;
        throughput *= payload.pdf;

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDirection;

        if (i > 2) {
            float russianRouletteProb = dot(throughput, 1.0 / 3.0);
            if (random.uniform() > russianRouletteProb) {
                break;
            }
            throughput /= russianRouletteProb;
        }
    }

    outputImage[pixel].rgb = lerp(pow(outputImage[pixel].rgb, 2.2), radiance, 1.0 / float(pushConstants.frame + 1));
    outputImage[pixel].rgb = pow(outputImage[pixel].rgb, 1.0 / 2.2);
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.hit = false;
}
