module denoising;

public float normalWeight(float3 n1, float3 n2)
{
    const float NORMAL_WEIGHT_PARAM = 128.0;
    return pow(max(dot(n1, n2), 0.0), NORMAL_WEIGHT_PARAM);
}

public float depthWeight(float3 p1, float3 p2, float2 grad)
{
    const float DEPTH_WEIGHT_PARAM = 64.0;
    if (p1.z == p2.z)
    {
        return 1.0;
    }
    float expectedDepthDifference = dot(grad, p2.xy - p1.xy);
    float depthDifference = p2.z - p1.z;
    return exp(-abs((depthDifference - expectedDepthDifference) / (expectedDepthDifference + 0.01)) * DEPTH_WEIGHT_PARAM);
}

public float depthWeightOriginalLen(float3 p1, float3 p2, float2 grad)
{
    const float DEPTH_WEIGHT_PARAM = 1.0;
    return exp(-abs(p1.z - p2.z) / (DEPTH_WEIGHT_PARAM * length(grad * (p1.xy - p2.xy)) + 0.01) * DEPTH_WEIGHT_PARAM);
}

public float depthWeightOriginalDot(float3 p1, float3 p2, float2 grad)
{
    const float DEPTH_WEIGHT_PARAM = 1.0;
    return exp(-abs(p1.z - p2.z) / (DEPTH_WEIGHT_PARAM * abs(dot(grad, (p1.xy - p2.xy))) + 0.01));
}
