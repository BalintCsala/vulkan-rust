module random;

public struct Random
{
    uint state;

    public __init(uint seed)
    {
        state = seed;
    }

    public __init(uint2 pixel, uint2 resolution, uint frame)
    {
        state = frame;
        state = (pixel.x + pixel.y * resolution.x + frame * resolution.x * resolution.y);
        uniformUint();
    }

    [mutating]
    public uint uniformUint()
    {
        uint newState = state * uint(747796405) + uint(2891336453);
        uint word = ((newState >> ((newState >> uint(28)) + uint(4))) ^ newState) * uint(277803737);
        state = (word >> uint(22)) ^ word;
        return state;
    }

    [mutating]
    public float uniform()
    {
        return float(uniformUint() & 0x7fffffffU) / float(0x7fffffff);
    }

    [mutating]
    public float2 uniform2()
    {
        return float2(uniform(), uniform());
    }

    [mutating]
    public float3 uniform3()
    {
        return float3(uniform(), uniform(), uniform());
    }

    [mutating]
    public float4 uniform4()
    {
        return float4(uniform(), uniform(), uniform(), uniform());
    }
}
