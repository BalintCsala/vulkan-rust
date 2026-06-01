implementing math;

public struct UnitQuat
{
    float4 values;

    public __init(float x, float y, float z, float w)
    {
        values = float4(x, y, z, w);
    }

    public __init(float4 val)
    {
        values = val;
    }

    public static UnitQuat fromRotationToZ(float3 dir)
    {
        if (dir.z < -0.999)
            return UnitQuat(1.0, 0.0, 0.0, 0.0);

        return UnitQuat(normalize(float4(dir.y, -dir.x, 0.0, 1.0 + dir.z)));
    }

    public UnitQuat invert()
    {
        return UnitQuat(float4(-values.xyz, values.w));
    }

    public float3 rotate(float3 p)
    {
        UnitQuat inv = invert();
        UnitQuat temp = this * UnitQuat(float4(p, 0.0));
        return cross(temp.values.xyz, inv.values.xyz) + temp.values.w * inv.values.xyz + temp.values.xyz * inv.values.w;
    }
}

public UnitQuat operator *(UnitQuat a, UnitQuat b)
{
    return UnitQuat(float4(
        cross(a.values.xyz, b.values.xyz) + a.values.xyz * b.values.w + b.values.xyz * a.values.w,
        a.values.w * b.values.w - dot(a.values.xyz, b.values.xyz)));
}
