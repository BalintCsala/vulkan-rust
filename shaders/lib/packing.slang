
module packing;

namespace octahedron {

// Source: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
float2 octWrap(float2 v) {
    return (1.0 - abs(v.yx)) * select(v.xy >= 0.0, 1.0, -1.0);
}

public float2 encode(float3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    n.xy = n.z >= 0.0 ? n.xy : octWrap(n.xy);
    n.xy = n.xy * 0.5 + 0.5;
    return n.xy;
}

public float3 decode(float2 f) {
    f = f * 2.0 - 1.0;

    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, -t, t);
    return normalize(n);
}

}

public uint packHDR(float3 color) {
    static float MAX_VALUE = 65408.0;
    static float MIN_VALUE = 1.52587e-5;

    color = max(color, 0.0);
    float maxCol = max3(color.r, color.g, color.b);
    if (maxCol < MIN_VALUE) {
        return 0;
    } 

    int exponent = max(-15, (int)floor(log2(maxCol))) + 1 + 15;
    float denom = exp2(exponent - 15 - 9);    

    uint r = (uint)round(color.r / denom);
    uint g = (uint)round(color.g / denom);
    uint b = (uint)round(color.b / denom);

    if (r > 511 || g > 511 || b > 511) {
        exponent++;
        denom *= 2.0;
        uint r = (uint)round(color.r / denom);
        uint g = (uint)round(color.g / denom);
        uint b = (uint)round(color.b / denom);
    }

    r = min(r, 511u);
    g = min(g, 511u);
    b = min(b, 511u);

    return r | (g << 9) | (b << 18) | ((uint)exponent << 27);
}

public float3 unpackHDR(uint packed) {
    uint exponent = (packed >> 27) & 0x1F;
    if (exponent == 0) {
        return 0.0;
    }

    uint r = packed & 0x1FF;
    uint g = (packed >> 9) & 0x1FF;
    uint b = (packed >> 18) & 0x1FF;

    float scale = exp2((int)exponent - 15 - 9);
    return float3(r, g, b) * scale;
}



