
module encoding;

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



