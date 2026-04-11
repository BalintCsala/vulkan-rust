module barycentrics;

public struct InterpolatedValue<T : IRWArray<float>> {
    public T value;
    public T ddx;
    public T ddy;
}

public struct Barycentrics {
    public float3 barycentrics;
    float3 ddx;
    float3 ddy;

    public static Barycentrics fromScreenPoints(float4 p0, float4 p1, float4 p2, float2 ndc, float2 resolution) {
        Barycentrics ret;

        float3 invW = 1.0 / float3(p0.w, p1.w, p2.w);

        float2 ndc0 = p0.xy * invW.x;
        float2 ndc1 = p1.xy * invW.y;
        float2 ndc2 = p2.xy * invW.z;

        float invArea = 1.0 / determinant(float2x2(ndc1 - ndc0, ndc2 - ndc0));

        ret.barycentrics.x = ((ndc1.x - ndc.x) * (ndc2.y - ndc.y) - (ndc1.y - ndc.y) * (ndc2.x - ndc.x)) * invArea;
        ret.barycentrics.y = ((ndc2.x - ndc.x) * (ndc0.y - ndc.y) - (ndc2.y - ndc.y) * (ndc0.x - ndc.x)) * invArea;
        ret.barycentrics.z = 1.0 - ret.barycentrics.x - ret.barycentrics.y;

        ret.ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invArea;
        ret.ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invArea;

        float3 num = ret.barycentrics * invW;
        float denom = dot(num, 1.0);
        ret.barycentrics = num / denom;

        float3 ddxNum = ret.ddx * invW;
        float3 ddyNum = ret.ddy * invW;

        float ddxDenom = dot(ret.ddx * invW, 1.0);
        float ddyDenom = dot(ret.ddy * invW, 1.0);

        float denom2 = denom * denom;

        ret.ddx = (ddxNum * denom - num * ddxDenom) / denom2 * (2.0 / resolution.x);
        ret.ddy = (ddyNum * denom - num * ddyDenom) / denom2 * (2.0 / resolution.y);

        return ret;
    }

    public InterpolatedValue<T> interpolate<T : IRWArray<float>>(T v0, T v1, T v2) {
        InterpolatedValue<T> interpolated;
        for (int i = 0; i < v0.getCount(); i++) {
            float3 v = float3(v0[i], v1[i], v2[i]);
            interpolated.value[i] = dot(v, barycentrics);
            interpolated.ddx[i] = dot(v, ddx);
            interpolated.ddy[i] = dot(v, ddy);
        }
        return interpolated;
    }

    public T interpolateWithoutDerivatives<T : IRWArray<float>>(T v0, T v1, T v2) {
        T value;
        for (int i = 0; i < v0.getCount(); i++) {
            float3 v = float3(v0[i], v1[i], v2[i]);
            value[i] = dot(v, barycentrics);
        }
        return value;
    }
};



