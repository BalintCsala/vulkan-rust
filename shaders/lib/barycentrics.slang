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
        // Adapted from http://filmicworlds.com/blog/visibility-buffer-rendering-with-material-graphs/
        Barycentrics ret;

        float3 invW = 1.0 / float3(p0.w, p1.w, p2.w);

        float2 ndc0 = p0.xy * invW.x;
        float2 ndc1 = p1.xy * invW.y;
        float2 ndc2 = p2.xy * invW.z;

        float invDet = 1.0 / determinant(float2x2(ndc2 - ndc1, ndc0 - ndc1));
        ret.ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
        ret.ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
        float ddxSum = dot(ret.ddx, 1.0);
        float ddySum = dot(ret.ddy, 1.0);

        float2 deltaVec = ndc - ndc0;
        float interpInvW = invW.x + deltaVec.x * ddxSum + deltaVec.y * ddySum;
        float interpW = 1.0 / interpInvW;

        ret.barycentrics.x = interpW * (invW[0] + deltaVec.x * ret.ddx.x + deltaVec.y * ret.ddy.x);
        ret.barycentrics.y = interpW * (0.0 + deltaVec.x * ret.ddx.y + deltaVec.y * ret.ddy.y);
        ret.barycentrics.z = interpW * (0.0 + deltaVec.x * ret.ddx.z + deltaVec.y * ret.ddy.z);

        ret.ddx *= 2.0 / resolution.x;
        ret.ddy *= 2.0 / resolution.y;
        ddxSum *= 2.0 / resolution.x;
        ddySum *= 2.0 / resolution.y;

        ret.ddy *= -1.0;
        ddySum *= -1.0;

        float interpW_ddx = 1.0 / (interpInvW + ddxSum);
        float interpW_ddy = 1.0 / (interpInvW + ddySum);

        ret.ddx = interpW_ddx * (ret.barycentrics * interpInvW + ret.ddx) - ret.barycentrics;
        ret.ddy = interpW_ddy * (ret.barycentrics * interpInvW + ret.ddy) - ret.barycentrics;

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



