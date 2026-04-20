module material;

import math;

public struct Material {
    public float3 baseColor;
    public float metallic;
    public float roughness;
    public float3 normal;
    public float3 f0;
    public float3 emissive;

    float3 fresnel(float cosTheta) {
        return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
    }

    public float3 brdf(float3 view, float3 light) {
        float3 radiance = 0.0;

        float3 halfway = normalize(view + light);

        float NdotH = dot(normal, halfway);
        float NdotL = dot(normal, light);
        float NdotV = dot(normal, view);

        // Diffuse (Hammon)
        float facing = dot(view, light) * 0.5 + 0.5;
        float rough = facing * (0.9 - 0.4 * facing) * (0.5 + NdotH) / NdotH;
        float smooth = 1.05 * (1.0 - pow(1.0 - NdotL, 5.0)) * (1.0 - pow(1.0 - NdotV, 5.0));
        float single = 1.0 / PI * lerp(smooth, rough, roughness);
        float multi = 0.1159 * roughness;

        float3 diffuse = baseColor * (single + baseColor * multi) * (1.0 - metallic);

        radiance += diffuse * saturate(NdotL);

        return radiance;
    }
}



