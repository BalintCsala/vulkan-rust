module material;

import math;
import random;
import packing;

public struct Material {
    public float3 baseColor;
    public float metallic;
    public float roughness;
    public float3 normal;
    public float3 emissive;

    float3 fresnel(float cosTheta) {
        float3 f0 = lerp(0.04, baseColor, metallic);
        return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
    }

    public float3 brdf(float3 view, float3 light) {
        float3 radiance = 0.0;

        float3 halfway = normalize(view + light);

        float NdotH = dot(normal, halfway);
        float NdotL = dot(normal, light);
        float NdotV = dot(normal, view);

        // // Diffuse (Hammon)
        // float facing = dot(view, light) * 0.5 + 0.5;
        // float rough = facing * (0.9 - 0.4 * facing) * (0.5 + NdotH) / NdotH;
        // float smooth = 1.05 * (1.0 - pow(1.0 - NdotL, 5.0)) * (1.0 - pow(1.0 - NdotV, 5.0));
        // float single = 1.0 / PI * lerp(smooth, rough, roughness);
        // float multi = 0.1159 * roughness;
        //
        // float3 diffuse = baseColor * (single + baseColor * multi) * (1.0 - metallic);
        float3 diffuse = baseColor;

        radiance += diffuse * saturate(NdotL);

        return radiance;
    }

    public float3 sampleCosineWeightedHemisphere(inout Random random)
    {
        float2 v = random.uniform2();
        float angle = 2.0 * PI * v.x;
        float u = 2.0 * v.y - 1.0;

        float3 directionOffset = float3(sqrt(1.0 - u * u) * float2(cos(angle), sin(angle)), u);
        return normalize(normal + directionOffset);
    }

    public uint4 pack() {
        uint4 packed;
        packed.x = packUnorm4x8(float4(pow(baseColor, 1.0 / 2.2), 1.0));
        packed.y = packUnorm4x8(float4(metallic, roughness, 0.0, 0.0));
        packed.z = packUnorm2x16(octahedron::encode(normal));
        packed.w = packHDR(emissive);
        return packed;
    }

    public static Material unpack(uint4 packed) {
    Material material;
        material.baseColor = pow(unpackUnorm4x8ToFloat(packed.x).rgb, 2.2);
        float2 metallicRoughness = unpackUnorm4x8ToFloat(packed.y).xy;
        material.metallic = metallicRoughness.x;
        material.roughness = metallicRoughness.y;
        material.normal = octahedron::decode(unpackUnorm2x16ToFloat(packed.z));
        material.emissive = unpackHDR(packed.w);
        return material;
    }
}
