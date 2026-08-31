module forward;

import barycentrics;

public struct FragmentData {
    public InterpolatedValue<float2> uvs[2];
    public float3x3 tbn;
}

public struct ModelData {
    public float3 *positions;
    public uint *indices;
    public float3 *normals;
    public float4 *tangents;
    public float2 *texcoords0;
    public float2 *texcoords1;
    public float4 *colors;
    public float4 *joints;
    public float4 *weights;

    public int baseColorTextureId : 16;
    public int baseColorTexcoordId : 8;
    public int baseColorSamplerId : 8;
    public float4 baseColorFactor;

    public int normalTextureId : 16;
    public int normalTexcoordId : 8;
    public int normalSamplerId : 8;

    public int metallicRoughnessTextureId : 16;
    public int metallicRoughnessTexcoordId : 8;
    public int metallicRoughnessSamplerId : 8;

    public int emissiveTextureId : 16;
    public int emissiveTexcoordId : 8;
    public int emissiveSamplerId : 8;

    public float3 emissiveFactor;

    public FragmentData getFragmentData(uint index0,
                                        uint index1,
                                        uint index2,
                                        float3 position0,
                                        float3 position1,
                                        float3 position2,
                                        Barycentrics barycentrics) {
        FragmentData fragmentData;
        if (texcoords0 == nullptr && texcoords1 == nullptr) {
            fragmentData.tbn = float3x3(
                1, 0, 0,
                0, 1, 0,
                0, 0, 1
            );
            return fragmentData;
        }

        float2 duv0 = 0.0;
        float2 duv1 = 0.0;
        if (texcoords0 != nullptr) {
            var uv0 = texcoords0[index0];
            var uv1 = texcoords0[index1];
            var uv2 = texcoords0[index2];

            duv0 = uv1 - uv0;
            duv1 = uv2 - uv0;

            fragmentData.uvs[0] = barycentrics.interpolate(uv0, uv1, uv2);
        }

        if (texcoords1 != nullptr) {
            var uv0 = texcoords1[index0];
            var uv1 = texcoords1[index1];
            var uv2 = texcoords1[index2];

            if  (texcoords0 == nullptr) {
                duv0 = uv1 - uv0;
                duv1 = uv2 - uv0;
            }

            fragmentData.uvs[1] = barycentrics.interpolate(uv0, uv1, uv2);
        }

        float3 normal;
        if (normals != nullptr) {
            var normal0 = normals[index0];
            var normal1 = normals[index1];
            var normal2 = normals[index2];

            normal = normalize(barycentrics.interpolateWithoutDerivatives(normal0, normal1, normal2));
        } else {
            normal = normalize(cross(position1 - position0, position2 - position0));
        }

        float4 tangent;
        if (tangents != nullptr && false) {
            var tangent0 = tangents[index0];
            var tangent1 = tangents[index1];
            var tangent2 = tangents[index2];
            tangent = normalize(barycentrics.interpolateWithoutDerivatives(tangent0, tangent1, tangent2));
        } else {
            float3 edge0 = position1 - position0;
            float3 edge1 = position2 - position0;
            tangent = float4(normalize((duv1.y * edge0 - duv0.y * edge1) / (duv0.x * duv1.y - duv1.x * duv0.y)), 1.0);
        }
        var bitangent = normalize(cross(tangent.xyz, normal) * tangent.w);
        fragmentData.tbn = float3x3(tangent.xyz, bitangent, normal);

        return fragmentData;
    }
};

public struct InstanceData {
    public float4x4 modelMatrix;
    public float3x3 normalMatrix;
    public uint modelId;
};

static uint TRIANGLE_COUNT_NUM_BITS = 18;

public uint encodeUniqueId(uint instanceId, uint triangleId) {
    return (instanceId << (TRIANGLE_COUNT_NUM_BITS + 1)) | (triangleId << 1) | 1;
}

public uint getInstanceIdFromUniqueId(uint uniqueId) {
    return (uniqueId >> (TRIANGLE_COUNT_NUM_BITS + 1));
}

public uint getTriangleIdFromUniqueId(uint uniqueId) {
    return (uniqueId >> 1) & ((1 << TRIANGLE_COUNT_NUM_BITS) - 1);
}



