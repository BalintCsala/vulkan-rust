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
    public float3 *tangents;
    public float2 *texcoords0;
    public float2 *texcoords1;
    public float4 *colors;
    public float4 *joints;
    public float4 *weights;

    public uint baseColorTextureId : 16;
    public uint baseColorTexcoordId : 8;
    public uint baseColorSamplerId : 8;

    public uint normalTextureId : 16;
    public uint normalTexcoordId : 8;
    public uint normalSamplerId : 8;

    public uint metallicRoughnessTextureId : 16;
    public uint metallicRoughnessTexcoordId : 8;
    public uint metallicRoughnessSamplerId : 8;

    public uint emissiveTextureId : 16;
    public uint emissiveTexcoordId : 8;
    public uint emissiveSamplerId : 8;

    public float3 emissiveFactor;

    public FragmentData getFragmentData(uint index0,
                                        uint index1,
                                        uint index2,
                                        float3 position0,
                                        float3 position1,
                                        float3 position2,
                                        Barycentrics barycentrics) {
        FragmentData fragmentData;
        float2 duv0;
        float2 duv1;
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

            fragmentData.uvs[1] = barycentrics.interpolate(uv0, uv1, uv2);
        }

        float3 normal;
        if (normals != nullptr) {
            var normal0 = normals[index0];
            var normal1 = normals[index1];
            var normal2 = normals[index2];

            normal = barycentrics.interpolateWithoutDerivatives(normal0, normal1, normal2);
        } else {
            normal = normalize(cross(position1 - position0, position2 - position0));
        }

        float3 tangent;
        if (tangents != nullptr) {
            var tangent0 = tangents[index0];
            var tangent1 = tangents[index1];
            var tangent2 = tangents[index2];
            tangent = barycentrics.interpolateWithoutDerivatives(tangent0, tangent1, tangent2);
        } else {
            float3 normal = normalize(cross(position1 - position0, position2 - position0));
            float3 edge0 = position1 - position0;
            float3 edge1 = position2 - position0;
            tangent = (duv1.y * edge0 - duv0.y * edge1) / (duv0.x * duv1.y - duv1.x * duv0.y);
        }
        var bitangent = normalize(cross(tangent, normal));
        fragmentData.tbn = float3x3(tangent, bitangent, normal);

        return fragmentData;
    }
};

public struct InstanceData {
    public float4x4 modelMatrix;
    public float4x4 normalMatrix;
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



