module forward;

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



