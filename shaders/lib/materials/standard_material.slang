public struct StandardMaterial {
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
}

