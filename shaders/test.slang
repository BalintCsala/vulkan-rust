struct VSOut {
    float4 position : SV_Position;
    float3 color;
};

struct ModelData {
    float3* positions;
    float3* normals;
    float3* tangents;
    float2* texcoords0;
    float2* texcoords1;
    float4* colors;
    float4* joints;
    float4* weights;
};

struct PushConstants {
    float4x4 viewProjection;
    ModelData *modelData;
};

[shader("vertex")]
VSOut vs(uint id: SV_VertexID, uniform PushConstants pushConstants) {
    VSOut output;
    output.position = mul(pushConstants.viewProjection, float4(pushConstants.modelData->positions[id], 1.0));
    output.color = pushConstants.modelData->normals[id];
    return output;
}

[shader("fragment")]
float4 fs(VSOut input) {
    return 0.0;
    return float4(input.color, 1.0);
}
