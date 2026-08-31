import lib.bindless;
import lib.forward;

struct PushConstants {
    float4x4 viewProjection;
    ModelData *models;
    InstanceData *instances;
};

[vk::push_constant]
PushConstants pushConstants;

struct VSOut {
    float4 position : SV_Position;
    float3 color;
};

[shader("vertex")]
VSOut vs(uint id: SV_VertexID, uint instanceId : SV_VulkanInstanceID) {
    VSOut output;

    var instance = pushConstants.instances[instanceId];
    var model    = pushConstants.models[instance.modelId];

    output.position = float4(model.positions[id], 1.0);
    output.position = mul(instance.modelMatrix, output.position);
    output.position = mul(pushConstants.viewProjection, output.position);

    output.color = model.colors[id].rgb;

    return output;
}

[shader("fragment")]
float4 fs(VSOut input) : SV_Target {
    return float4(input.color, 1.0);
}


