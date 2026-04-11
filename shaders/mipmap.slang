import lib.bindless;

struct PushConstants {
    uint baseImageId;
    uint numOfMips;
};

[vk::push_constant]
PushConstants pushConstants;

groupshared float4 sharedMem[32 + 1][32];

[shader("compute")]
[numthreads(32, 32)]
void cs(uint2 gid: SV_DispatchThreadID, uint2 lid: SV_GroupThreadID) {
    var base = getStorageImage<RWTexture2D<float4>>(pushConstants.baseImageId);
    uint2 resolution;
    base.GetDimensions(resolution.x, resolution.y);

    bool inside = gid.x < resolution.x && gid.y < resolution.y;

    if (inside) {
        float4 value = base[gid * 2 + uint2(0, 0)] + base[gid * 2 + uint2(1, 0)] + base[gid * 2 + uint2(0, 1)] +
                       base[gid * 2 + uint2(1, 1)];
        value *= 0.25;
        sharedMem[lid.x][lid.y] = value;
        getStorageImage<RWTexture2D<float4>>(pushConstants.baseImageId + 1)[gid] = value;
    }

    var mask = 1;
    var step = 1;
    for (uint i = 1; i < pushConstants.numOfMips; i++) {
        GroupMemoryBarrierWithGroupSync();
        if (inside && all((lid & mask) == 0)) {
            float4 value = sharedMem[lid.x][lid.y] + sharedMem[lid.x + step][lid.y] + sharedMem[lid.x][lid.y + step] +
                           sharedMem[lid.x + step][lid.y];
            value *= 0.25;

            sharedMem[lid.x][lid.y] = value;
            getStorageImage<RWTexture2D<float4>>(pushConstants.baseImageId + i + 1)[gid / step / 2] = value;
        }
        mask = (mask << 1) | 1;
        step *= 2;
    }
}



