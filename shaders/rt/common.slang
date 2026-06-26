module common;

import lib.forward;
import lib.random;
import lib.material;

public struct RTPushConstantsBase {
    public uint64_t tlas;
    public ModelData *models;
    public InstanceData *instances;
}

[raypayload] 
public struct Payload {
    public PackedMaterial material : read(caller) : write(closesthit);
    public float dist : read(caller) : write(caller, closesthit, anyhit, miss);
}
