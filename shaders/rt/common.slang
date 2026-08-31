module common;

import lib.forward;
import lib.random;
import lib.material;

public struct RTPushConstantsBase<M> {
    public uint64_t tlas;
    public InstanceData<M> *instances;
}

[raypayload] 
public struct Payload {
    public uint4 packedMaterial : read(caller) : write(closesthit);
    public float dist : read(caller) : write(caller, closesthit, anyhit, miss);
}
