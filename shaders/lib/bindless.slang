module bindless;

[vk::binding(0)]
__DynamicResource<__DynamicResourceKind.General> sampledImages[];

[vk::binding(1)]
__DynamicResource<__DynamicResourceKind.General> storageImages[];

[vk::binding(2)]
__DynamicResource<__DynamicResourceKind.Sampler> samplers[];

public T getSampledImage<T>(int id)
    where T : IOpaqueDescriptor {
    return sampledImages[NonUniformResourceIndex(id)].asOpaqueDescriptor<T>();
}

public T getStorageImage<T>(int id)
    where T : IOpaqueDescriptor {
    return storageImages[NonUniformResourceIndex(id)].asOpaqueDescriptor<T>();
}

public T getSampler<T>(int id)
    where T : IOpaqueDescriptor {
    return samplers[NonUniformResourceIndex(id)].asOpaqueDescriptor<T>();
}



