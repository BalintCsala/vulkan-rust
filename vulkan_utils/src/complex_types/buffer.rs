use std::sync::Arc;

use ash::vk;
use vk_mem::{Alloc, Allocation, AllocationCreateFlags, AllocationCreateInfo, MemoryUsage};

use crate::{
    utility_functions::assign_debug_name,
    wrappers::{allocator::Allocator, device::Device},
};

#[derive(Clone)]
pub struct Buffer {
    allocator: Arc<Allocator>,
    pub handle: vk::Buffer,
    allocation: Option<Allocation>,
    pub address: vk::DeviceAddress,
}

impl Buffer {
    pub fn new(
        device: &Arc<Device>,
        allocator: Arc<Allocator>,
        usage: vk::BufferUsageFlags,
        size: u64,
        name: &str,
    ) -> Self {
        let (buffer, allocation) = unsafe {
            allocator
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .size(size),
                    &AllocationCreateInfo {
                        flags: AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                            | AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD
                            | AllocationCreateFlags::MAPPED,
                        usage: MemoryUsage::Auto,
                        ..Default::default()
                    },
                )
                .unwrap()
        };

        assign_debug_name(device, buffer, name);

        let address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        Self {
            allocator,
            handle: buffer,
            allocation: Some(allocation),
            address,
        }
    }

    pub fn write<T>(&mut self, data: &[T], offset: usize) {
        let dst = self
            .allocator
            .get_allocation_info(
                self.allocation
                    .as_ref()
                    .expect("Buffer was already destroyed"),
            )
            .mapped_data;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                dst.byte_add(offset) as *mut T,
                data.len(),
            );
        };
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(mut allocation) = self.allocation.take() {
            unsafe {
                self.allocator.destroy_buffer(self.handle, &mut allocation);
            };
        }
    }
}
