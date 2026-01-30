use std::sync::Arc;

use ash::vk;

use crate::rendering::wrappers::device::Device;

const INITIAL_COMMAND_BUFFER_COUNT: u32 = 8;

pub struct CommandCache {
    device: Arc<Device>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    next_id: usize,
}

impl CommandCache {
    pub fn new(device: Arc<Device>) -> Self {
        let command_pool = unsafe {
            device
                .create_command_pool(&vk::CommandPoolCreateInfo::default(), None)
                .unwrap()
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(INITIAL_COMMAND_BUFFER_COUNT)
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .unwrap()
        };

        Self {
            device,
            command_pool,
            command_buffers,
            next_id: 0,
        }
    }

    pub fn get_command_buffer(&mut self) -> vk::CommandBuffer {
        if self.next_id == self.command_buffers.len() {
            let mut new_command_buffers = unsafe {
                self.device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(INITIAL_COMMAND_BUFFER_COUNT)
                            .command_pool(self.command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )
                    .unwrap()
            };
            self.command_buffers.append(&mut new_command_buffers);
        }
        self.next_id += 1;
        self.command_buffers[self.next_id - 1]
    }

    pub fn reset(&mut self) {
        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        self.next_id = 0;
    }
}

impl Drop for CommandCache {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(
                self.command_pool,
                &self.command_buffers.drain(..).collect::<Vec<_>>(),
            );
        };
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        };
    }
}
