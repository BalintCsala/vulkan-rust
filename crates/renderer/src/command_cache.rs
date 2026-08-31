use std::sync::Arc;

use ash::vk;

use vulkan_utils::wrappers::device::Device;

const COMMAND_BUFFER_COUNT: usize = 64;

pub struct CommandCache {
    device: Arc<Device>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    semaphore: vk::Semaphore,
    next_id: usize,
}

impl CommandCache {
    pub fn new(device: Arc<Device>, queue: vk::Queue) -> Self {
        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap()
        };

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_buffer_count(COMMAND_BUFFER_COUNT as u32)
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY),
                )
                .unwrap()
        };

        let semaphore = unsafe {
            device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(0),
                    ),
                    None,
                )
                .unwrap()
        };

        Self {
            device,
            queue,
            command_pool,
            command_buffers,
            semaphore,
            next_id: 1,
        }
    }

    pub fn run_command<T: FnOnce(&vk::CommandBuffer)>(&mut self, fence: vk::Fence, callback: T) {
        let current_tail = unsafe {
            self.device
                .get_semaphore_counter_value(self.semaphore)
                .unwrap()
        };

        if current_tail as usize + COMMAND_BUFFER_COUNT < self.next_id {
            unsafe {
                self.device
                    .wait_semaphores(
                        &vk::SemaphoreWaitInfo::default()
                            .semaphores(&[self.semaphore])
                            .values(&[current_tail + 1]),
                        u64::MAX,
                    )
                    .unwrap();
            };
        }

        let command_buffer = self.command_buffers[self.next_id % COMMAND_BUFFER_COUNT];

        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap();
        };

        unsafe {
            self.device
                .begin_command_buffer(
                    command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }

        callback(&command_buffer);

        unsafe {
            self.device.end_command_buffer(command_buffer).unwrap();
        }

        unsafe {
            self.device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[
                            vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)
                        ])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.semaphore)
                            .value(self.next_id as u64)])],
                    fence,
                )
                .unwrap();
        };

        self.next_id += 1;
    }
}

impl Drop for CommandCache {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.semaphore, None);
        };
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        };
        self.command_buffers.clear();
    }
}
