use std::sync::Arc;

use ash::{
    ext::debug_utils,
    khr::{
        acceleration_structure, deferred_host_operations, ray_tracing_pipeline, surface, swapchain,
    },
    vk,
};
use bevy::ecs::resource::Resource;
use vk_mem::{AllocatorCreateFlags, AllocatorCreateInfo};
use winit::raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use vulkan_utils::{
    complex_types::image::Image,
    utility_functions::assign_debug_name,
    wrappers::{allocator::Allocator, device::Device, instance::Instance},
};

const FRAMES_IN_FLIGHT: usize = 3;

struct FrameData {
    device: Arc<Device>,
    command_buffer: vk::CommandBuffer,
    image_acquired: vk::Semaphore,
    deletion_queue: Vec<Box<dyn Send + Sync>>,
}

impl FrameData {
    pub fn new(device: Arc<Device>, frame_id: usize, command_pool: vk::CommandPool) -> Self {
        let image_acquired = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap()
        };
        assign_debug_name(
            &device,
            image_acquired,
            &format!("Image acquired semaphore #{}", frame_id),
        );

        let command_buffer = unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .command_buffer_count(1),
                )
                .unwrap()[0]
        };

        Self {
            device,
            image_acquired,
            command_buffer,
            deletion_queue: Vec::new(),
        }
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.image_acquired, None);
        };
    }
}

struct SwapchainImageData {
    rendering_finished: vk::Semaphore,
    image: Image,
}

#[derive(Resource)]
pub struct VulkanState {
    _entry: ash::Entry,
    pub instance: Arc<Instance>,
    pub physical_device: vk::PhysicalDevice,
    pub device: Arc<Device>,
    pub queue: vk::Queue,

    command_pool: vk::CommandPool,

    surface_instance: surface::Instance,
    surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    min_image_count: u32,
    pub extent: vk::Extent2D,

    swapchain_device: swapchain::Device,
    swapchain: Option<vk::SwapchainKHR>,
    images: Vec<SwapchainImageData>,
    image_id: Option<u32>,

    frames: Vec<FrameData>,
    frame_id: usize,

    frame_semaphore: vk::Semaphore,

    pub allocator: Arc<Allocator>,
}

impl VulkanState {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Self {
        let entry = ash::Entry::linked();

        let instance = {
            let mut extensions = ash_window::enumerate_required_extensions(display_handle)
                .unwrap()
                .to_vec();
            extensions.push(debug_utils::NAME.as_ptr());

            Arc::new(Instance::new(
                &entry,
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_extension_names(&extensions),
            ))
        };

        let surface_instance = surface::Instance::new(&entry, &instance);

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .unwrap()
        };

        let physical_device = unsafe {
            let physical_devices = instance.enumerate_physical_devices().unwrap();
            *physical_devices
                .iter()
                .find(|&physical_device| {
                    let properties = instance.get_physical_device_properties(*physical_device);
                    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap_or(physical_devices.first().unwrap())
        };

        let surface_format = unsafe {
            let surface_formats = surface_instance
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap();
            *surface_formats
                .iter()
                .find(|&surface_format| {
                    surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                        && (surface_format.format == vk::Format::R8G8B8A8_SRGB
                            || surface_format.format == vk::Format::B8G8R8A8_SRGB)
                })
                .unwrap_or(&surface_formats[0])
        };

        let present_mode = vk::PresentModeKHR::MAILBOX;

        let surface_capabilities = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(physical_device, surface)
                .unwrap()
        };
        let min_image_count = surface_capabilities.min_image_count + 5;

        let device = Arc::new(Device::new(
            instance.clone(),
            &physical_device,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(0)
                    .queue_priorities(&[1.0])])
                .enabled_extension_names(&[
                    swapchain::NAME.as_ptr(),
                    acceleration_structure::NAME.as_ptr(),
                    ray_tracing_pipeline::NAME.as_ptr(),
                    deferred_host_operations::NAME.as_ptr(),
                ])
                .push_next(
                    &mut vk::PhysicalDeviceFeatures2::default().features(
                        vk::PhysicalDeviceFeatures::default()
                            .shader_int16(true)
                            .shader_int64(true)
                            .geometry_shader(true)
                            .fragment_stores_and_atomics(true),
                    ),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan12Features::default()
                        .buffer_device_address(true)
                        .descriptor_indexing(true)
                        .runtime_descriptor_array(true)
                        .descriptor_binding_partially_bound(true)
                        .shader_sampled_image_array_non_uniform_indexing(true)
                        .shader_storage_image_array_non_uniform_indexing(true)
                        .descriptor_binding_sampled_image_update_after_bind(true)
                        .descriptor_binding_storage_image_update_after_bind(true)
                        .shader_int8(true)
                        .scalar_block_layout(true)
                        .timeline_semaphore(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceVulkan13Features::default()
                        .synchronization2(true)
                        .dynamic_rendering(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                        .acceleration_structure(true),
                )
                .push_next(
                    &mut vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                        .ray_tracing_pipeline(true),
                ),
        ));

        let queue = unsafe {
            device.get_device_queue2(
                &vk::DeviceQueueInfo2::default()
                    .queue_family_index(0)
                    .queue_index(0),
            )
        };

        let command_pool = unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap()
        };

        let allocator = {
            let mut create_info = AllocatorCreateInfo::new(&instance, &device, physical_device);
            create_info.flags = AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

            Arc::new(Allocator::new(device.clone(), create_info))
        };

        let swapchain_device = swapchain::Device::new(&instance, &device);

        let frames = (0..FRAMES_IN_FLIGHT)
            .map(|frame_id| FrameData::new(device.clone(), frame_id, command_pool))
            .collect();

        let extent = vk::Extent2D::default().width(width).height(height);

        let frame_semaphore = unsafe {
            device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(FRAMES_IN_FLIGHT as u64 - 1),
                    ),
                    None,
                )
                .unwrap()
        };

        Self {
            _entry: entry,
            instance,
            physical_device,
            device,
            command_pool,

            surface_instance,
            surface,
            extent,
            queue,
            allocator,
            swapchain: None,
            swapchain_device,
            images: Vec::new(),
            image_id: None,
            frames,
            frame_id: 0,
            present_mode,
            surface_format,
            min_image_count,
            frame_semaphore,
        }
    }

    fn create_swapchain(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        };
        let semaphores: Vec<_> = if !self.images.is_empty() {
            self.images
                .drain(..)
                .map(|image_data| image_data.rendering_finished)
                .collect()
        } else {
            (0..self.min_image_count)
                .map(|id| unsafe {
                    let semaphore = self
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .unwrap();
                    assign_debug_name(
                        &self.device,
                        semaphore,
                        &format!("Rendering finished semaphore #{}", id),
                    );
                    semaphore
                })
                .collect()
        };

        let old_swapchain = self.swapchain.take();

        let swapchain = unsafe {
            self.swapchain_device
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .present_mode(self.present_mode)
                        .image_format(self.surface_format.format)
                        .image_color_space(self.surface_format.color_space)
                        .min_image_count(self.min_image_count)
                        .image_usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        )
                        .image_extent(self.extent)
                        .surface(self.surface)
                        .image_array_layers(1)
                        .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null())),
                    None,
                )
                .unwrap()
        };

        self.images = unsafe {
            self.swapchain_device
                .get_swapchain_images(swapchain)
                .unwrap()
                .iter()
                .zip(semaphores)
                .map(|(&image, rendering_finished)| SwapchainImageData {
                    rendering_finished,
                    image: Image::from_raw(
                        self.device.clone(),
                        image,
                        self.surface_format.format,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageType::TYPE_2D,
                    ),
                })
                .collect()
        };
        self.images.iter().enumerate().for_each(|(i, image_data)| {
            assign_debug_name(
                &self.device,
                image_data.image.handle,
                &format!("Swapchain image #{}", i),
            );
        });

        self.swapchain = Some(swapchain);

        if let Some(swapchain) = old_swapchain {
            unsafe {
                self.swapchain_device.destroy_swapchain(swapchain, None);
            };
        }
    }

    pub fn queue_object_for_deletion(&mut self, obj: Box<dyn Send + Sync>) {
        self.frames[self.frame_id % FRAMES_IN_FLIGHT]
            .deletion_queue
            .push(obj);
    }

    pub fn current_image(&mut self) -> &mut Image {
        match self.image_id {
            Some(image_id) => &mut self.images[image_id as usize].image,
            None => panic!("Frame has not been started yet"),
        }
    }

    pub fn get_command_buffer(&self) -> vk::CommandBuffer {
        self.frames[self.frame_id % FRAMES_IN_FLIGHT].command_buffer
    }

    pub fn start_frame(&mut self, width: u32, height: u32) -> bool {
        if self.swapchain.is_none() || width != self.extent.width || height != self.extent.height {
            self.extent = vk::Extent2D::default().width(width).height(height);
            self.create_swapchain();
        }
        let swapchain = self.swapchain.unwrap();

        unsafe {
            self.device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.frame_semaphore])
                        .values(&[self.frame_id as u64]),
                    u64::MAX,
                )
                .unwrap()
        };

        let (image_id, _) = unsafe {
            match self.swapchain_device.acquire_next_image(
                swapchain,
                u64::MAX,
                self.frames[self.frame_id % FRAMES_IN_FLIGHT].image_acquired,
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(_) => {
                    self.create_swapchain();
                    return false;
                }
            }
        };

        unsafe {
            self.device
                .reset_command_buffer(
                    self.get_command_buffer(),
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
        };

        unsafe {
            self.device
                .begin_command_buffer(
                    self.get_command_buffer(),
                    &vk::CommandBufferBeginInfo::default(),
                )
                .unwrap();
        };

        self.image_id = Some(image_id);
        true
    }

    pub fn end_frame(&mut self) {
        let image_id = self.image_id.expect("Frame was not started yet");

        unsafe {
            self.device
                .end_command_buffer(self.get_command_buffer())
                .unwrap();
        };

        unsafe {
            self.device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                            .command_buffer(self.get_command_buffer())])
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.frames[self.frame_id % FRAMES_IN_FLIGHT].image_acquired)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)])
                        .signal_semaphore_infos(&[
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.images[image_id as usize].rendering_finished)
                                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
                            vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.frame_semaphore)
                                .value((self.frame_id + FRAMES_IN_FLIGHT) as u64),
                        ])],
                    vk::Fence::null(),
                )
                .unwrap();
        };

        unsafe {
            match self.swapchain_device.queue_present(
                self.queue,
                &vk::PresentInfoKHR::default()
                    .swapchains(&[self.swapchain.expect("Swapchain was not created yet")])
                    .image_indices(&[image_id])
                    .wait_semaphores(&[self.images[image_id as usize].rendering_finished]),
            ) {
                Ok(_) => (),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.create_swapchain(),
                Err(e) => panic!("Error: {}", e),
            }
        };

        self.frame_id += 1;
        self.image_id = None;

        self.frames[self.frame_id % FRAMES_IN_FLIGHT]
            .deletion_queue
            .clear();
    }
}

impl Drop for VulkanState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        self.frames.clear();

        self.images.drain(..).for_each(|image_data| unsafe {
            self.device
                .destroy_semaphore(image_data.rendering_finished, None);
        });
        unsafe {
            self.device.destroy_semaphore(self.frame_semaphore, None);
        };
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }

        if let Some(swapchain) = self.swapchain {
            unsafe {
                self.swapchain_device.destroy_swapchain(swapchain, None);
            }
        }
        unsafe {
            self.surface_instance.destroy_surface(self.surface, None);
        }
    }
}
