#include "glm/fwd.hpp"
#include <volk/volk.h>

#include <iostream>
#include <fstream>
#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <array>
#include <stdexcept>
#include <algorithm>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "baked_model.hpp"


namespace
{


	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define SHADERDIR_ "assets/cw4/shaders/"
#		define ASSETSDIR_ "assets/cw4/"
		constexpr char const* kRenderPassAVertShaderPath = SHADERDIR_ "renderPassA.vert.spv";
		constexpr char const* kRenderPassAFragShaderPath = SHADERDIR_ "renderPassA.frag.spv";
		constexpr char const* kRenderPassBVertShaderPath = SHADERDIR_ "renderPassB.vert.spv";
		constexpr char const* kRenderPassBFragShaderPath = SHADERDIR_ "renderPassB.frag.spv";
		constexpr char const* bakedModelPath = ASSETSDIR_ "suntemple.comp5822mesh";
#		undef SHADERDIR_
#		undef ASSETSDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
		constexpr VkFormat kBaseColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
		constexpr VkFormat kMaterialPropertiesFormat = VK_FORMAT_R8G8_UNORM;
		constexpr VkFormat kEmissiveFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
		constexpr VkFormat kNormalFormat = VK_FORMAT_A2R10G10B10_UNORM_PACK32;
		constexpr VkFormat kBrightTexture = VK_FORMAT_A2R10G10B10_UNORM_PACK32;

		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;
		constexpr float kCameraBaseSpeed = 1.7f;
		constexpr float kCameraFastMult = 5.f;
		constexpr float kCameraSlowMult = 0.05f;

		constexpr float kCameraMouseSensitivity = 0.01f;

		constexpr int shadowMapWidth = 8192;
		constexpr int shadowMapHeight = 8192;

		glm::vec3 gLightPosition(0.0f, 0.0f, 0.0f);
		glm::vec3 gLightColor(1.0f, 1.0f, 1.0f);
		float gLightOrbitSpeed;
		bool glightMoving = false;
		glm::vec3 gMoveDirection(-1.0f, 0.0f, 0.0f);
	}

	// GLFW callbacks
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);

	void glfw_callback_button(GLFWwindow*, int, int, int);

	void glfw_callback_motion(GLFWwindow*, double, double);

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		orbit,
		max
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	void update_user_state(UserState&, float aElapsedTime);

	// Uniform data
	namespace glsl
	{
		struct SceneUniform
		{
			//Note: need to be careful about the packing / alignment here!
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::mat4 lightSpaceMatrix;
			glm::vec3 cameraPos;

			alignas(16) glm::vec3 lightPosition;
			alignas(16) glm::vec3 lightColor;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less"
			" than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");

	}



	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);

	void create_swapchain_framebuffers(
		lut::VulkanWindow const&,
		VkRenderPass,
		std::vector<lut::Framebuffer>&,
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const& aState
	);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	namespace cw2
	{
		struct BakedMeshBuffers
		{
			lut::Buffer positions;
			lut::Buffer normals;
			lut::Buffer texcoords;
			lut::Buffer tangents;
			lut::Buffer indices;
		};
		
		std::string extractFilename(const std::string& path);
		bool pathExists(const std::string& path, const std::vector<std::string>& paths);
		int create_data_buffers(lut::VulkanContext const& aContext,
			lut::Allocator const& aAllocator,
			BakedModel& bakedModel,
			std::vector<BakedMeshBuffers>& meshBuffers);

		lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow);
	}

	namespace cw3
	{
		struct MaterialColorUniform
		{ //alignment to respect std::140 layout rules
			alignas(16) glm::vec3 baseColor;
			float roughness;
			float metalness;
			alignas(16)glm::vec3 emissiveColor;
		};

		lut::RenderPass create_render_pass_A(lut::VulkanWindow const& aWindow);
		lut::RenderPass create_render_pass_B(lut::VulkanWindow const& aWindow);
		lut::PipelineLayout create_fullscreen_pipeline_layout(lut::VulkanContext const& aContext,
			VkDescriptorSetLayout aSceneLayout,
			VkDescriptorSetLayout aMaterialLayout,
			VkDescriptorSetLayout aShadowLayout);

		lut::Pipeline create_fullscreen_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout);
		
		//recording and submitting
		void submit_commands_A(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence,
			VkSemaphore imageAvailableSemaphore, VkSemaphore renderASemaphore, VkSemaphore renderBSemaphore);
		void submit_commands_B(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence,
			VkSemaphore imageAvailableSemaphore, VkSemaphore renderASemaphore, VkSemaphore renderBSemaphore);

	}

	namespace cw4
	{
		lut::Sampler create_shadow_sampler(lut::VulkanContext const& aContext);
		std::tuple<lut::Image, lut::ImageView> create_shadow_map_image_view(lut::VulkanWindow const& aWindow,
			lut::Allocator const& aAllocator);
		lut::DescriptorSetLayout create_shadow_map_descriptor_layout(lut::VulkanWindow const& aWindow);
		lut::PipelineLayout create_shadow_pipeline_layout(lut::VulkanContext const& aContext,
			VkDescriptorSetLayout aSceneLayout);
		lut::Pipeline create_shadow_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout);
		lut::Framebuffer create_shadow_framebuffers(lut::VulkanWindow const& aWindow,
			VkRenderPass aRenderPass,
			VkImageView  aDepthView);

		int create_shadow_map_descriptor_set(
			lut::VulkanContext const& aContext,
			VkDescriptorPool aPool,
			VkDescriptorSetLayout const& aShadowLayout,
			lut::ImageView const& aImageView,
			lut::Sampler const& aSampler,
			VkDescriptorSet& descriptorSet);

		void record_commands_A(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass,
			VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsLayout,
			VkExtent2D const& aImageExtent,
			VkBuffer aSceneUBO,
			glsl::SceneUniform const& aSceneUniform,
			VkDescriptorSet aSceneDescriptors,
			std::vector<cw2::BakedMeshBuffers>& materialBuffers,
			BakedModel const& aBakedModel
		);
		void record_commands_B(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass,
			VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsLayout,
			VkExtent2D const& aImageExtent,
			VkBuffer aSceneUBO,
			glsl::SceneUniform const& aSceneUniform,
			VkDescriptorSet aSceneDescriptors,
			VkDescriptorSet aShadowMapDescriptor,
			std::vector<VkDescriptorSet>& aMaterialDescriptors,
			std::vector<cw2::BakedMeshBuffers>& materialBuffers,
			BakedModel const& aBakedModel,
			VkImage shadowMapImage
		);

		int create_material_descriptor_sets(
			lut::VulkanContext const& aContext,
			VkDescriptorPool aPool,
			VkDescriptorSetLayout const& aMaterialLayout,
			std::vector<BakedMaterialInfo> const& aMaterials,
			std::vector<lut::ImageView> const& aImageViews,
			lut::Sampler const& aSampler,
			std::vector<VkDescriptorSet>& descriptorSetVector,
			lut::ImageView const& dummyImageView);
	}
}

int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	// Configure the GLFW window

	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	// Intialize resources
	//shadow geometry pass
	lut::RenderPass renderPassA = cw3::create_render_pass_A(window);
	//geometry pass and final pass
	lut::RenderPass renderPassB = cw3::create_render_pass_B(window);

	// create scene descriptor set layout
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	// create material descriptor set layout
	lut::DescriptorSetLayout materialLayout = cw2::create_material_descriptor_layout(window);
	// create intermediate texture layout
	lut::DescriptorSetLayout shadowMapLayout = cw4::create_shadow_map_descriptor_layout(window);

	//used for render pass A
	lut::PipelineLayout shadowLayout = cw4::create_shadow_pipeline_layout(window, sceneLayout.handle);
	lut::Pipeline shadowPassPipe = cw4::create_shadow_pipeline(window, renderPassA.handle, shadowLayout.handle);

	//used for render pass B
	lut::PipelineLayout fullscreenLayout = cw3::create_fullscreen_pipeline_layout(window, sceneLayout.handle, materialLayout.handle, shadowMapLayout.handle);
	lut::Pipeline fullscreenPipe = cw3::create_fullscreen_pipeline(window, renderPassB.handle, fullscreenLayout.handle);

	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);
	
	// --------------------------------- CW4 ---------------------------------

	auto [shadowDepthBuffer, shadowDepthBufferView] = cw4::create_shadow_map_image_view(window, allocator);
	lut::Framebuffer framebufferA = cw4::create_shadow_framebuffers(window, renderPassA.handle, shadowDepthBufferView.handle);

	std::vector<lut::Framebuffer> framebuffersB;
	create_swapchain_framebuffers(window, renderPassB.handle, framebuffersB, depthBufferView.handle);

	// --------------------------------- CW3 ---------------------------------

	lut::CommandPool cpool_A = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
	lut::CommandPool cpool_B = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	std::vector<VkCommandBuffer> cbuffers_A;
	std::vector<VkCommandBuffer> cbuffers_B;
	std::vector<lut::Fence> cbfences_A;
	std::vector<lut::Fence> cbfences_B;

	cbuffers_A.emplace_back(lut::alloc_command_buffer(window, cpool_A.handle));
	cbfences_A.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	for (std::size_t i = 0; i < framebuffersB.size(); ++i)
	{
		cbuffers_B.emplace_back(lut::alloc_command_buffer(window, cpool_B.handle));
		cbfences_B.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderAFinished = lut::create_semaphore(window);
	lut::Semaphore renderBFinished = lut::create_semaphore(window);

	// create scene uniform buffer with lut::create_buffer()
	lut::Buffer sceneUBO = lut::create_buffer(
		allocator,
		sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	//TODO- (Section 3) create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	// allocate descriptor set for uniform buffer
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	// initialize descriptor set with vkUpdateDescriptorSets
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	//TODO- (Section 4) create default texture sampler
	lut::Sampler defaultSampler = lut::create_default_sampler(window);

	//------------------------------- CW4 ----------------------------------------------
	lut::Sampler shadowSampler = cw4::create_shadow_sampler(window);

	//------------------------------- CW4 ----------------------------------------------
	//CWK3 variables
	std::vector<lut::Image> textureImages;
	std::vector<lut::ImageView> textureImageViews;
	std::vector<cw2::BakedMeshBuffers> bakedMeshBuffers;
	std::vector<VkDescriptorSet> materialTextureDescriptors;
	std::vector<cw3::MaterialColorUniform> materialColorUniforms{};
	lut::Image dummyImage;
	lut::ImageView dummyImageView;

	//loading data
	BakedModel bakedModel = load_baked_model(cfg::bakedModelPath);
	materialTextureDescriptors.resize(bakedModel.materials.size());
	std::vector<std::string> alphaMaskTexturePaths{};
	std::vector<std::string> normalMapTexturePaths{};
	
	for (std::size_t i = 0; i < bakedModel.materials.size(); i++)
	{
		if (bakedModel.materials[i].alphaMaskTextureId != 0xffffffff)
		{
			alphaMaskTexturePaths.push_back(bakedModel.textures[bakedModel.materials[i].alphaMaskTextureId].path);
		}
		if (bakedModel.materials[i].normalMapTextureId != 0xffffffff)
		{
			normalMapTexturePaths.push_back(bakedModel.textures[bakedModel.materials[i].normalMapTextureId].path);
		}
	}

	//upload mesh data -- same for cw3 and cw4?
	cw2::create_data_buffers(window, allocator, bakedModel, bakedMeshBuffers);

	//create textureImages and ImageViews
	for (std::size_t i = 0; i < bakedModel.textures.size(); i++)
	{
		lut::CommandPool loadCmdPool = lut::create_command_pool(
			window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
		std::string filename = cw2::extractFilename(bakedModel.textures[i].path);


		if (filename[0] == 'r' ||
			filename[0] == 'm' ||
			cw2::pathExists(bakedModel.textures[i].path, alphaMaskTexturePaths) == true ||
			cw2::pathExists(bakedModel.textures[i].path, normalMapTexturePaths) == true)
		{
			lut::Image img = lut::load_image_texture2d(
				bakedModel.textures[i].path.c_str(),
				window,
				loadCmdPool.handle,
				allocator,
				VK_FORMAT_R8G8B8A8_UNORM);
			textureImages.push_back(std::move(img));

			lut::ImageView imgView = lut::create_image_view_texture2D(window,
				textureImages[i].image,
				VK_FORMAT_R8G8B8A8_UNORM);
			textureImageViews.push_back(std::move(imgView));
		}
		else
		{ //base color
			lut::Image img = lut::load_image_texture2d(
				bakedModel.textures[i].path.c_str(),
				window,
				loadCmdPool.handle,
				allocator,
				VK_FORMAT_R8G8B8A8_SRGB);
			textureImages.push_back(std::move(img));

			lut::ImageView imgView = lut::create_image_view_texture2D(window,
				textureImages[i].image,
				VK_FORMAT_R8G8B8A8_SRGB);
			textureImageViews.push_back(std::move(imgView));
		}

	}

	//create 1 descriptor set per material texture ( 3 image views in 1 descriptor set)
	cw4::create_material_descriptor_sets(window, dpool.handle, materialLayout.handle,
		bakedModel.materials, textureImageViews, defaultSampler, materialTextureDescriptors, dummyImageView);

	VkDescriptorSet shadowMapDescriptor{};
	cw4::create_shadow_map_descriptor_set(window, dpool.handle, shadowMapLayout.handle, shadowDepthBufferView,
		shadowSampler, shadowMapDescriptor);

	//------------------------------- CW3 - end ----------------------------------------------
	// Application main loop
	bool recreateSwapchain = false;

	auto previousClock = Clock_::now();

	while (!glfwWindowShouldClose(window.window))
	{
		glfwPollEvents();
		// Recreate swap chain?
		if (recreateSwapchain)
		{
			//TODO: (Section 1) re-create swapchain and associated resources - see Exercise 3!
			//We need to destroy several objects, which may still be in use by the GPU
			vkDeviceWaitIdle(window.device);
			//Recreate them
			auto const changes = recreate_swapchain(window);
			if (changes.changedFormat)
			{
				renderPassA = cw3::create_render_pass_A(window);
				renderPassB = cw3::create_render_pass_B(window);
			}
			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				std::tie(shadowDepthBuffer, shadowDepthBufferView) = cw4::create_shadow_map_image_view(window, allocator);
				
			}

			framebuffersB.clear();
			framebufferA = cw4::create_shadow_framebuffers(window, renderPassA.handle, shadowDepthBufferView.handle);
			create_swapchain_framebuffers(window, renderPassB.handle, framebuffersB, depthBufferView.handle);

			if (changes.changedSize)
			{
				shadowPassPipe = cw4::create_shadow_pipeline(window, renderPassA.handle, shadowLayout.handle);
				fullscreenPipe = cw3::create_fullscreen_pipeline(window, renderPassB.handle, fullscreenLayout.handle);
			}
			recreateSwapchain = false;
			continue;
		}


		//acquire swapchain image.
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		// Recreate swap chain?
		if (VK_SUBOPTIMAL_KHR == acquireRes ||
			VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			// occurs when window is resized
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire next swapchain image\n"
				"vkAcquireNextImageKHR() returned %s",
				lut::to_string(acquireRes).c_str());
		}
		assert(std::size_t(imageIndex) < cbfences_B.size());
		//wait for fence for command buffer A
		if (auto const res = vkWaitForFences(window.device,
			1,
			&cbfences_A[0].handle,
			VK_TRUE,
			std::numeric_limits<std::uint64_t>::max());
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s",
				0,
				lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device,
			1,
			&cbfences_A[0].handle);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s",
				0,
				lut::to_string(res).c_str());
		}
		// wait for command buffer B to be available
		if (auto const res = vkWaitForFences(window.device,
			1,
			&cbfences_B[imageIndex].handle,
			VK_TRUE,
			std::numeric_limits<std::uint64_t>::max());
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n"
				"vkWaitForFences() returned %s",
				imageIndex,
				lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device,
			1,
			&cbfences_B[imageIndex].handle);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n"
				"vkResetFences() returned %s",
				imageIndex,
				lut::to_string(res).c_str());
		}

		//Update state
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;

		update_user_state(state, dt);

		//Prepare data for this frame
		glsl::SceneUniform sceneUniforms{};
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state);


		//TODO- (Section 1)  - record and submit commands
		assert(std::size_t(imageIndex) < cbuffers_B.size());
		assert(std::size_t(imageIndex) < framebuffersB.size());


		cw4::record_commands_A(cbuffers_A[0], renderPassA.handle, framebufferA.handle, shadowPassPipe.handle,
			shadowLayout.handle, window.swapchainExtent,
			sceneUBO.buffer, sceneUniforms, sceneDescriptors, bakedMeshBuffers, bakedModel);
		cw4::record_commands_B(cbuffers_B[imageIndex], renderPassB.handle, framebuffersB[imageIndex].handle,
			fullscreenPipe.handle, fullscreenLayout.handle,
			window.swapchainExtent, sceneUBO.buffer, sceneUniforms, sceneDescriptors, shadowMapDescriptor,
			materialTextureDescriptors,
			bakedMeshBuffers, bakedModel, shadowDepthBuffer.image);

		cw3::submit_commands_A(
			window,
			cbuffers_A[0],
			cbfences_A[0].handle,
			imageAvailable.handle,
			renderAFinished.handle,
			renderBFinished.handle
		);

		cw3::submit_commands_B(
			window,
			cbuffers_B[imageIndex],
			cbfences_B[imageIndex].handle,
			imageAvailable.handle,
			renderAFinished.handle,
			renderBFinished.handle
		);

		//TODO- (Section 1)  - present rendered images (note: use the present_results() method)
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderBFinished.handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			recreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n"
				"vkQueuePresentKHR() returned %s",
				imageIndex,
				lut::to_string(presentRes).c_str());
		}
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle(window.device);


	return 0;
}
catch (std::exception const& eErr)
{
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, "Error: %s\n", eErr.what());
	return 1;
}


namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}

		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;


		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;
		case GLFW_KEY_SPACE:
			state->inputMap[std::size_t(EInputState::orbit)] = !isReleased;
			break;

		default:
			;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
			{
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else
			{
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			//Only update the rotation on the second frame of mouse nav
			//ensures that previousX and Y variables are initialised to sensible values

			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}

		auto const move = aElapsedTime * cfg::kCameraBaseSpeed *
			(aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);
		cfg::gLightOrbitSpeed = aElapsedTime * 2.0f;


		if (aState.inputMap[std::size_t(EInputState::forward)])
		{
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		}
		if (aState.inputMap[std::size_t(EInputState::backward)])
		{
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));
		}

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
		{
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		}
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
		{
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));
		}

		if (aState.inputMap[std::size_t(EInputState::levitate)])
		{
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		}
		if (aState.inputMap[std::size_t(EInputState::sink)])
		{
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));
		}
		if (aState.inputMap[std::size_t(EInputState::orbit)])
		{
			//switching states for light movement
			cfg::glightMoving = !cfg::glightMoving;
		}
	}
}

namespace
{
	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight,
		UserState const& aState)
	{
		//TODO- (Section 3) initialize SceneUniform members
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);

		aSceneUniforms.projection[1][1] *= -1.f; //Mirror y axis

		aSceneUniforms.camera = glm::inverse(aState.camera2world);

		aSceneUniforms.cameraPos = aState.camera2world[3];

		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;

		if (cfg::glightMoving == true)
		{

			if (cfg::gLightPosition.x < -3.0f)
			{
				cfg::gMoveDirection = glm::vec3(1.0f, 0.0f, 0.0f);
			}
			if (cfg::gLightPosition.x > 3.0f)
			{
				cfg::gMoveDirection = glm::vec3(-1.0f, 0.0f, 0.0f);
			}
			cfg::gLightPosition = cfg::gLightPosition + cfg::gLightOrbitSpeed * cfg::gMoveDirection;
		}

		aSceneUniforms.lightPosition = cfg::gLightPosition;
		aSceneUniforms.lightColor = cfg::gLightColor;
		glm::vec3 direction = aSceneUniforms.lightPosition + glm::vec3(0.0f, 3.0f, -16.0f);
		glm::mat4 lightView = glm::lookAt(
			aSceneUniforms.lightPosition,
			direction,
			glm::vec3(0.0f, 1.0f, 0.0f)
		);
		glm::mat4 lightProjection = glm::perspective(
			lut::Radians(90.0_degf).value(),
			cfg::shadowMapWidth / (float)cfg::shadowMapHeight,
			cfg::kCameraNear,
			100.0f
		);

		aSceneUniforms.lightSpaceMatrix = lightProjection * lightView;

	}

	namespace cw2
	{
		std::string extractFilename(const std::string& path)
		{
			std::size_t position = path.find_last_of("/\\");
			return (position != std::string::npos) ? path.substr(position + 1) : path;
		}

		bool pathExists(const std::string& path, const std::vector<std::string>& paths)
		{
			bool ok = false;
			for (std::size_t i = 0; i < paths.size(); i++)
			{
				if (path == paths[i])
				{
					ok = true;
					break;
				}
			}
			return ok;
		}

		lut::DescriptorSetLayout create_material_descriptor_layout(lut::VulkanWindow const& aWindow)
		{
			VkDescriptorSetLayoutBinding bindings[4]{};

			//Base Color
			bindings[0].binding = 0;
			bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[0].descriptorCount = 1;
			bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings[0].pImmutableSamplers = nullptr;

			//Roughness texture
			bindings[1].binding = 1;
			bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[1].descriptorCount = 1;
			bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings[1].pImmutableSamplers = nullptr;

			//Metal texture
			bindings[2].binding = 2;
			bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[2].descriptorCount = 1;
			bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings[2].pImmutableSamplers = nullptr;

			//Normal map texture
			bindings[3].binding = 3;
			bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[3].descriptorCount = 1;
			bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings[3].pImmutableSamplers = nullptr;

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
			layoutInfo.pBindings = bindings;

			VkDescriptorSetLayout layout = VK_NULL_HANDLE;
			if (auto const res = vkCreateDescriptorSetLayout(
				aWindow.device,
				&layoutInfo,
				nullptr,
				&layout);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create descriptor set layout\n"
					"vkCreateDescriptorSetLayout() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::DescriptorSetLayout(aWindow.device, layout);
		}

		int create_data_buffers(lut::VulkanContext const& aContext,
			lut::Allocator const& aAllocator,
			BakedModel& bakedModel,
			std::vector<BakedMeshBuffers>& meshBuffers)
		{
			meshBuffers.clear();
			//for all meshes in bakedModel
			for (std::size_t i = 0; i < bakedModel.meshes.size(); i++)
			{
				std::vector<float> pos = {};
				std::vector<float> norm = {};
				std::vector<float> tex = {};
				std::vector<float> tangent = {};

				for (std::size_t j = 0; j < bakedModel.meshes[i].positions.size(); j++)
				{
					pos.push_back(bakedModel.meshes[i].positions[j].x);
					pos.push_back(bakedModel.meshes[i].positions[j].y);
					pos.push_back(bakedModel.meshes[i].positions[j].z);
				}
				for (std::size_t j = 0; j < bakedModel.meshes[i].normals.size(); j++)
				{
					norm.push_back(bakedModel.meshes[i].normals[j].x);
					norm.push_back(bakedModel.meshes[i].normals[j].y);
					norm.push_back(bakedModel.meshes[i].normals[j].z);
				}
				for (std::size_t j = 0; j < bakedModel.meshes[i].texcoords.size(); j++)
				{
					tex.push_back(bakedModel.meshes[i].texcoords[j].x);
					tex.push_back(bakedModel.meshes[i].texcoords[j].y);
				}
				for (std::size_t j = 0; j < bakedModel.meshes[i].tangents.size(); j++)
				{
					tangent.push_back(bakedModel.meshes[i].tangents[j].x);
					tangent.push_back(bakedModel.meshes[i].tangents[j].y);
					tangent.push_back(bakedModel.meshes[i].tangents[j].z);
					tangent.push_back(bakedModel.meshes[i].tangents[j].w);
				}

				std::size_t posSize = pos.size() * sizeof(float);
				std::size_t normSize = norm.size() * sizeof(float);
				std::size_t texSize = tex.size() * sizeof(float);
				std::size_t tanSize = tangent.size() * sizeof(float);
				std::size_t indicesSize = bakedModel.meshes[i].indices.size() * sizeof(std::uint32_t);

				lut::Buffer vertexPosGPU = lut::create_buffer(
					aAllocator,
					posSize,
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VMA_MEMORY_USAGE_GPU_ONLY
				);
				lut::Buffer vertexNormGPU = lut::create_buffer(
					aAllocator,
					normSize,
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VMA_MEMORY_USAGE_GPU_ONLY
				);
				lut::Buffer vertexTexGPU = lut::create_buffer(
					aAllocator,
					texSize,
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VMA_MEMORY_USAGE_GPU_ONLY
				);
				lut::Buffer vertexTangentGPU = lut::create_buffer(
					aAllocator,
					tanSize,
					VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VMA_MEMORY_USAGE_GPU_ONLY
				);
				lut::Buffer vertexIndexGPU = lut::create_buffer(
					aAllocator,
					indicesSize,
					VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
					VMA_MEMORY_USAGE_GPU_ONLY
				);


				//staging buffers
				lut::Buffer posStaging = lut::create_buffer(
					aAllocator,
					posSize,
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VMA_MEMORY_USAGE_CPU_TO_GPU
				);
				lut::Buffer normStaging = lut::create_buffer(
					aAllocator,
					normSize,
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VMA_MEMORY_USAGE_CPU_TO_GPU
				);
				lut::Buffer texStaging = lut::create_buffer(
					aAllocator,
					texSize,
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VMA_MEMORY_USAGE_CPU_TO_GPU
				);
				lut::Buffer tangentStaging = lut::create_buffer(
					aAllocator,
					tanSize,
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VMA_MEMORY_USAGE_CPU_TO_GPU
				);
				lut::Buffer indexStaging = lut::create_buffer(
					aAllocator,
					indicesSize,
					VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					VMA_MEMORY_USAGE_CPU_TO_GPU
				);

				void* posPtr = nullptr;
				if (auto const res = vmaMapMemory(aAllocator.allocator,
					posStaging.allocation,
					&posPtr);
					VK_SUCCESS != res)
				{
					throw lut::Error("Mapping memory for writing\n"
						"vmaMapMemory() returned %s",
						lut::to_string(res).c_str());
				}
				std::memcpy(posPtr, static_cast<const void*>(pos.data()), posSize);
				vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

				void* normPtr = nullptr;
				if (auto const res = vmaMapMemory(aAllocator.allocator,
					normStaging.allocation,
					&normPtr);
					VK_SUCCESS != res)
				{
					throw lut::Error("Mapping memory for writing\n"
						"vmaMapMemory() returned %s",
						lut::to_string(res).c_str());
				}
				std::memcpy(normPtr, static_cast<const void*>(norm.data()), normSize);
				vmaUnmapMemory(aAllocator.allocator, normStaging.allocation);

				void* texPtr = nullptr;
				if (auto const res = vmaMapMemory(aAllocator.allocator,
					texStaging.allocation,
					&texPtr);
					VK_SUCCESS != res)
				{
					throw lut::Error("Mapping memory for writing\n"
						"vmaMapMemory() returned %s",
						lut::to_string(res).c_str());
				}
				std::memcpy(texPtr, static_cast<const void*>(tex.data()), texSize);
				vmaUnmapMemory(aAllocator.allocator, texStaging.allocation);

				void* tanPtr = nullptr;
				if (auto const res = vmaMapMemory(aAllocator.allocator,
					tangentStaging.allocation,
					&tanPtr);
					VK_SUCCESS != res)
				{
					throw lut::Error("Mapping memory for writing\n"
						"vmaMapMemory() returned %s",
						lut::to_string(res).c_str());
				}
				std::memcpy(tanPtr, static_cast<const void*>(tangent.data()), tanSize);
				vmaUnmapMemory(aAllocator.allocator, tangentStaging.allocation);

				void* indexPtr = nullptr;
				if (auto const res = vmaMapMemory(aAllocator.allocator,
					indexStaging.allocation,
					&indexPtr);
					VK_SUCCESS != res)
				{
					throw lut::Error("Mapping memory for writing\n"
						"vmaMapMemory() returned %s",
						lut::to_string(res).c_str());
				}
				std::memcpy(indexPtr, static_cast<const void*>(bakedModel.meshes[i].indices.data()), indicesSize);
				vmaUnmapMemory(aAllocator.allocator, indexStaging.allocation);

				// We need to ensure that the Vulkan resources are alive until all the
				// transfers have completed
				lut::Fence uploadComplete = create_fence(aContext);

				//Queue data uploads from staging buffers to the final buffers
				//This uses a separate command pool for simplicity
				lut::CommandPool uploadPool = create_command_pool(aContext);
				VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

				VkCommandBufferBeginInfo beginInfo{};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				beginInfo.flags = 0;
				beginInfo.pInheritanceInfo = nullptr;

				if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo);
					VK_SUCCESS != res)
				{
					throw lut::Error("Beginning command buffer recording\n"
						"vkBeginCommandBuffer() returned %s",
						lut::to_string(res).c_str());
				}

				VkBufferCopy pcopy{};
				pcopy.size = posSize;
				vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);
				lut::buffer_barrier(uploadCmd,
					vertexPosGPU.buffer,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

				VkBufferCopy ncopy{};
				ncopy.size = normSize;
				vkCmdCopyBuffer(uploadCmd, normStaging.buffer, vertexNormGPU.buffer, 1, &ncopy);
				lut::buffer_barrier(uploadCmd,
					vertexNormGPU.buffer,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

				VkBufferCopy tcopy{};
				tcopy.size = texSize;
				vkCmdCopyBuffer(uploadCmd, texStaging.buffer, vertexTexGPU.buffer, 1, &tcopy);
				lut::buffer_barrier(uploadCmd,
					vertexTexGPU.buffer,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

				VkBufferCopy tancopy{};
				tancopy.size = tanSize;
				vkCmdCopyBuffer(uploadCmd, tangentStaging.buffer, vertexTangentGPU.buffer, 1, &tancopy);
				lut::buffer_barrier(uploadCmd,
					vertexTangentGPU.buffer,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);

				VkBufferCopy icopy{};
				icopy.size = indicesSize;
				vkCmdCopyBuffer(uploadCmd, indexStaging.buffer, vertexIndexGPU.buffer, 1, &icopy);
				lut::buffer_barrier(uploadCmd,
					vertexIndexGPU.buffer,
					VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_ACCESS_INDEX_READ_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);


				if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
				{
					throw lut::Error("Ending command buffer recording\n"
						"vkEndCommandBuffer() returned %s",
						lut::to_string(res).c_str());
				}

				// Submit transfer commands
				VkSubmitInfo submitInfo{};
				submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
				submitInfo.commandBufferCount = 1;
				submitInfo.pCommandBuffers = &uploadCmd;
				if (auto const res = vkQueueSubmit(aContext.graphicsQueue,
					1,
					&submitInfo,
					uploadComplete.handle);
					VK_SUCCESS != res)
				{
					throw lut::Error("Submitting commands\n"
						"vkQueueSubmit() returned %s",
						lut::to_string(res).c_str());
				}

				//// Wait for commands to finish before we destroy the temporary resources
				//// required for the transfers (staging buffers, command pool, ...)
				//// not destroyed implicitly, but by destructors of labutils wrappers

				if (auto const res = vkWaitForFences(aContext.device,
					1,
					&uploadComplete.handle,
					VK_TRUE,
					std::numeric_limits<std::uint64_t>::max());
					VK_SUCCESS != res)
				{
					throw lut::Error("Waiting for upload to complete\n"
						"vkWaitForFences() returned %s",
						lut::to_string(res).c_str());
				}

				BakedMeshBuffers currentBuffers = BakedMeshBuffers{
					std::move(vertexPosGPU),
					std::move(vertexNormGPU),
					std::move(vertexTexGPU),
					std::move(vertexTangentGPU),
					std::move(vertexIndexGPU)
				};
				meshBuffers.push_back(std::move(currentBuffers));

			}
			return 1;
		}
	} //--end cw2 namespace

	namespace cw3
	{

		lut::RenderPass create_render_pass_A(lut::VulkanWindow const& aWindow)
		{
			VkAttachmentDescription attachments[1]{};
			//depth buffer
			attachments[0].format = cfg::kDepthFormat;
			attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentReference depthAttachment{};
			depthAttachment.attachment = 0;
			depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpasses[1]{};
			subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpasses[0].pDepthStencilAttachment = &depthAttachment;

			// Subpass dependencies
			VkSubpassDependency dependency[1]{};
			dependency[0].srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency[0].dstSubpass = 0;
			dependency[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependency[0].srcAccessMask = 0;
			dependency[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
			dependency[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			passInfo.attachmentCount = 1;
			passInfo.pAttachments = attachments;
			passInfo.subpassCount = 1;
			passInfo.pSubpasses = subpasses;
			passInfo.dependencyCount = 1;
			passInfo.pDependencies = dependency;

			std::cout << "Creating render pass A" << std::endl;
			VkRenderPass rpass = VK_NULL_HANDLE;
			if (auto const res = vkCreateRenderPass(aWindow.device,
				&passInfo,
				nullptr,
				&rpass);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create render pass A\n"
					"vkCreateRenderPass() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::RenderPass(aWindow.device, rpass);

		}

		lut::RenderPass create_render_pass_B(lut::VulkanWindow const& aWindow)
		{
			VkAttachmentDescription attachments[2]{};
			attachments[0].format = aWindow.swapchainFormat;
			attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			attachments[1].format = cfg::kDepthFormat;
			attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
			attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkAttachmentReference subpassAttachments[1]{};
			subpassAttachments[0].attachment = 0;
			subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkAttachmentReference depthAttachment{};
			depthAttachment.attachment = 1;
			depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpasses[1]{};
			subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpasses[0].colorAttachmentCount = 1;
			subpasses[0].pColorAttachments = subpassAttachments;
			subpasses[0].pDepthStencilAttachment = &depthAttachment;

			VkRenderPassCreateInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			passInfo.attachmentCount = 2;
			passInfo.pAttachments = attachments;
			passInfo.subpassCount = 1;
			passInfo.pSubpasses = subpasses;
			passInfo.dependencyCount = 0;
			passInfo.pDependencies = nullptr;

			std::cout << "Creating render pass B" << std::endl;
			VkRenderPass rpass = VK_NULL_HANDLE;
			if (auto const res = vkCreateRenderPass(aWindow.device,
				&passInfo,
				nullptr,
				&rpass);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create render pass B\n"
					"vkCreateRenderPass() returned %s",
					lut::to_string(res).c_str());
			}
			return lut::RenderPass(aWindow.device, rpass);
		}

		lut::PipelineLayout create_fullscreen_pipeline_layout(lut::VulkanContext const& aContext,
			VkDescriptorSetLayout aSceneLayout,
			VkDescriptorSetLayout aMaterialLayout,
			VkDescriptorSetLayout aShadowLayout)
		{
			VkDescriptorSetLayout layouts[] = {
				//Order must match the set = N in the shaders
				aSceneLayout, //set 0
				aMaterialLayout, // set 1
				aShadowLayout //set 2
			};

			VkPipelineLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
			layoutInfo.pSetLayouts = layouts;
			layoutInfo.pushConstantRangeCount = 0;
			layoutInfo.pPushConstantRanges = nullptr;

			VkPipelineLayout layout = VK_NULL_HANDLE;
			if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create pipeline layout\n"
					"VkCreatePipelineLayout() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::PipelineLayout(aContext.device, layout);
		}

		lut::Pipeline create_fullscreen_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
		{
			lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kRenderPassBVertShaderPath);
			lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kRenderPassBFragShaderPath);

			VkPipelineShaderStageCreateInfo stages[2]{};
			stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
			stages[0].module = vert.handle;
			stages[0].pName = "main";

			stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			stages[1].module = frag.handle;
			stages[1].pName = "main";

			VkVertexInputBindingDescription vertexInputs[4]{};
			vertexInputs[0].binding = 0; //position
			vertexInputs[0].stride = sizeof(float) * 3; //Changed for 3D
			vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			vertexInputs[1].binding = 1; //normals
			vertexInputs[1].stride = sizeof(float) * 3;
			vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			vertexInputs[2].binding = 2; //texcoords
			vertexInputs[2].stride = sizeof(float) * 2;
			vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			vertexInputs[3].binding = 3; //tangents
			vertexInputs[3].stride = sizeof(float) * 4;
			vertexInputs[3].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			VkVertexInputAttributeDescription vertexAttributes[4]{};
			vertexAttributes[0].binding = 0;
			vertexAttributes[0].location = 0;
			vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT; //changed
			vertexAttributes[0].offset = 0;

			vertexAttributes[1].binding = 1;
			vertexAttributes[1].location = 1;
			vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
			vertexAttributes[1].offset = 0;

			vertexAttributes[2].binding = 2;
			vertexAttributes[2].location = 2;
			vertexAttributes[2].format = VK_FORMAT_R32G32_SFLOAT;
			vertexAttributes[2].offset = 0;

			vertexAttributes[3].binding = 3;
			vertexAttributes[3].location = 3;
			vertexAttributes[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
			vertexAttributes[3].offset = 0;

			VkPipelineVertexInputStateCreateInfo inputInfo{};
			inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			inputInfo.vertexBindingDescriptionCount = 4; //number of vertexInputs above;
			inputInfo.pVertexBindingDescriptions = vertexInputs;
			inputInfo.vertexAttributeDescriptionCount = 4;
			inputInfo.pVertexAttributeDescriptions = vertexAttributes;

			//Define which primitive the input is
			//assembled into for rasterization
			VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
			assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			assemblyInfo.primitiveRestartEnable = VK_FALSE;

			//Tesselation state (not in ex2)

			//Define Viewport and scissor regions
			VkViewport viewport{};
			viewport.x = 0.f;
			viewport.y = 0.f;
			viewport.width = float(aWindow.swapchainExtent.width);
			viewport.height = float(aWindow.swapchainExtent.height);
			viewport.minDepth = 0.f;
			viewport.maxDepth = 1.f;

			VkRect2D scissor{};
			scissor.offset = VkOffset2D{ 0, 0 };
			scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

			//DEPTH test disabled for second render pass
			VkPipelineDepthStencilStateCreateInfo depthInfo{};
			depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthInfo.depthTestEnable = VK_TRUE;
			depthInfo.depthWriteEnable = VK_TRUE;
			depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
			depthInfo.minDepthBounds = 0.f;
			depthInfo.maxDepthBounds = 1.f;

			VkPipelineViewportStateCreateInfo viewportInfo{};
			viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportInfo.viewportCount = 1;
			viewportInfo.pViewports = &viewport;
			viewportInfo.scissorCount = 1;
			viewportInfo.pScissors = &scissor;

			//Rasterization State - most states exist in OpenGL as well
			VkPipelineRasterizationStateCreateInfo rasterInfo{};
			rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterInfo.depthClampEnable = VK_FALSE;
			rasterInfo.rasterizerDiscardEnable = VK_FALSE;
			rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
			//rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
			rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rasterInfo.lineWidth = 1.f; //required

			//Multisample State
			VkPipelineMultisampleStateCreateInfo samplingInfo{};
			samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

			//Depth / Stencil State - not used in Exercise 2

			//Color Blend State
			//We define one blend state per color attachment
			//this example uses a single color attachment, so we only need one.
			//we don't do any blending, so we can ignore most of the members;
			VkPipelineColorBlendAttachmentState blendStates[1]{};
			blendStates[0].blendEnable = VK_FALSE;
			blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT
				| VK_COLOR_COMPONENT_G_BIT
				| VK_COLOR_COMPONENT_B_BIT
				| VK_COLOR_COMPONENT_A_BIT;

			VkPipelineColorBlendStateCreateInfo blendInfo{};
			blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			blendInfo.logicOpEnable = VK_FALSE;
			blendInfo.attachmentCount = 1;
			blendInfo.pAttachments = blendStates;

			//Dynamic States - exervise 2 does not use any dynamic state
			// we draw to a fixed-size image/framebuffer

			//Create pipeline
			VkGraphicsPipelineCreateInfo pipeInfo{};
			pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

			pipeInfo.stageCount = 2; // vertex + fragment shader
			pipeInfo.pStages = stages;

			pipeInfo.pVertexInputState = &inputInfo;
			pipeInfo.pInputAssemblyState = &assemblyInfo;
			pipeInfo.pTessellationState = nullptr;
			pipeInfo.pViewportState = &viewportInfo;
			pipeInfo.pRasterizationState = &rasterInfo;
			pipeInfo.pMultisampleState = &samplingInfo;
			pipeInfo.pDepthStencilState = &depthInfo;
			pipeInfo.pColorBlendState = &blendInfo;
			pipeInfo.pDynamicState = nullptr;

			pipeInfo.layout = aPipelineLayout;
			pipeInfo.renderPass = aRenderPass;
			pipeInfo.subpass = 0; //first subpass of aRenderPass

			VkPipeline pipe = VK_NULL_HANDLE;
			if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
				VK_NULL_HANDLE, //not using pipeline cache
				1,
				&pipeInfo,
				nullptr,
				&pipe);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create graphics pipeline\n"
					"vkCreateGraphicsPipelines() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::Pipeline(aWindow.device, pipe);
		}

		void submit_commands_A(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence,
			VkSemaphore imageAvailableSemaphore, VkSemaphore renderASemaphore, VkSemaphore renderBSemaphore)
		{

			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &aCmdBuff;

			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = &renderASemaphore;

			if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to submit command buffer to queue\n"
					"vkQueueSubmit() returned %s",
					lut::to_string(res).c_str());
			}
		}


		void submit_commands_B(lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence,
			VkSemaphore imageAvailableSemaphore, VkSemaphore renderASemaphore, VkSemaphore renderBSemaphore)
		{
			VkPipelineStageFlags waitPipelineStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
			VkSemaphore waitSemaphores[] = { imageAvailableSemaphore, renderASemaphore };


			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &aCmdBuff;

			submitInfo.waitSemaphoreCount = 2;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitPipelineStages;

			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = &renderBSemaphore;

			if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to submit command buffer to queue\n"
					"vkQueueSubmit() returned %s",
					lut::to_string(res).c_str());
			}
		}


	}

	namespace cw4
	{
		lut::Sampler create_shadow_sampler(lut::VulkanContext const& aContext)
		{
			VkSamplerCreateInfo samplerInfo{};
			samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
			samplerInfo.magFilter = VK_FILTER_LINEAR;
			samplerInfo.minFilter = VK_FILTER_LINEAR;
			samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
			samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			samplerInfo.anisotropyEnable = VK_FALSE;
			samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
			samplerInfo.compareEnable = VK_TRUE;
			samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

			VkSampler sampler = VK_NULL_HANDLE;
			if (auto const res = vkCreateSampler(aContext.device,
				&samplerInfo,
				nullptr,
				&sampler);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create sampler\n"
					"vkCreateSampler() returned %s",
					lut::to_string(res).c_str());
			}
			return lut::Sampler(aContext.device, sampler);
		}


		std::tuple<lut::Image, lut::ImageView> create_shadow_map_image_view(lut::VulkanWindow const& aWindow,
			lut::Allocator const& aAllocator)
		{
			VkImageCreateInfo imageInfo{};
			imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.format = cfg::kDepthFormat;
			imageInfo.extent.width = cfg::shadowMapWidth;
			imageInfo.extent.height = cfg::shadowMapHeight;
			imageInfo.extent.depth = 1;
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

			VmaAllocationCreateInfo allocInfo{};
			allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

			VkImage image = VK_NULL_HANDLE;
			VmaAllocation allocation = VK_NULL_HANDLE;

			if (auto const res = vmaCreateImage(aAllocator.allocator,
				&imageInfo,
				&allocInfo,
				&image,
				&allocation,
				nullptr);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to allocate depth buffer image.\n"
					"vmaCreateImage() returned %s",
					lut::to_string(res).c_str());
			}

			lut::Image depthImage(aAllocator.allocator, image, allocation);

			//Create the image view
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = depthImage.image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = cfg::kDepthFormat;
			viewInfo.components = VkComponentMapping{};
			viewInfo.subresourceRange = VkImageSubresourceRange{
				VK_IMAGE_ASPECT_DEPTH_BIT,
				0, 1,
				0, 1
			};

			VkImageView view = VK_NULL_HANDLE;
			if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create image view\n"
					"vkCreateImageView() returned %s",
					lut::to_string(res).c_str());
			}

			return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
		}


		lut::DescriptorSetLayout create_shadow_map_descriptor_layout(lut::VulkanWindow const& aWindow)
		{
			VkDescriptorSetLayoutBinding bindings[1]{};

			// Depth texture
			bindings[0].binding = 0;
			bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[0].descriptorCount = 1;
			bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
			bindings[0].pImmutableSamplers = nullptr;

			VkDescriptorSetLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
			layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
			layoutInfo.pBindings = bindings;

			VkDescriptorSetLayout layout = VK_NULL_HANDLE;
			if (auto const res = vkCreateDescriptorSetLayout(
				aWindow.device,
				&layoutInfo,
				nullptr,
				&layout);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create descriptor set layout\n"
					"vkCreateDescriptorSetLayout() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::DescriptorSetLayout(aWindow.device, layout);
		}


		lut::PipelineLayout create_shadow_pipeline_layout(lut::VulkanContext const& aContext,
			VkDescriptorSetLayout aSceneLayout)
		{
			VkDescriptorSetLayout layouts[] = {
				//Order must match the set = N in the shaders
				aSceneLayout //set 0
			};

			VkPipelineLayoutCreateInfo layoutInfo{};
			layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
			layoutInfo.pSetLayouts = layouts;
			layoutInfo.pushConstantRangeCount = 0;
			layoutInfo.pPushConstantRanges = nullptr;

			VkPipelineLayout layout = VK_NULL_HANDLE;
			if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create pipeline layout\n"
					"VkCreatePipelineLayout() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::PipelineLayout(aContext.device, layout);
		}


		lut::Pipeline create_shadow_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
		{
			lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kRenderPassAVertShaderPath);

			VkPipelineShaderStageCreateInfo stages[1]{};
			stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
			stages[0].module = vert.handle;
			stages[0].pName = "main";

			VkVertexInputBindingDescription vertexInputs[1]{};
			vertexInputs[0].binding = 0; //position
			vertexInputs[0].stride = sizeof(float) * 3; //Changed for 3D
			vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			VkVertexInputAttributeDescription vertexAttributes[1]{};
			vertexAttributes[0].binding = 0;
			vertexAttributes[0].location = 0;
			vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT; //changed
			vertexAttributes[0].offset = 0;

			VkPipelineVertexInputStateCreateInfo inputInfo{};
			inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			inputInfo.vertexBindingDescriptionCount = 1; //number of vertexInputs above;
			inputInfo.pVertexBindingDescriptions = vertexInputs;
			inputInfo.vertexAttributeDescriptionCount = 1;
			inputInfo.pVertexAttributeDescriptions = vertexAttributes;

			//Define which primitive the input is
			//assembled into for rasterization
			VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
			assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			assemblyInfo.primitiveRestartEnable = VK_FALSE;

			//Tesselation state (not in ex2)

			//Define Viewport and scissor regions
			VkViewport viewport{};
			viewport.x = 0.f;
			viewport.y = 0.f;
			viewport.width = float(cfg::shadowMapWidth);
			viewport.height = float(cfg::shadowMapHeight);
			viewport.minDepth = 0.f;
			viewport.maxDepth = 1.f;

			VkRect2D scissor{};
			scissor.offset = VkOffset2D{ 0, 0 };
			scissor.extent = VkExtent2D{ cfg::shadowMapWidth, cfg::shadowMapHeight };

			//DEPTH INFO FOR DEPTH BUFFER
			VkPipelineDepthStencilStateCreateInfo depthInfo{};
			depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthInfo.depthTestEnable = VK_TRUE;
			depthInfo.depthWriteEnable = VK_TRUE;
			depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
			depthInfo.minDepthBounds = 0.f;
			depthInfo.maxDepthBounds = 1.f;

			VkPipelineViewportStateCreateInfo viewportInfo{};
			viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportInfo.viewportCount = 1;
			viewportInfo.pViewports = &viewport;
			viewportInfo.scissorCount = 1;
			viewportInfo.pScissors = &scissor;

			//Rasterization State - most states exist in OpenGL as well
			VkPipelineRasterizationStateCreateInfo rasterInfo{};
			rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterInfo.depthClampEnable = VK_FALSE;
			rasterInfo.rasterizerDiscardEnable = VK_FALSE;
			rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
			rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
			rasterInfo.lineWidth = 1.f; //required

			//Multisample State
			VkPipelineMultisampleStateCreateInfo samplingInfo{};
			samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

			//Create pipeline
			VkGraphicsPipelineCreateInfo pipeInfo{};
			pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

			pipeInfo.stageCount = 1; // vertex + fragment shader
			pipeInfo.pStages = stages;

			pipeInfo.pVertexInputState = &inputInfo;
			pipeInfo.pInputAssemblyState = &assemblyInfo;
			pipeInfo.pTessellationState = nullptr;
			pipeInfo.pViewportState = &viewportInfo;
			pipeInfo.pRasterizationState = &rasterInfo;
			pipeInfo.pMultisampleState = &samplingInfo;
			pipeInfo.pDepthStencilState = &depthInfo;
			pipeInfo.pColorBlendState = nullptr;
			pipeInfo.pDynamicState = nullptr;

			pipeInfo.layout = aPipelineLayout;
			pipeInfo.renderPass = aRenderPass;
			pipeInfo.subpass = 0; //first subpass of aRenderPass

			VkPipeline pipe = VK_NULL_HANDLE;
			if (auto const res = vkCreateGraphicsPipelines(aWindow.device,
				VK_NULL_HANDLE, //not using pipeline cache
				1,
				&pipeInfo,
				nullptr,
				&pipe);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create graphics pipeline\n"
					"vkCreateGraphicsPipelines() returned %s",
					lut::to_string(res).c_str());
			}

			return lut::Pipeline(aWindow.device, pipe);
		}

		void record_commands_A(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass,
			VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsLayout,
			VkExtent2D const& aImageExtent,
			VkBuffer aSceneUBO,
			glsl::SceneUniform const& aSceneUniform,
			VkDescriptorSet aSceneDescriptors,
			std::vector<cw2::BakedMeshBuffers>& materialBuffers,
			BakedModel const& aBakedModel
		)
		{
			VkCommandBufferBeginInfo begInfo{};
			begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			begInfo.pInheritanceInfo = nullptr;

			if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to begin recording command buffer\n"
					"vkBeginCommandBuffer() returned %s",
					lut::to_string(res).c_str());
			}

			//Upload scene uniforms
			lut::buffer_barrier(aCmdBuff,
				aSceneUBO,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT);

			vkCmdUpdateBuffer(aCmdBuff,
				aSceneUBO, 0,
				sizeof(glsl::SceneUniform),
				&aSceneUniform);

			lut::buffer_barrier(aCmdBuff,
				aSceneUBO,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);

			//Begin Render pass
			VkClearValue clearValues[1]{};
			clearValues[0].depthStencil.depth = 1.f;

			VkExtent2D extent{};
			extent.width = cfg::shadowMapWidth;
			extent.height = cfg::shadowMapHeight;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPass;
			passInfo.framebuffer = aFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = extent;
			passInfo.clearValueCount = 1;
			passInfo.pClearValues = clearValues;

			//---------BEGIN RENDERPASS------------
			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
			vkCmdBindDescriptorSets(aCmdBuff,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				aGraphicsLayout,
				0,
				1,
				&aSceneDescriptors,
				0,
				nullptr);

			for (std::size_t i = 0; i < materialBuffers.size(); i++)
			{

				VkBuffer buffers[1] = { materialBuffers[i].positions.buffer };
				VkDeviceSize offsets[1]{};

				vkCmdBindVertexBuffers(aCmdBuff, 0, 1, buffers, offsets);
				vkCmdBindIndexBuffer(aCmdBuff, materialBuffers[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

				vkCmdDrawIndexed(aCmdBuff, aBakedModel.meshes[i].indices.size(), 1, 0, 0, 0);

			}

			//End the render pass
			vkCmdEndRenderPass(aCmdBuff);

			//End command recording
			if (auto const res = vkEndCommandBuffer(aCmdBuff);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to end recording command buffer\n"
					"vkEndCommandBuffer() returned %s",
					lut::to_string(res).c_str());
			}
		}

		void record_commands_B(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass,
			VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkPipelineLayout aGraphicsLayout,
			VkExtent2D const& aImageExtent,
			VkBuffer aSceneUBO,
			glsl::SceneUniform const& aSceneUniform,
			VkDescriptorSet aSceneDescriptors,
			VkDescriptorSet aShadowMapDescriptor,
			std::vector<VkDescriptorSet>& aMaterialDescriptors,
			std::vector<cw2::BakedMeshBuffers>& materialBuffers,
			BakedModel const& aBakedModel,
			VkImage shadowMapImage
		)
		{
			VkCommandBufferBeginInfo begInfo{};
			begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			begInfo.pInheritanceInfo = nullptr;

			if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to begin recording command buffer\n"
					"vkBeginCommandBuffer() returned %s",
					lut::to_string(res).c_str());
			}

			//Upload scene uniforms
			lut::buffer_barrier(aCmdBuff,
				aSceneUBO,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT);

			vkCmdUpdateBuffer(aCmdBuff,
				aSceneUBO, 0,
				sizeof(glsl::SceneUniform),
				&aSceneUniform);

			lut::buffer_barrier(aCmdBuff,
				aSceneUBO,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

			//changing image layout for depth buffer
			lut::image_barrier(aCmdBuff,
				shadowMapImage,
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT,
				0, 1,
				0, 1 },
				VK_QUEUE_FAMILY_IGNORED,
				VK_QUEUE_FAMILY_IGNORED
			);

			//Begin Render pass
			VkClearValue clearValues[2]{};
			//Clear to a dark gray background
			clearValues[0].color.float32[0] = 0.1f;
			clearValues[0].color.float32[1] = 0.1f;
			clearValues[0].color.float32[2] = 0.1f;
			clearValues[0].color.float32[3] = 1.f;

			clearValues[1].depthStencil.depth = 1.f;

			VkRenderPassBeginInfo passInfo{};
			passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			passInfo.renderPass = aRenderPass;
			passInfo.framebuffer = aFramebuffer;
			passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
			passInfo.renderArea.extent = aImageExtent;
			passInfo.clearValueCount = 2;
			passInfo.pClearValues = clearValues;

			//---------BEGIN RENDERPASS------------
			vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
			vkCmdBindDescriptorSets(aCmdBuff,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				aGraphicsLayout,
				0,
				1,
				&aSceneDescriptors,
				0,
				nullptr);

			vkCmdBindDescriptorSets(aCmdBuff,
				VK_PIPELINE_BIND_POINT_GRAPHICS,
				aGraphicsLayout,
				2,
				1,
				&aShadowMapDescriptor,
				0,
				nullptr);

			for (std::size_t i = 0; i < materialBuffers.size(); i++)
			{
				
					vkCmdBindDescriptorSets(
						aCmdBuff,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						aGraphicsLayout,
						1,
						1,
						&aMaterialDescriptors[aBakedModel.meshes[i].materialId],
						0,
						nullptr
					);

					VkBuffer buffers[4] = { materialBuffers[i].positions.buffer,
					materialBuffers[i].normals.buffer,
					materialBuffers[i].texcoords.buffer,
					materialBuffers[i].tangents.buffer };
					VkDeviceSize offsets[4]{};

					vkCmdBindVertexBuffers(aCmdBuff, 0, 4, buffers, offsets);
					vkCmdBindIndexBuffer(aCmdBuff, materialBuffers[i].indices.buffer, 0, VK_INDEX_TYPE_UINT32);

					vkCmdDrawIndexed(aCmdBuff, aBakedModel.meshes[i].indices.size(), 1, 0, 0, 0);
				
			}

			//End the render pass
			vkCmdEndRenderPass(aCmdBuff);


			//End command recording
			if (auto const res = vkEndCommandBuffer(aCmdBuff);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to end recording command buffer\n"
					"vkEndCommandBuffer() returned %s",
					lut::to_string(res).c_str());
			}
		}


		int create_material_descriptor_sets(
			lut::VulkanContext const& aContext,
			VkDescriptorPool aPool,
			VkDescriptorSetLayout const& aMaterialLayout,
			std::vector<BakedMaterialInfo> const& aMaterials,
			std::vector<lut::ImageView> const& aImageViews,
			lut::Sampler const& aSampler,
			std::vector<VkDescriptorSet>& descriptorSetVector,
			lut::ImageView const& dummyImageView)
		{
			int index = 0;

			for (BakedMaterialInfo const& material : aMaterials)
			{
				VkDescriptorSet descSet = lut::alloc_desc_set(aContext, aPool,
					aMaterialLayout);
				VkWriteDescriptorSet descW[4]{};
				VkDescriptorImageInfo colorTextureInfo{};
				VkDescriptorImageInfo roughnessTextureInfo{};
				VkDescriptorImageInfo metalTextureInfo{};
				VkDescriptorImageInfo normalMapTextureInfo{};

				// Base color texture
				colorTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				colorTextureInfo.sampler = aSampler.handle;
				colorTextureInfo.imageView = aImageViews[material.baseColorTextureId].handle;
				descW[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descW[0].dstSet = descSet;
				descW[0].dstBinding = 0;
				descW[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descW[0].descriptorCount = 1;
				descW[0].pImageInfo = &colorTextureInfo;

				// Roughness texture
				roughnessTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				roughnessTextureInfo.sampler = aSampler.handle;
				roughnessTextureInfo.imageView = aImageViews[material.roughnessTextureId].handle;
				descW[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descW[1].dstSet = descSet;
				descW[1].dstBinding = 1;
				descW[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descW[1].descriptorCount = 1;
				descW[1].pImageInfo = &roughnessTextureInfo;

				// Metal texture
				metalTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				metalTextureInfo.sampler = aSampler.handle;
				metalTextureInfo.imageView = aImageViews[material.metalnessTextureId].handle;
				descW[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descW[2].dstSet = descSet;
				descW[2].dstBinding = 2;
				descW[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descW[2].descriptorCount = 1;
				descW[2].pImageInfo = &metalTextureInfo;

				// Normal map texture
				normalMapTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				normalMapTextureInfo.sampler = aSampler.handle;
				normalMapTextureInfo.imageView = aImageViews[material.normalMapTextureId].handle;
				descW[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descW[3].dstSet = descSet;
				descW[3].dstBinding = 3;
				descW[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descW[3].descriptorCount = 1;
				descW[3].pImageInfo = &normalMapTextureInfo;
				
				auto numSets = sizeof(descW) / sizeof(descW[0]);
				vkUpdateDescriptorSets(aContext.device, numSets, descW, 0, nullptr);
				descriptorSetVector[index] = descSet;
				
				index++;
			}
			return 1;
		}

		int create_shadow_map_descriptor_set(
			lut::VulkanContext const& aContext,
			VkDescriptorPool aPool,
			VkDescriptorSetLayout const& aShadowLayout,
			lut::ImageView const& aImageView,
			lut::Sampler const& aSampler,
			VkDescriptorSet& descriptorSet)
		{
			
			VkDescriptorSet descSet = lut::alloc_desc_set(aContext, aPool,
				aShadowLayout);
			VkWriteDescriptorSet descW[1]{};
			VkDescriptorImageInfo shadowTextureInfo{};

			// Shadow texture
			shadowTextureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			shadowTextureInfo.sampler = aSampler.handle;
			shadowTextureInfo.imageView = aImageView.handle;
			descW[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descW[0].dstSet = descSet;
			descW[0].dstBinding = 0;
			descW[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descW[0].descriptorCount = 1;
			descW[0].pImageInfo = &shadowTextureInfo;

			auto numSets = sizeof(descW) / sizeof(descW[0]);
			vkUpdateDescriptorSets(aContext.device, numSets, descW, 0, nullptr);
			descriptorSet = descSet;
			return 1;
		}

		lut::Framebuffer create_shadow_framebuffers(lut::VulkanWindow const& aWindow,
			VkRenderPass aRenderPass,
			VkImageView  aDepthView)
		{

			VkImageView attachments[1] = {
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 1;
			fbInfo.pAttachments = attachments;
			fbInfo.width = cfg::shadowMapWidth;
			fbInfo.height = cfg::shadowMapHeight;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for intermediate textures &zu\n"
					"vkCreateFramebuffer() returned %s",
					0,
					lut::to_string(res).c_str());
			}

			return lut::Framebuffer(aWindow.device, fb);
		}

	}
}

namespace
{

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow,
		VkRenderPass aRenderPass,
		std::vector<lut::Framebuffer>& aFramebuffers,
		VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = {
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0;
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb);
				VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image &zu\n"
					"vkCreateFramebuffer() returned %s",
					i,
					lut::to_string(res).c_str());
			}
			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert(aWindow.swapViews.size() == aFramebuffers.size());
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		//throw lut::Error( "Not yet implemented" ); //TODO- (Section 3) implement me!
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0; //number must match the index of the corresponding
		//binding = N declaration in the shader(s)!

		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device,
			&layoutInfo,
			nullptr,
			&layout);
			VK_SUCCESS != res)
		{
			throw lut::Error(" Unable to create descriptor set layout\n"
				"vkCreateDescriptorSetLayout() returned %s",
				lut::to_string(res).c_str());
		}

		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow,
		lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator,
			&imageInfo,
			&allocInfo,
			&image,
			&allocation,
			nullptr);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n"
				"vmaCreateImage() returned %s",
				lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);

		//Create the image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};

		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view);
			VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n"
				"vkCreateImageView() returned %s",
				lut::to_string(res).c_str());
		}

		return { std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}

}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 