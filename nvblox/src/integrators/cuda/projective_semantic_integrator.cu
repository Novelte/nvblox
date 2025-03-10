/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "nvblox/integrators/projective_semantic_integrator.h"

#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

ProjectiveSemanticIntegrator::ProjectiveSemanticIntegrator()
    : ProjectiveIntegratorBase() {
  sphere_tracer_.maximum_ray_length_m(max_integration_distance_m_);
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
  semantic_color_map_host_.push_back(Color::Blue());
  for (size_t i=1; i < 255; i++)
  {
    semantic_color_map_host_.push_back(Color(rand()%255, rand()%255, rand()%255));
  }
  semantic_color_map_host_.push_back(Color::White());

  semantic_color_map_device_ = semantic_color_map_host_;
}

ProjectiveSemanticIntegrator::~ProjectiveSemanticIntegrator() {
  finish();
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveSemanticIntegrator::finish() const {
  cudaStreamSynchronize(integration_stream_);
}

void ProjectiveSemanticIntegrator::integrateFrame(
    const SemanticImage& semantic_frame, const Transform& T_L_C, const Camera& camera,
    const TsdfLayer& tsdf_layer, SemanticLayer* semantic_layer,
    std::vector<Index3D>* updated_blocks) {
  timing::Timer semantic_timer("semantic/integrate");
  CHECK_NOTNULL(semantic_layer);
  CHECK_EQ(tsdf_layer.block_size(), semantic_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      semantic_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  timing::Timer blocks_in_view_timer("semantic/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices = view_calculator_.getBlocksInViewPlanes(
      T_L_C, camera, semantic_layer->block_size(),
      max_integration_distance_m_ + truncation_distance_m);
  blocks_in_view_timer.Stop();

  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  timing::Timer blocks_in_band_timer(
      "semantic/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  blocks_in_band_timer.Stop();

  // Allocate blocks (CPU)
  // We allocate semantic blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer("semantic/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, semantic_layer);
  allocate_blocks_timer.Stop();

  // Create a synthetic depth image
  timing::Timer sphere_trace_timer("semantic/integrate/sphere_trace");
  std::shared_ptr<const DepthImage> synthetic_depth_image_ptr =
      sphere_tracer_.renderImageOnGPU(
          camera, T_L_C, tsdf_layer, truncation_distance_m, MemoryType::kDevice,
          sphere_tracing_ray_subsampling_factor_);
  sphere_trace_timer.Stop();

  // Update identified blocks
  // Calls out to the child-class implementing the integation (GPU)
  timing::Timer update_blocks_timer("semantic/integrate/update_blocks");
  updateBlocks(block_indices, semantic_frame, *synthetic_depth_image_ptr, T_L_C,
               camera, truncation_distance_m, semantic_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

// Integrate Frame with Color
void ProjectiveSemanticIntegrator::integrateFrame(
    const SemanticImage& semantic_frame, const Transform& T_L_C, const Camera& camera,
    const TsdfLayer& tsdf_layer, SemanticLayer* semantic_layer, ColorLayer* color_layer,
    std::vector<Index3D>* updated_blocks) {
  timing::Timer semantic_timer("semantic/integrate");
  CHECK_NOTNULL(semantic_layer);
  CHECK_NOTNULL(color_layer);
  CHECK_EQ(tsdf_layer.block_size(), semantic_layer->block_size());
  CHECK_EQ(tsdf_layer.block_size(), color_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      semantic_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  timing::Timer blocks_in_view_timer("semantic/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices = view_calculator_.getBlocksInViewPlanes(
      T_L_C, camera, semantic_layer->block_size(),
      max_integration_distance_m_ + truncation_distance_m);
  blocks_in_view_timer.Stop();

  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  timing::Timer blocks_in_band_timer(
      "semantic/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  blocks_in_band_timer.Stop();

  // Allocate blocks (CPU)
  // We allocate semantic blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer("semantic/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, semantic_layer);
  allocateBlocksWhereRequired(block_indices, color_layer);
  allocate_blocks_timer.Stop();

  // Create a synthetic depth image
  timing::Timer sphere_trace_timer("semantic/integrate/sphere_trace");
  std::shared_ptr<const DepthImage> synthetic_depth_image_ptr =
      sphere_tracer_.renderImageOnGPU(
          camera, T_L_C, tsdf_layer, truncation_distance_m, MemoryType::kDevice,
          sphere_tracing_ray_subsampling_factor_);
  sphere_trace_timer.Stop();

  // Update identified blocks
  // Calls out to the child-class implementing the integation (GPU)
  timing::Timer update_blocks_timer("semantic/integrate/update_blocks");
  updateBlocks(block_indices, semantic_frame, *synthetic_depth_image_ptr, T_L_C,
               camera, truncation_distance_m, semantic_layer, color_layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

void ProjectiveSemanticIntegrator::sphere_tracing_ray_subsampling_factor(
    int sphere_tracing_ray_subsampling_factor) {
  CHECK_GT(sphere_tracing_ray_subsampling_factor, 0);
  sphere_tracing_ray_subsampling_factor_ =
      sphere_tracing_ray_subsampling_factor;
}

int ProjectiveSemanticIntegrator::sphere_tracing_ray_subsampling_factor() const {
  return sphere_tracing_ray_subsampling_factor_;
}

__device__ inline bool updateVoxel(const Semantic semantic_measured,
                                            SemanticVoxel* voxel_ptr,
                                            const float voxel_depth_m,
                                            const float truncation_distance_m,
                                            const float max_weight) {
  // NOTE(alexmillane): We integrate all voxels passed to this function, We
  // should probably not do this. We should no update some based on occlusion
  // and their distance in the distance field....
  // TODO(alexmillane): The above.

  // Read CURRENT voxel values (from global GPU memory)
  const Semantic voxel_semantic_current = voxel_ptr->id;
  const float voxel_weight_current = voxel_ptr->weight;
  // Fuse
  constexpr float measurement_weight = 1.0f;
  Semantic fused_semantic;
  float weight = 0.0f;
  // If the same semantic id, update the confidence to the average
  if(voxel_semantic_current.id == semantic_measured.id)
  {
    fused_semantic = semantic_measured;
    weight = fmin(measurement_weight + voxel_weight_current, max_weight);
  }
  // If semantic id is different, keep the larger one and drop a little for the disagreement
  else
  {
    fused_semantic = voxel_weight_current > measurement_weight ? voxel_semantic_current : semantic_measured;
    weight = fmin(0.5*voxel_weight_current, max_weight);
  }

  // Write NEW voxel values (to global GPU memory)
  voxel_ptr->id = fused_semantic;
  voxel_ptr->weight = weight;
  return true;
}

__device__ inline bool updateVoxel(const Semantic semantic_measured,
                                            SemanticVoxel* voxel_ptr,
                                            ColorVoxel* color_voxel_ptr,
                                            const float voxel_depth_m,
                                            const float truncation_distance_m,
                                            const float max_weight,
                                            const Color* color_map) {
  // NOTE(alexmillane): We integrate all voxels passed to this function, We
  // should probably not do this. We should no update some based on occlusion
  // and their distance in the distance field....
  // TODO(alexmillane): The above.

  // Read CURRENT voxel values (from global GPU memory)
  const Semantic voxel_semantic_current = voxel_ptr->id;
  const float voxel_weight_current = voxel_ptr->weight;
  // Fuse
  constexpr float measurement_weight = 1.0f;
  Semantic fused_semantic;
  float weight = 0.0f;
  // If the same semantic id, update the confidence to the average
  if(voxel_semantic_current.id == semantic_measured.id)
  {
    fused_semantic = semantic_measured;
    weight = fmin(measurement_weight + voxel_weight_current, max_weight);
  }
  // If semantic id is different, keep the larger one and drop a little for the disagreement
  else
  {
    fused_semantic = voxel_weight_current > measurement_weight ? voxel_semantic_current : semantic_measured;
    weight = fmin(0.5*voxel_weight_current, max_weight);
  }

  // Write NEW voxel values (to global GPU memory)
  voxel_ptr->id = fused_semantic;
  voxel_ptr->weight = weight;
  color_voxel_ptr->semantic_color = color_map[fused_semantic.id];
  return true;
}

__global__ void integrateBlocks(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Semantic* semantic_image, const int semantic_rows, const int semantic_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float truncation_distance_m, const float max_weight,
    const float max_integration_distance, const int depth_subsample_factor,
    SemanticBlock** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel<Camera>(block_indices_device_ptr, camera, T_C_L,
                                  block_size, &u_px, &voxel_depth_m,
                                  &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!interpolation::interpolate2DClosest<float>(
          depth_image, u_px_depth, depth_rows, depth_cols, &surface_depth_m)) {
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > truncation_distance_m) {
    return;
  }

  Semantic image_value;
  if (!interpolation::interpolate2DClosest<Semantic>(semantic_image, u_px, semantic_rows,
                                                 semantic_cols, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  SemanticVoxel* voxel_ptr =
      &(block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  updateVoxel(image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
              max_weight);
}

__global__ void integrateBlocks(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const Semantic* semantic_image, const int semantic_rows, const int semantic_cols,
    const float* depth_image, const int depth_rows, const int depth_cols,
    const Transform T_C_L, const float block_size,
    const float truncation_distance_m, const float max_weight,
    const float max_integration_distance, const int depth_subsample_factor,
    SemanticBlock** semantic_block_device_ptrs, ColorBlock** color_block_device_ptrs,
    const Color* color_map) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel<Camera>(block_indices_device_ptr, camera, T_C_L,
                                  block_size, &u_px, &voxel_depth_m,
                                  &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  const Eigen::Vector2f u_px_depth =
      u_px / static_cast<float>(depth_subsample_factor);
  float surface_depth_m;
  if (!interpolation::interpolate2DClosest<float>(
          depth_image, u_px_depth, depth_rows, depth_cols, &surface_depth_m)) {
    return;
  }

  // Occlusion testing
  // Get the distance of the voxel from the rendered surface. If outside
  // truncation band, skip.
  const float voxel_distance_from_surface = surface_depth_m - voxel_depth_m;
  if (fabsf(voxel_distance_from_surface) > truncation_distance_m) {
    return;
  }

  Semantic image_value;
  if (!interpolation::interpolate2DClosest<Semantic>(semantic_image, u_px, semantic_rows,
                                                 semantic_cols, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  SemanticVoxel* semantic_voxel_ptr =
      &(semantic_block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  ColorVoxel* color_voxel_ptr =
      &(color_block_device_ptrs[blockIdx.x]
            ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  updateVoxel(image_value, semantic_voxel_ptr, color_voxel_ptr, voxel_depth_m, 
              truncation_distance_m, max_weight, color_map);
}

void ProjectiveSemanticIntegrator::updateBlocks(
    const std::vector<Index3D>& block_indices, const SemanticImage& semantic_frame,
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, SemanticLayer* semantic_layer_ptr) {
  CHECK_NOTNULL(semantic_layer_ptr);
  CHECK_EQ(semantic_frame.rows() % depth_frame.rows(), 0);
  CHECK_EQ(semantic_frame.cols() % depth_frame.cols(), 0);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();
  const int depth_subsampling_factor = semantic_frame.rows() / depth_frame.rows();
  CHECK_EQ(semantic_frame.cols() / depth_frame.cols(), depth_subsampling_factor);

  // Expand the buffers when needed
  if (num_blocks > block_indices_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    block_indices_device_.reserve(new_size);
    semantic_block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    semantic_block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  semantic_block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, semantic_layer_ptr);

  // Transfer to the device
  block_indices_device_ = block_indices_host_;
  semantic_block_ptrs_device_ = semantic_block_ptrs_host_;

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices.size();
  // clang-format off
  integrateBlocks<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      block_indices_device_.data(),
      camera,
      semantic_frame.dataConstPtr(),
      semantic_frame.rows(),
      semantic_frame.cols(),
      depth_frame.dataConstPtr(),
      depth_frame.rows(),
      depth_frame.cols(),
      T_C_L,
      semantic_layer_ptr->block_size(),
      truncation_distance_m,
      max_weight_,
      max_integration_distance_m_,
      depth_subsampling_factor,
      semantic_block_ptrs_device_.data());
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Finish processing of the frame before returning control
  finish();
}

// Update Blocks with Color Layer
void ProjectiveSemanticIntegrator::updateBlocks(
    const std::vector<Index3D>& block_indices, const SemanticImage& semantic_frame,
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, SemanticLayer* semantic_layer_ptr, 
    ColorLayer* color_layer_ptr) {
  CHECK_NOTNULL(semantic_layer_ptr);
  CHECK_NOTNULL(color_layer_ptr);
  CHECK_EQ(semantic_frame.rows() % depth_frame.rows(), 0);
  CHECK_EQ(semantic_frame.cols() % depth_frame.cols(), 0);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();
  const int depth_subsampling_factor = semantic_frame.rows() / depth_frame.rows();
  CHECK_EQ(semantic_frame.cols() / depth_frame.cols(), depth_subsampling_factor);

  // Expand the buffers when needed
  if (num_blocks > block_indices_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    block_indices_device_.reserve(new_size);
    semantic_block_ptrs_device_.reserve(new_size);
    color_block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    semantic_block_ptrs_host_.reserve(new_size);
    color_block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  semantic_block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, semantic_layer_ptr);
  color_block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, color_layer_ptr);

  // Transfer to the device
  block_indices_device_ = block_indices_host_;
  semantic_block_ptrs_device_ = semantic_block_ptrs_host_;
  color_block_ptrs_device_ = color_block_ptrs_host_;


  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices.size();
  // clang-format off
  integrateBlocks<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      block_indices_device_.data(),
      camera,
      semantic_frame.dataConstPtr(),
      semantic_frame.rows(),
      semantic_frame.cols(),
      depth_frame.dataConstPtr(),
      depth_frame.rows(),
      depth_frame.cols(),
      T_C_L,
      semantic_layer_ptr->block_size(),
      truncation_distance_m,
      max_weight_,
      max_integration_distance_m_,
      depth_subsampling_factor,
      semantic_block_ptrs_device_.data(),
      color_block_ptrs_device_.data(),
      semantic_color_map_device_.data());
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Finish processing of the frame before returning control
  finish();
}

extern __global__ void checkBlocksInTruncationBand(
    const VoxelBlock<TsdfVoxel>** block_device_ptrs,
    const float truncation_distance_m,
    bool* contains_truncation_band_device_ptr);

std::vector<Index3D>
ProjectiveSemanticIntegrator::reduceBlocksToThoseInTruncationBand(
    const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m) {
  // Check 1) Are the blocks allocated
  // - performed on the CPU because the hash-map is on the CPU
  std::vector<Index3D> block_indices_check_1;
  block_indices_check_1.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (tsdf_layer.isBlockAllocated(block_idx)) {
      block_indices_check_1.push_back(block_idx);
    }
  }

  if (block_indices_check_1.empty()) {
    return block_indices_check_1;
  }

  // Check 2) Does each of the blocks have a voxel within the truncation band
  // - performed on the GPU because the blocks are there
  // Get the blocks we need to check
  std::vector<const TsdfBlock*> block_ptrs =
      getBlockPtrsFromIndices(block_indices_check_1, tsdf_layer);

  const int num_blocks = block_ptrs.size();

  // Expand the buffers when needed
  if (num_blocks > truncation_band_block_ptrs_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    truncation_band_block_ptrs_host_.reserve(new_size);
    truncation_band_block_ptrs_device_.reserve(new_size);
    block_in_truncation_band_device_.reserve(new_size);
    block_in_truncation_band_host_.reserve(new_size);
  }

  // Host -> Device
  truncation_band_block_ptrs_host_ = block_ptrs;
  truncation_band_block_ptrs_device_ = truncation_band_block_ptrs_host_;

  // Prepare output space
  block_in_truncation_band_device_.resize(num_blocks);

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  checkBlocksInTruncationBand<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      truncation_band_block_ptrs_device_.data(),
      truncation_distance_m,
      block_in_truncation_band_device_.data());
  // clang-format on
  checkCudaErrors(cudaStreamSynchronize(integration_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back
  block_in_truncation_band_host_ = block_in_truncation_band_device_;

  // Filter the indices using the result
  std::vector<Index3D> block_indices_check_2;
  block_indices_check_2.reserve(block_indices_check_1.size());
  for (int i = 0; i < block_indices_check_1.size(); i++) {
    if (block_in_truncation_band_host_[i] == true) {
      block_indices_check_2.push_back(block_indices_check_1[i]);
    }
  }

  return block_indices_check_2;
}

}  // namespace nvblox
