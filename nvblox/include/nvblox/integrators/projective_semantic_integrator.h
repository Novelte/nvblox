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
#pragma once

#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"
#include "nvblox/integrators/projective_integrator_base.h"
#include "nvblox/rays/sphere_tracer.h"

namespace nvblox {

/// A class performing semantic intregration
///
/// Integrates a depth images into semantic layers. The "projective" is a describes
/// one type of integration. Namely that voxels in view are projected into the
/// depth image (the alternative being casting rays out from the camera).
class ProjectiveSemanticIntegrator : public ProjectiveIntegratorBase {
 public:
  ProjectiveSemanticIntegrator();
  virtual ~ProjectiveSemanticIntegrator();

  /// Blocks until GPU operations are complete
  /// Ensure outstanding operations are finished (relevant for integrators
  /// launching asynchronous work)
  void finish() const override;

  /// Integrates a semantic image into the passed semantic layer.
  /// @param semantic_frame A semantic image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param tsdf_layer The TSDF layer with which the semantic layer associated.
  /// Semantic integration is only performed on the voxels corresponding to the
  /// truncation band of this layer.
  /// @param semantic_layer A pointer to the layer into which this image will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const SemanticImage& semantic_frame, const Transform& T_L_C,
                      const Camera& camera, const TsdfLayer& tsdf_layer,
                      SemanticLayer* semantic_layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Integrates a semantic image into the passed semantic layer.
  /// @param semantic_frame A semantic image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param tsdf_layer The TSDF layer with which the semantic layer associated.
  /// Semantic integration is only performed on the voxels corresponding to the
  /// truncation band of this layer.
  /// @param semantic_layer A pointer to the semantic layer into which this 
  /// image will be intergrated.
  /// @param color_layer A pointer to the color layer into which this image will
  /// be intergrated.
  /// @param color_map A pointer to the color map.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const SemanticImage& semantic_frame, const Transform& T_L_C,
                      const Camera& camera, const TsdfLayer& tsdf_layer,
                      SemanticLayer* semantic_layer, ColorLayer* color_layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Returns the sphere tracer used for semantic integration.
  /// In order to perform semantic integration from an rgb image we have to
  /// determine which surfaces are in view. We use sphere tracing for this
  /// purpose.
  const SphereTracer& sphere_tracer() const { return sphere_tracer_; }
  /// Returns the object used to calculate the blocks in camera views.
  /// In order to perform semantic integration from an rgb image we have to
  /// determine which surfaces are in view. We use sphere tracing for this
  /// purpose.
  SphereTracer& sphere_tracer() { return sphere_tracer_; }

  /// A parameter getter
  /// We find a surface on which to integrate semantic by sphere tracing from the
  /// camera. This is a relatively expensive operation. This parameter controls
  /// how many rays are traced for an image. For example, for a 100px by 100px
  /// image with a subsampling factor of 4, 25x25 rays are traced.
  /// @returns the ray subsampling factor
  int sphere_tracing_ray_subsampling_factor() const;

  /// A parameter setter
  /// See sphere_tracing_ray_subsampling_factor()
  /// @param sphere_tracing_ray_subsampling_factor the ray subsampling factor
  void sphere_tracing_ray_subsampling_factor(
      int sphere_tracing_ray_subsampling_factor);

 protected:
  // Given a set of blocks in view (block_indices) perform semantic updates on all
  // voxels within these blocks on the GPU.
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const SemanticImage& semantic_frame,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    SemanticLayer* semantic_layer_ptr);

  // Given a set of blocks in view (block_indices) perform semantic updates on all
  // voxels within these blocks on the GPU.
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const SemanticImage& semantic_frame,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    SemanticLayer* semantic_layer_ptr, ColorLayer* color_layer_ptr);

  // Takes a list of block indices and returns a subset containing the block
  // indices containing at least on voxel inside the truncation band of the
  // passed TSDF layer.
  std::vector<Index3D> reduceBlocksToThoseInTruncationBand(
      const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

  // Params
  int sphere_tracing_ray_subsampling_factor_ = 4;

  // Object to do ray tracing to generate occlusions
  SphereTracer sphere_tracer_;

  // Blocks to integrate on the current call and their indices
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<SemanticBlock*> semantic_block_ptrs_device_;
  device_vector<ColorBlock*> color_block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<SemanticBlock*> semantic_block_ptrs_host_;
  host_vector<ColorBlock*> color_block_ptrs_host_;
  // Color Maps
  device_vector<Color> semantic_color_map_device_;
  host_vector<Color> semantic_color_map_host_;
  


  // Buffers for getting blocks in truncation band
  device_vector<const TsdfBlock*> truncation_band_block_ptrs_device_;
  host_vector<const TsdfBlock*> truncation_band_block_ptrs_host_;
  device_vector<bool> block_in_truncation_band_device_;
  host_vector<bool> block_in_truncation_band_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox