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
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {

MeshBlock::MeshBlock(MemoryType memory_type)
    : vertices(memory_type),
      normals(memory_type),
      colors(memory_type),
      semantic_colors(memory_type),
      triangles(memory_type) {}

MeshBlock::MeshBlock(const MeshBlock& mesh_block)
    : MeshBlock(mesh_block, mesh_block.vertices.memory_type()) {}

MeshBlock::MeshBlock(const MeshBlock& mesh_block, MemoryType memory_type)
    : vertices(mesh_block.vertices, memory_type),
      normals(mesh_block.normals, memory_type),
      colors(mesh_block.colors, memory_type),
      semantic_colors(mesh_block.semantic_colors, memory_type),
      triangles(mesh_block.triangles, memory_type) {}

void MeshBlock::clear() {
  vertices.resize(0);
  normals.resize(0);
  triangles.resize(0);
  colors.resize(0);
  semantic_colors.resize(0);
}

void MeshBlock::resizeToNumberOfVertices(size_t new_size) {
  vertices.resize(new_size);
  normals.resize(new_size);
  triangles.resize(new_size);
}

void MeshBlock::reserveNumberOfVertices(size_t new_capacity) {
  vertices.reserve(new_capacity);
  normals.reserve(new_capacity);
  triangles.reserve(new_capacity);
}

MeshBlock::Ptr MeshBlock::allocate(MemoryType memory_type) {
  return std::make_shared<MeshBlock>(memory_type);
}

std::vector<Vector3f> MeshBlock::getVertexVectorOnCPU() const {
  return vertices.toVector();
}

std::vector<Vector3f> MeshBlock::getNormalVectorOnCPU() const {
  return normals.toVector();
}

std::vector<int> MeshBlock::getTriangleVectorOnCPU() const {
  return triangles.toVector();
}

std::vector<Color> MeshBlock::getColorVectorOnCPU() const {
  return colors.toVector();
}

std::vector<Color> MeshBlock::getSemanticColorVectorOnCPU() const {
  return semantic_colors.toVector();
}

size_t MeshBlock::size() const { return vertices.size(); }

size_t MeshBlock::capacity() const { return vertices.capacity(); }

void MeshBlock::expandColorsToMatchVertices() {
  colors.reserve(vertices.capacity());
  colors.resize(vertices.size());
  semantic_colors.reserve(vertices.capacity());
  semantic_colors.resize(vertices.size());
}

// Set the pointers to point to the mesh block.
CudaMeshBlock::CudaMeshBlock(MeshBlock* block) {
  CHECK_NOTNULL(block);
  vertices = block->vertices.data();
  normals = block->normals.data();
  triangles = block->triangles.data();
  colors = block->colors.data();
  semantic_colors = block->semantic_colors.data();

  vertices_size = block->vertices.size();
  triangles_size = block->triangles.size();
}

}  // namespace nvblox