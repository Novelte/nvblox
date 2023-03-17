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

#include <cuda_runtime.h>
#include <stdint.h>
#include <cmath>

namespace nvblox {

struct Semantic {
  __host__ __device__ Semantic() : id(0) {}
  __host__ __device__ Semantic( uint8_t _id)
      : id(_id) {}

  uint8_t id;

  bool operator==(const Semantic& other) const {
    return (id == other.id);
  }

  // Default class for unknown object
  static const Semantic Unknown() { return Semantic(255); }
};

}  // namespace nvblox
