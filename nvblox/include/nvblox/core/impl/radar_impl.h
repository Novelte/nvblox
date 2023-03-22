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

#include "math.h"

#include <glog/logging.h>

namespace nvblox {

Radar::Radar(int num_azimuth_divisions, int num_elevation_divisions,
             float horizontal_fov_rad, float vertical_fov_rad)
    : num_azimuth_divisions_(num_azimuth_divisions),
      num_elevation_divisions_(num_elevation_divisions),
      horizontal_fov_rad_(horizontal_fov_rad),
      vertical_fov_rad_(vertical_fov_rad) {
  // Even numbers of beams allowed
  CHECK(num_azimuth_divisions_ % 2 == 0);

  // Angular distance between pixels
  // Note(alexmillane): Note the difference in division by N vs. (N-1) below.
  // This is because in the azimuth direction there's a wrapping around. The
  // point at pi/-pi is not double sampled, generating this difference.
  rads_per_pixel_azimuth_ =
      horizontal_fov_rad_ / static_cast<float>(num_azimuth_divisions_ - 1);
  rads_per_pixel_elevation_ =
      vertical_fov_rad_ / static_cast<float>(num_elevation_divisions_ - 1);

  // printf("rads_per_pixel_elevation_   %f\n", rads_per_pixel_elevation_);
  // printf("rads_per_pixel_azimuth_     %f\n", rads_per_pixel_azimuth_);

  // Inverse of the above
  elevation_pixels_per_rad_ = 1.0f / rads_per_pixel_elevation_;
  azimuth_pixels_per_rad_ = 1.0f / rads_per_pixel_azimuth_;

  // The angular lower-extremes of the image-plane
  // NOTE(alexmillane): Because beams pass through the angular extremes of the
  // FoV, the corresponding lower extreme pixels start half a pixel width
  // below this.
  // Note(alexmillane): Note that we use polar angle here, not elevation.
  // Polar is from the top of the sphere down, elevation, the middle up.
  start_polar_angle_rad_ = M_PI / 2.0f - (vertical_fov_rad / 2.0f +
                                          rads_per_pixel_elevation_ / 2.0f);

  start_azimuth_angle_rad_ = M_PI / 2.0f - (horizontal_fov_rad / 2.0f +
                                          rads_per_pixel_azimuth_ / 2.0f);
  
  // printf("start_azimuth_angle      %f\n", start_azimuth_angle_rad_);
  // printf("num_azimuth_divisions_   %d\n", num_azimuth_divisions_);
  // printf("horizontal_fov_rad       %f\n\n", horizontal_fov_rad);

  // printf("start_polar_angle        %f\n", start_polar_angle_rad_);
  // printf("num_elevation_divisions_ %d\n", num_elevation_divisions_);
  // printf("vertical_fov_rad_        %f\n\n", vertical_fov_rad_);
}

int Radar::num_azimuth_divisions() const { return num_azimuth_divisions_; }

int Radar::num_elevation_divisions() const { return num_elevation_divisions_; }

float Radar::vertical_fov_rad() const { return vertical_fov_rad_; }

float Radar::horizontal_fov_rad() const { return horizontal_fov_rad_; }

int Radar::numel() const {
  return num_azimuth_divisions_ * num_elevation_divisions_;
}

int Radar::cols() const { return num_azimuth_divisions_; }

int Radar::rows() const { return num_elevation_divisions_; }

bool Radar::project(const Vector3f& p_C, Vector2f* u_C) const {
  // To spherical coordinates
  const float r = p_C.norm();
  constexpr float kMinProjectionEps = 0.01;
  if (r < kMinProjectionEps) {
    return false;
  }
  const float polar_angle_rad = acos(p_C.z() / r);
  const float azimuth_angle_rad = atan2(p_C.x(), p_C.y());

  // To image plane coordinates
  float u_float =
      (azimuth_angle_rad - start_azimuth_angle_rad_) * azimuth_pixels_per_rad_;
  float v_float =
      (polar_angle_rad - start_polar_angle_rad_) * elevation_pixels_per_rad_;

  // Points out of FOV
  if (u_float < 0.0f || u_float >= (float)num_azimuth_divisions_) {
    // printf("failing u_float \n");    
    // printf("x %f     y: %f\n", p_C.x(), p_C.y());
    // printf("azimuth_angle_rad       %f \n", azimuth_angle_rad);
    // printf("ufloat                  %f \n", u_float);
    return false;
  }
  // Points out of FOV
  // NOTE(alexmillane): It should be impossible to escape the -pi-to-pi range in
  // azimuth due to wrap around this. Therefore we don't check.
  if (v_float < 0.0f || v_float >= (float)num_elevation_divisions_) {
    // printf("failing v_float \n");
    // printf("polar_angle_rad         %f \n", polar_angle_rad);
    // printf("vfloat                  %f \n", v_float);
    return false;
  }

  // Write output
  *u_C = Vector2f(u_float, v_float);
  return true;
}

bool Radar::project(const Vector3f& p_C, Index2D* u_C) const {
  Vector2f u_C_float;
  bool res = project(p_C, &u_C_float);
  *u_C = u_C_float.array().floor().matrix().cast<int>();
  return res;
}

float Radar::getDepth(const Vector3f& p_C) const { return p_C.norm(); }

Vector2f Radar::pixelIndexToImagePlaneCoordsOfCenter(const Index2D& u_C) const {
  // The index cast to a float is the coordinates of the lower corner of the
  // pixel.
  return u_C.cast<float>() + Vector2f(0.5f, 0.5f);
}

Index2D Radar::imagePlaneCoordsToPixelIndex(const Vector2f& u_C) const {
  // NOTE(alexmillane): We do floor rather than a straight truncation such that
  // we handle negative image plane coordinates.
  return u_C.array().floor().cast<int>();
}

Vector3f Radar::unprojectFromImagePlaneCoordinates(const Vector2f& u_C,
                                                   const float depth) const {
  return depth * vectorFromImagePlaneCoordinates(u_C);
}

Vector3f Radar::unprojectFromPixelIndices(const Index2D& u_C,
                                          const float depth) const {
  return depth * vectorFromPixelIndices(u_C);
}

Vector3f Radar::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  // NOTE(alexmillane): We don't do any bounds checking, i.e. that the point is
  // actually on the image plane.
  const float polar_angle_rad =
      u_C.y() * rads_per_pixel_elevation_ + start_polar_angle_rad_;
  const float azimuth_angle_rad =
      u_C.x() * rads_per_pixel_azimuth_ + start_azimuth_angle_rad_;
  return Vector3f(sin(azimuth_angle_rad) * sin(polar_angle_rad),
                  cos(azimuth_angle_rad) * sin(polar_angle_rad),
                  cos(polar_angle_rad));
}

Vector3f Radar::vectorFromPixelIndices(const Index2D& u_C) const {
  return vectorFromImagePlaneCoordinates(
      pixelIndexToImagePlaneCoordsOfCenter(u_C));
}

AxisAlignedBoundingBox Radar::getViewAABB(const Transform& T_L_C,
                                          const float min_depth,
                                          const float max_depth) const {
  // The AABB is a square centered at the radars location where the height is
  // determined by the radar FoV.
  // NOTE(alexmillane): The min depth is ignored in this function, it is a
  // parameter so it matches with camera's getViewAABB()

  // Remark, how does this fit in Radar Case should get the corner
  AxisAlignedBoundingBox box(
      Vector3f(0, -max_depth * sin(horizontal_fov_rad_ / 2.0f),
               -max_depth * sin(vertical_fov_rad_ / 2.0f)),
      Vector3f(max_depth, max_depth * sin(horizontal_fov_rad_ / 2.0f),
               max_depth * sin(vertical_fov_rad_ / 2.0f)));

  // Translate the box to the sensor's location (note that orientation doesn't
  // matter as the radar sees in the circle)
  box.translate(T_L_C.translation());
  return box;
}

size_t Radar::Hash::operator()(const Radar& radar) const {
  // Taken from:
  // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
  size_t az_hash = std::hash<int>()(radar.num_azimuth_divisions_);
  size_t el_hash = std::hash<int>()(radar.num_elevation_divisions_);
  size_t h_fov_hash = std::hash<float>()(radar.horizontal_fov_rad_);
  size_t v_fov_hash = std::hash<float>()(radar.vertical_fov_rad_);
  return ((az_hash ^ (el_hash << 1)) >> 1) ^ (h_fov_hash << 1) ^ (v_fov_hash << 1);
}

bool operator==(const Radar& lhs, const Radar& rhs) {
  return (lhs.num_azimuth_divisions_ == rhs.num_azimuth_divisions_) &&
         (lhs.num_elevation_divisions_ == rhs.num_elevation_divisions_) &&
         (std::fabs(lhs.horizontal_fov_rad_ - rhs.horizontal_fov_rad_) <
          std::numeric_limits<float>::epsilon()) &&
         (std::fabs(lhs.vertical_fov_rad_ - rhs.vertical_fov_rad_) <
          std::numeric_limits<float>::epsilon());
}

}  // namespace nvblox
