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
#include <gtest/gtest.h>

#include "nvblox/core/radar.h"
#include "nvblox/core/types.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-3;
class RadarTest
    : public ::testing::Test {
 protected:
  RadarTest() {}
};

class ParameterizedRadarTest
    : public RadarTest,
      public ::testing::WithParamInterface<std::tuple<int, int, float, float>> {
 protected:
  // Yo dawg I heard you like params
};

TEST_P(ParameterizedRadarTest, Extremes) {
  // Radar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float horizontal_fov_deg = std::get<2>(params);
  const float vertical_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;

  Radar radar(num_azimuth_divisions, num_elevation_divisions, horizontal_fov_rad, vertical_fov_rad);

  //-------------------
  // Elevation extremes
  //-------------------
  float azimuth_center_pixel =
      static_cast<float>(num_azimuth_divisions) / 2.0f;
  float elevation_center_pixel = 
      static_cast<float>(num_elevation_divisions) / 2.0f;

  float elevation_top_pixel = 0.5f;
  float elevation_bottom_pixel =
      static_cast<float>(num_elevation_divisions) - 0.5f;

  const float x_dist = 10;
  const float z_dist = x_dist * tan(vertical_fov_rad / 2.0f);
  const float y_dist = x_dist * tan(horizontal_fov_rad / 2.0f);

  // Top beam
  Vector2f u_C;
  Vector3f p = Vector3f(x_dist, 0.0, z_dist);
  EXPECT_TRUE(radar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_top_pixel, kFloatEpsilon);

  // Center
  p = Vector3f(x_dist, 0.0f, 0.0f);
  EXPECT_TRUE(radar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Bottom beam
  p = Vector3f(x_dist, 0.0f, -z_dist);
  EXPECT_TRUE(radar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_bottom_pixel, kFloatEpsilon);

  //-----------------
  // Azimuth extremes
  //-----------------

  float azimuth_left_pixel = 0.5;
  float azimuth_right_pixel = (float)num_azimuth_divisions - 0.5;

  // Right
  p = Vector3f(x_dist, -y_dist, 0.0);
  EXPECT_TRUE(radar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_right_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Forwards
  p = Vector3f(x_dist, 0.0, 0.0);
  EXPECT_TRUE(radar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Left
  p = Vector3f(x_dist, y_dist, 0.0);
  EXPECT_TRUE(radar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_left_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);
}

TEST_P(ParameterizedRadarTest, SphereTest) {
  // Radar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float horizontal_fov_deg = std::get<2>(params);
  const float vertical_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;

  Radar radar(num_azimuth_divisions, num_elevation_divisions, horizontal_fov_rad, vertical_fov_rad);

  // Pointcloud
  Eigen::MatrixX3f pointcloud(num_azimuth_divisions * num_elevation_divisions,
                              3);
  Eigen::MatrixXf desired_image(num_elevation_divisions, num_azimuth_divisions);

  // Construct a pointcloud of a points at random distances.
  const float azimuth_increments_rad = 
      horizontal_fov_rad / ((float)num_azimuth_divisions - 1);
  const float polar_increments_rad =
      vertical_fov_rad / ((float)num_elevation_divisions - 1);

  const float start_azimuth_angle_rad = M_PI / 2.0f - (horizontal_fov_rad / 2.0f);
  const float start_polar_angle_rad = M_PI / 2.0f - (vertical_fov_rad / 2.0f);

  int point_idx = 0;
  for (int az_idx = 0; az_idx < num_azimuth_divisions; az_idx++) {
    for (int el_idx = 0; el_idx < num_elevation_divisions; el_idx++) {
      const float azimuth_rad = 
          az_idx * azimuth_increments_rad + start_azimuth_angle_rad;
      const float polar_rad =
          el_idx * polar_increments_rad + start_polar_angle_rad;

      constexpr float max_depth = 10.0;
      constexpr float min_depth = 1.0;
      const float distance =
          test_utils::randomFloatInRange(min_depth, max_depth);

      const float x = distance * sin(polar_rad) * sin(azimuth_rad);
      const float y = distance * sin(polar_rad) * cos(azimuth_rad);
      const float z = distance * cos(polar_rad);

      pointcloud(point_idx, 0) = x;
      pointcloud(point_idx, 1) = y;
      pointcloud(point_idx, 2) = z;

      desired_image(el_idx, az_idx) = distance;

      point_idx++;
    }
  }

  // Project the pointcloud to a depth image
  Eigen::MatrixXf reprojected_image(num_elevation_divisions,
                                    num_azimuth_divisions);
  for (int point_idx = 0; point_idx < pointcloud.rows(); point_idx++) {
    // Projection
    Vector2f u_C_float;
    EXPECT_TRUE(radar.project(pointcloud.row(point_idx), &u_C_float));
    Index2D u_C_int;
    EXPECT_TRUE(radar.project(pointcloud.row(point_idx), &u_C_int));

    // Check that this is at the center of a pixel
    Vector2f corner_dist = u_C_float - u_C_float.array().floor().matrix();
    constexpr float kReprojectionEpsilon = 0.001;
    EXPECT_NEAR(corner_dist.x(), 0.5f, kReprojectionEpsilon);
    EXPECT_NEAR(corner_dist.y(), 0.5f, kReprojectionEpsilon);

    // Add to depth image
    reprojected_image(u_C_int.y(), u_C_int.x()) =
        pointcloud.row(point_idx).norm();
  }

  const Eigen::MatrixXf error_image =
      desired_image.array() - reprojected_image.array();
  float max_error = error_image.maxCoeff();
  EXPECT_NEAR(max_error, 0.0, kFloatEpsilon);
}

TEST_P(ParameterizedRadarTest, OutOfBoundsTest) {
  // Radar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float horizontal_fov_deg = std::get<2>(params);
  const float vertical_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;

  Radar radar(num_azimuth_divisions, num_elevation_divisions, horizontal_fov_rad, vertical_fov_rad);

  Vector2f u_C_float;
  Index2D u_C_int;

  // Outside on top and bottom
  const float rads_per_pixel_elevation =
      vertical_fov_rad / static_cast<float>(num_elevation_divisions - 1);
  const float x_dist = 10.0f;
  const float z_dist = x_dist * tan(vertical_fov_rad / 2.0f +
                                    rads_per_pixel_elevation / 2.0f + 0.001);

  EXPECT_FALSE(radar.project(Vector3f(x_dist, 0.0f, z_dist), &u_C_float));
  EXPECT_FALSE(radar.project(Vector3f(x_dist, 0.0f, -z_dist), &u_C_float));
  EXPECT_FALSE(radar.project(Vector3f(x_dist, 0.0f, z_dist), &u_C_int));
  EXPECT_FALSE(radar.project(Vector3f(x_dist, 0.0f, -z_dist), &u_C_int));

  // Outside on left and right
  const float rads_per_pixel_azimuth =
      horizontal_fov_rad / static_cast<float>(num_azimuth_divisions - 1);
  const float y_dist = x_dist * tan(horizontal_fov_rad / 2.0f +
                                    rads_per_pixel_azimuth / 2.0f + 0.001);

  EXPECT_FALSE(radar.project(Vector3f(x_dist, y_dist, 0.0f), &u_C_float));
  EXPECT_FALSE(radar.project(Vector3f(x_dist, -y_dist, 0.0f), &u_C_float));
  EXPECT_FALSE(radar.project(Vector3f(x_dist, y_dist, 0.0f), &u_C_int));
  EXPECT_FALSE(radar.project(Vector3f(x_dist, -y_dist, 0.0f), &u_C_int));

  // Outside on behind
  EXPECT_FALSE(radar.project(Vector3f(-x_dist, 0.0f, 0.0f), &u_C_float));
  EXPECT_FALSE(radar.project(Vector3f(-x_dist, 0.0f, 0.0f), &u_C_int));
}

TEST_P(ParameterizedRadarTest, PixelToRayExtremes) {
  // Radar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float horizontal_fov_deg = std::get<2>(params);
  const float vertical_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  const float half_horizontal_fov_rad = horizontal_fov_rad / 2.0f;
  const float half_vertical_fov_rad = vertical_fov_rad / 2.0f;

  Radar radar(num_azimuth_divisions, num_elevation_divisions, horizontal_fov_rad, vertical_fov_rad);

  // Special pixels to use
  const float middle_elevation_pixel = ((float)num_elevation_divisions - 1.0f) / 2.0f;
  const float middle_azimuth_pixel = ((float)num_azimuth_divisions -1.0f) / 2.0f;

  const float middle_elevation_coords = middle_elevation_pixel + 0.5f;
  const float middle_azimuth_coords = middle_azimuth_pixel + 0.5f;

  //-----------------
  // Azimuth extremes
  //-----------------

  // NOTE(alexmillane): We get larger than I would expect errors on some of the
  // components. I had to turn up the allowable error a bit.
  constexpr float kAllowableVectorError = 0.02;

  // Forwards
  Vector3f v_C = radar.vectorFromImagePlaneCoordinates(
      Vector2f(middle_azimuth_coords, middle_elevation_coords));
  EXPECT_NEAR(v_C.x(), 1.0f, kAllowableVectorError);
  EXPECT_NEAR(v_C.y(), 0.0f, kAllowableVectorError);
  EXPECT_NEAR(v_C.z(), 0.0f, kAllowableVectorError);

  // Right
  const Vector3f right_of_azimuth_forward_vector(
      cos(half_horizontal_fov_rad), -sin(half_horizontal_fov_rad), 0.0f);
  v_C = radar.vectorFromImagePlaneCoordinates(
      Vector2f(num_azimuth_divisions - 1 + 0.5, middle_elevation_coords));
  EXPECT_NEAR(v_C.dot(right_of_azimuth_forward_vector), 1.0f, kFloatEpsilon);

  // Left
  const Vector3f left_of_azimuth_forward_vector(
      cos(half_horizontal_fov_rad), sin(half_horizontal_fov_rad), 0.0f);
  v_C = radar.vectorFromImagePlaneCoordinates(
      Vector2f(0.5, middle_elevation_coords));
  EXPECT_NEAR(v_C.dot(left_of_azimuth_forward_vector), 1.0f, kFloatEpsilon);

  //-----------------
  // Elevation extremes
  //-----------------
  const Vector3f top_of_elevation_forward_vector(
      cos(half_vertical_fov_rad), 0.0f, sin(half_vertical_fov_rad));
  const Vector3f bottom_of_elevation_forward_vector(
      cos(half_vertical_fov_rad), 0.0f, -sin(half_vertical_fov_rad));

  v_C = radar.vectorFromImagePlaneCoordinates(
      Vector2f(middle_azimuth_coords, 0.5));
  EXPECT_NEAR(v_C.dot(top_of_elevation_forward_vector), 1.0f, kFloatEpsilon);

  v_C = radar.vectorFromImagePlaneCoordinates(
      Vector2f(middle_azimuth_coords, num_elevation_divisions - 1 + 0.5));
  EXPECT_NEAR(v_C.dot(bottom_of_elevation_forward_vector), 1.0f, kFloatEpsilon);
}

INSTANTIATE_TEST_CASE_P(
    ParameterizedRadarTests, ParameterizedRadarTest,
    ::testing::Values(std::tuple<int, int, float, float>(4, 3, 60.0f, 40.0f),
                      std::tuple<int, int, float, float>(1024, 16, 80.0f, 30.0f)));

// INSTANTIATE_TEST_CASE_P(
//     ParameterizedRadarTests, ParameterizedRadarTest,
//     ::testing::Values(std::tuple<int, int, float, float>(4, 3, 60.0f, 40.0f)));

TEST_P(ParameterizedRadarTest, RandomPixelRoundTrips) {
  // Make sure this is deterministic.
  std::srand(0);

  // Radar Params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float horizontal_fov_deg = std::get<2>(params);
  const float vertical_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  const float half_vertical_fov_rad = vertical_fov_rad / 2.0;

  Radar radar(num_azimuth_divisions, num_elevation_divisions, horizontal_fov_rad, vertical_fov_rad);

  // Test a large number of points
  const int kNumberOfPointsToTest = 10000;
  for (int i = 0; i < kNumberOfPointsToTest; i++) {
    // Random point on the image plane
    const Vector2f u_C(test_utils::randomFloatInRange(
                           0.0f, static_cast<float>(num_azimuth_divisions)),
                       test_utils::randomFloatInRange(
                           0.0f, static_cast<float>(num_elevation_divisions)));
    // Pixel -> Ray
    const Vector3f v_C = radar.vectorFromImagePlaneCoordinates(u_C);
    // Randomly scale ray
    const Vector3f p_C = v_C * test_utils::randomFloatInRange(0.1f, 10.0f);
    // Project back to image plane
    Vector2f u_C_reprojected;
    EXPECT_TRUE(radar.project(p_C, &u_C_reprojected));
    // Check we get back to where we started
    constexpr float kAllowableReprojectionError = 0.001;
    EXPECT_NEAR((u_C_reprojected - u_C).maxCoeff(), 0.0f,
                kAllowableReprojectionError);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
