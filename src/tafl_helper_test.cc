
#include "tafl_helper.h"

#include "gtest/gtest.h"

namespace alphazero::tafl_helper {
namespace {

// NOLINTNEXTLINE
TEST(TaflHelper, Mirror) {
  int width = 5;
  int height = 5;
  PlayHistory base;
  base.v = Vector<float>{3};
  base.v.setRandom();
  base.pi = Vector<float>{width * height * (width + height)};
  base.pi.setZero();
  base.canonical = Tensor<float, 3>{3, height, width};
  base.canonical.setZero();

  base.canonical(0, 2, 1) = 1;
  base.canonical(1, 2, 3) = 1;
  base.canonical(1, 4, 1) = 1;

  base.pi(policyLocation(width, height, 2, 1, false, 0)) = 1;
  base.pi(policyLocation(width, height, 2, 1, false, 2)) = 1;

  base.pi(policyLocation(width, height, 2, 1, true, 0)) = 1;
  base.pi(policyLocation(width, height, 2, 1, true, 1)) = 1;
  base.pi(policyLocation(width, height, 2, 1, true, 3)) = 1;

  base.canonical(2, 2, 2) = 1;
  base.pi(policyLocation(width, height, 2, 2, false, 1)) = 1;
  base.pi(policyLocation(width, height, 2, 2, false, 4)) = 1;
  base.pi(policyLocation(width, height, 2, 2, true, 0)) = 1;
  base.pi(policyLocation(width, height, 2, 2, true, 3)) = 1;

  PlayHistory mirror = mirrorWidth(base);
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        if ((c == 0 && h == 2 && w == 3) || (c == 1 && h == 2 && w == 1) ||
            (c == 1 && h == 4 && w == 3) || (c == 2 && h == 2 && w == 2)) {
          EXPECT_EQ(mirror.canonical(c, h, w), 1)
              << "found no value at (" << c << "," << h << "," << w << ")";
        } else {
          EXPECT_EQ(mirror.canonical(c, h, w), 0)
              << "found a value at (" << c << "," << h << "," << w << ")";
        }
      }
    }
  }
  for (int from_h = 0; from_h < height; ++from_h) {
    for (int from_w = 0; from_w < width; ++from_w) {
      for (int w = 0; w < width; ++w) {
        if ((from_h == 2 && from_w == 3 && (w == 2 || w == 4)) ||
            (from_h == 2 && from_w == 2 && (w == 0 || w == 3))) {
          EXPECT_EQ(mirror.pi(policyLocation(width, height, from_h, from_w,
                                             false, w)),
                    1)
              << "found no policy at (" << from_h << "," << from_w << "," << w
              << ")";
        } else {
          EXPECT_EQ(mirror.pi(policyLocation(width, height, from_h, from_w,
                                             false, w)),
                    0)
              << "found a policy at (" << from_h << "," << from_w << "," << w
              << ")";
        }
      }
      for (int h = 0; h < height; ++h) {
        if ((from_h == 2 && from_w == 3 && (h == 0 || h == 1 || h == 3)) ||
            (from_h == 2 && from_w == 2 && (h == 0 || h == 3))) {
          EXPECT_EQ(
              mirror.pi(policyLocation(width, height, from_h, from_w, true, h)),
              1)
              << "found no policy at (" << from_h << "," << from_w << "," << h
              << ")";
        } else {
          EXPECT_EQ(
              mirror.pi(policyLocation(width, height, from_h, from_w, true, h)),
              0)
              << "found a policy at (" << from_h << "," << from_w << "," << h
              << ")";
        }
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(mirror.v(i), base.v(i));
  }
}

TEST(TaflHelper, Rot90) {
  int width = 5;
  int height = 5;
  PlayHistory base;
  base.v = Vector<float>{3};
  base.v.setRandom();
  base.pi = Vector<float>{width * height * (width + height)};
  base.pi.setZero();
  base.canonical = Tensor<float, 3>{3, height, width};
  base.canonical.setZero();

  base.canonical(0, 2, 1) = 1;
  base.canonical(1, 2, 3) = 1;
  base.canonical(1, 4, 1) = 1;

  base.pi(policyLocation(width, height, 2, 1, false, 0)) = 1;
  base.pi(policyLocation(width, height, 2, 1, false, 2)) = 1;

  base.pi(policyLocation(width, height, 2, 1, true, 0)) = 1;
  base.pi(policyLocation(width, height, 2, 1, true, 1)) = 1;
  base.pi(policyLocation(width, height, 2, 1, true, 3)) = 1;

  base.canonical(2, 2, 2) = 1;
  base.pi(policyLocation(width, height, 2, 2, false, 1)) = 1;
  base.pi(policyLocation(width, height, 2, 2, false, 4)) = 1;
  base.pi(policyLocation(width, height, 2, 2, true, 0)) = 1;
  base.pi(policyLocation(width, height, 2, 2, true, 3)) = 1;

  PlayHistory rot = rot90Clockwise(base);
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        if ((c == 0 && h == 1 && w == 2) || (c == 1 && h == 3 && w == 2) ||
            (c == 1 && h == 1 && w == 0) || (c == 2 && h == 2 && w == 2)) {
          EXPECT_EQ(rot.canonical(c, h, w), 1)
              << "found no value at (" << c << "," << h << "," << w << ")";
        } else {
          EXPECT_EQ(rot.canonical(c, h, w), 0)
              << "found a value at (" << c << "," << h << "," << w << ")";
        }
      }
    }
  }
  for (int from_h = 0; from_h < height; ++from_h) {
    for (int from_w = 0; from_w < width; ++from_w) {
      for (int w = 0; w < width; ++w) {
        if ((from_h == 1 && from_w == 2 && (w == 1 || w == 3 || w == 4)) ||
            (from_h == 2 && from_w == 2 && (w == 1 || w == 4))) {
          EXPECT_EQ(
              rot.pi(policyLocation(width, height, from_h, from_w, false, w)),
              1)
              << "found no policy at (" << from_h << "," << from_w << "," << w
              << ")";
        } else {
          EXPECT_EQ(
              rot.pi(policyLocation(width, height, from_h, from_w, false, w)),
              0)
              << "found a policy at (" << from_h << "," << from_w << "," << w
              << ")";
        }
      }
      for (int h = 0; h < height; ++h) {
        if ((from_h == 1 && from_w == 2 && (h == 0 || h == 2)) ||
            (from_h == 2 && from_w == 2 && (h == 1 || h == 4))) {
          EXPECT_EQ(
              rot.pi(policyLocation(width, height, from_h, from_w, true, h)), 1)
              << "found no policy at (" << from_h << "," << from_w << "," << h
              << ")";
        } else {
          EXPECT_EQ(
              rot.pi(policyLocation(width, height, from_h, from_w, true, h)), 0)
              << "found a policy at (" << from_h << "," << from_w << "," << h
              << ")";
        }
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(rot.v(i), base.v(i));
  }
}

}  // namespace
}  // namespace alphazero::tafl_helper