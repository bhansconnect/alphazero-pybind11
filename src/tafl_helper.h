#pragma once

#include "game_state.h"

namespace alphazero::tafl_helper {

int policyLocation(int width, int height, int from_h, int from_w,
                   bool height_move, int new_loc) {
  if (height_move) {
    return (from_h * width + from_w) * (width + height) + width + new_loc;
  } else {
    return (from_h * width + from_w) * (width + height) + new_loc;
  }
}

PlayHistory mirrorWidth(const PlayHistory& base) noexcept {
  int channels = base.canonical.dimension(0);
  int height = base.canonical.dimension(1);
  int width = base.canonical.dimension(2);

  PlayHistory out;
  out.v = Vector<float>{3};
  out.canonical = Tensor<float, 3>{channels, height, width};
  out.pi = Vector<float>{base.pi.size()};
  // out.v.setZero();
  // out.canonical.setZero();
  // out.pi.setZero();
  out.v = base.v;

  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        out.canonical(c, h, width - 1 - w) = base.canonical(c, h, w);
      }
    }
  }

  for (int from_h = 0; from_h < height; ++from_h) {
    for (int from_w = 0; from_w < width; ++from_w) {
      for (int w = 0; w < width; ++w) {
        out.pi(policyLocation(width, height, from_h, width - 1 - from_w, false,
                              width - 1 - w)) =
            base.pi(policyLocation(width, height, from_h, from_w, false, w));
      }
      for (int h = 0; h < height; ++h) {
        out.pi(policyLocation(width, height, from_h, width - 1 - from_w, true,
                              h)) =
            base.pi(policyLocation(width, height, from_h, from_w, true, h));
      }
    }
  }
  return out;
}

PlayHistory rot90Clockwise(const PlayHistory& base) noexcept {
  int channels = base.canonical.dimension(0);
  int height = base.canonical.dimension(1);
  int width = base.canonical.dimension(2);
  assert(width == height);

  PlayHistory out;
  out.v = Vector<float>{3};
  out.canonical = Tensor<float, 3>{channels, height, width};
  out.pi = Vector<float>{base.pi.size()};
  // out.v.setZero();
  // out.canonical.setZero();
  // out.pi.setZero();
  out.v = base.v;

  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height / 2; ++h) {
      for (int w = h; w < height - h - 1; ++w) {
        out.canonical(c, h, w) = base.canonical(c, height - 1 - w, h);
        out.canonical(c, height - 1 - w, h) =
            base.canonical(c, height - 1 - h, height - 1 - w);
        out.canonical(c, height - 1 - h, height - 1 - w) =
            base.canonical(c, w, height - 1 - h);
        out.canonical(c, w, height - 1 - h) = base.canonical(c, h, w);
      }
    }
    if (height % 2 == 1) {
      out.canonical(c, height / 2, width / 2) =
          base.canonical(c, height / 2, width / 2);
    }
  }

  for (int base_h = 0; base_h < height / 2; ++base_h) {
    for (int base_w = base_h; base_w < height - base_h - 1; ++base_w) {
      for (int w = 0; w < width; ++w) {
        out.pi(policyLocation(width, height, base_h, base_w, false, w)) =
            base.pi(policyLocation(width, height, height - 1 - base_w, base_h,
                                   true, width - 1 - w));
        out.pi(policyLocation(width, height, height - 1 - base_w, base_h, false,
                              w)) =
            base.pi(policyLocation(width, height, height - 1 - base_h,
                                   height - 1 - base_w, true, width - 1 - w));
        out.pi(policyLocation(width, height, height - 1 - base_h,
                              height - 1 - base_w, false, w)) =
            base.pi(policyLocation(width, height, base_w, height - 1 - base_h,
                                   true, width - 1 - w));
        out.pi(policyLocation(width, height, base_w, height - 1 - base_h, false,
                              w)) =
            base.pi(policyLocation(width, base_h, base_w, height - 1 - base_w,
                                   true, width - 1 - w));
      }
      for (int h = 0; h < height; ++h) {
        out.pi(policyLocation(width, height, base_h, base_w, true, h)) =
            base.pi(policyLocation(width, height, height - 1 - base_w, base_h,
                                   false, h));
        out.pi(policyLocation(width, height, height - 1 - base_w, base_h, true,
                              h)) =
            base.pi(policyLocation(width, height, height - 1 - base_h,
                                   height - 1 - base_w, false, h));
        out.pi(policyLocation(width, height, height - 1 - base_h,
                              height - 1 - base_w, true, h)) =
            base.pi(policyLocation(width, height, base_w, height - 1 - base_h,
                                   false, h));
        out.pi(policyLocation(width, height, base_w, height - 1 - base_h, true,
                              h)) =
            base.pi(policyLocation(width, base_h, base_w, height - 1 - base_w,
                                   false, h));
      }
    }
  }
  if (height % 2 == 1) {
    for (int w = 0; w < width; ++w) {
      out.pi(policyLocation(width, height, height / 2, width / 2, false, w)) =
          base.pi(policyLocation(width, height, height / 2, width / 2, true,
                                 width - 1 - w));
    }
    for (int h = 0; h < height; ++h) {
      out.pi(policyLocation(width, height, height / 2, width / 2, true, h)) =
          base.pi(
              policyLocation(width, height, height / 2, width / 2, false, h));
    }
  }
  return out;
}

[[nodiscard]] std::vector<PlayHistory> eightSym(
    const PlayHistory& base) noexcept {
  std::vector<PlayHistory> out{base};
  for (int i = 0; i < 3; ++i) {
    out.push_back(rot90Clockwise(out[i]));
  }
  for (int i = 0; i < 4; ++i) {
    out.push_back(mirrorWidth(out[i]));
  }
  return out;
}

}  // namespace alphazero::tafl_helper