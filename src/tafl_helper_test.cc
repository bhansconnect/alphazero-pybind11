
#include "tafl_helper.h"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace alphazero::tafl_helper {
namespace {

// Build a PlayHistory whose pi and canonical entries are all-distinct, so any
// symmetry transform's effect is a detectable permutation. v is set to fixed
// values to check it is carried through unchanged.
PlayHistory MakeDistinct(int n) {
  PlayHistory base;
  base.v = Vector<float>{3};
  base.v(0) = 0.1f;
  base.v(1) = 0.2f;
  base.v(2) = 0.3f;
  const int nm = n * n * (n + n);
  base.pi = Vector<float>{nm};
  for (int i = 0; i < nm; ++i) {
    base.pi(i) = static_cast<float>(i + 1);
  }
  base.canonical = Tensor<float, 3>{3, n, n};
  int idx = 1;
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < n; ++h) {
      for (int w = 0; w < n; ++w) {
        base.canonical(c, h, w) = static_cast<float>(idx++);
      }
    }
  }
  return base;
}

bool SamePi(const PlayHistory& a, const PlayHistory& b) {
  if (a.pi.size() != b.pi.size()) return false;
  for (int i = 0; i < a.pi.size(); ++i) {
    if (a.pi(i) != b.pi(i)) return false;
  }
  return true;
}

bool SameCanonical(const PlayHistory& a, const PlayHistory& b) {
  for (int c = 0; c < a.canonical.dimension(0); ++c) {
    for (int h = 0; h < a.canonical.dimension(1); ++h) {
      for (int w = 0; w < a.canonical.dimension(2); ++w) {
        if (a.canonical(c, h, w) != b.canonical(c, h, w)) return false;
      }
    }
  }
  return true;
}

// A correct symmetry must be a bijection of the action space: the output pi is
// a pure rearrangement of the input, so the sorted multisets must match. (The
// original height-arg bug made rot90 map two source actions onto one target,
// leaving others unwritten -- caught here, missed by the hand-picked spot
// checks below.)
bool IsPiPermutation(const PlayHistory& out, const PlayHistory& base) {
  if (out.pi.size() != base.pi.size()) return false;
  std::vector<float> a(out.pi.data(), out.pi.data() + out.pi.size());
  std::vector<float> b(base.pi.data(), base.pi.data() + base.pi.size());
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

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

TEST(TaflHelper, EightSym) {
  int width = 2;
  int height = 2;
  PlayHistory base;
  base.v = Vector<float>{3};
  base.v.setRandom();
  base.pi = Vector<float>{width * height * (width + height)};
  base.pi.setZero();
  base.canonical = Tensor<float, 3>{3, height, width};
  base.canonical.setZero();

  base.canonical(0, 0, 0) = 1;
  base.canonical(0, 0, 1) = 2;
  base.canonical(0, 1, 0) = 3;
  base.canonical(0, 1, 1) = 4;

  std::vector<PlayHistory> syms = eightSym(base);

  std::cout << syms.size() << '\n';
  for (auto& sym : syms) {
    std::cout << sym.canonical(0, 0, 0) << ',' << sym.canonical(0, 0, 1)
              << '\n';
    std::cout << sym.canonical(0, 1, 0) << ',' << sym.canonical(0, 1, 1)
              << '\n';
    std::cout << '\n';
  }
  EXPECT_TRUE(true);
}

// rot90 must be a bijection on pi and have order 4 (four turns == identity),
// for every real tafl board size. Distinct entries make any dropped/duplicated
// index visible.
TEST(TaflHelper, Rot90IsBijectionAndOrder4) {
  for (int n : {5, 7, 11}) {
    PlayHistory base = MakeDistinct(n);
    PlayHistory r1 = rot90Clockwise(base);
    EXPECT_TRUE(IsPiPermutation(r1, base))
        << "rot90 pi is not a permutation for n=" << n;
    PlayHistory r4 =
        rot90Clockwise(rot90Clockwise(rot90Clockwise(r1)));
    EXPECT_TRUE(SamePi(r4, base)) << "rot90^4 != identity (pi) for n=" << n;
    EXPECT_TRUE(SameCanonical(r4, base))
        << "rot90^4 != identity (canonical) for n=" << n;
    for (int i = 0; i < 3; ++i) EXPECT_EQ(r1.v(i), base.v(i));
  }
}

// mirrorWidth must be a bijection on pi and have order 2.
TEST(TaflHelper, MirrorIsBijectionAndOrder2) {
  for (int n : {5, 7, 11}) {
    PlayHistory base = MakeDistinct(n);
    PlayHistory m1 = mirrorWidth(base);
    EXPECT_TRUE(IsPiPermutation(m1, base))
        << "mirror pi is not a permutation for n=" << n;
    PlayHistory m2 = mirrorWidth(m1);
    EXPECT_TRUE(SamePi(m2, base)) << "mirror^2 != identity (pi) for n=" << n;
    EXPECT_TRUE(SameCanonical(m2, base))
        << "mirror^2 != identity (canonical) for n=" << n;
    for (int i = 0; i < 3; ++i) EXPECT_EQ(m1.v(i), base.v(i));
  }
}

// The 8 dihedral symmetries must each be a pi permutation, carry v unchanged,
// and be pairwise distinct (the group acts faithfully on a distinct board).
TEST(TaflHelper, EightSymAreDistinctPermutations) {
  const int n = 7;
  PlayHistory base = MakeDistinct(n);
  std::vector<PlayHistory> syms = eightSym(base);
  EXPECT_EQ(syms.size(), 8u);
  for (const auto& s : syms) {
    EXPECT_TRUE(IsPiPermutation(s, base));
    for (int i = 0; i < 3; ++i) EXPECT_EQ(s.v(i), base.v(i));
  }
  for (size_t i = 0; i < syms.size(); ++i) {
    for (size_t j = i + 1; j < syms.size(); ++j) {
      EXPECT_FALSE(SamePi(syms[i], syms[j]))
          << "symmetries " << i << " and " << j << " are identical";
    }
  }
}

// Spatial-head equivariance guard: the pi permutation must equal a TRUE board
// transform of each move's from-square AND its target-square, including the
// row-slide<->column-slide axis swap a 90-deg rotation induces. Bijection /
// group-order tests alone would NOT catch a clockwise-pi-vs-counter-clockwise-
// board or a mismatched axis swap -- this would. `xf` maps (h,w)->(h',w').
template <typename XF>
void ExpectGeometricEquivariance(int n, const PlayHistory& out,
                                 const PlayHistory& base, XF xf) {
  const int span = n + n;
  for (int m = 0; m < n * n * span; ++m) {
    int nl = m % span;
    const bool hm = nl >= n;
    if (hm) nl -= n;
    const int pl = m / span;
    const int fh = pl / n;
    const int fw = pl % n;
    const int th = hm ? nl : fh;  // target square of the slide
    const int tw = hm ? fw : nl;
    const std::pair<int, int> nf = xf(fh, fw);
    const std::pair<int, int> nt = xf(th, tw);
    if (nf == nt) continue;  // degenerate self-move (never a legal tafl move)
    const int nm = (nf.first == nt.first)
                       ? policyLocation(n, n, nf.first, nf.second, false,
                                        nt.second)
                       : policyLocation(n, n, nf.first, nf.second, true,
                                        nt.first);
    EXPECT_EQ(out.pi(nm), base.pi(m))
        << "geometric mismatch for move " << m << " at n=" << n;
  }
}

TEST(TaflHelper, Rot90IsGeometricallyEquivariant) {
  for (int n : {5, 7, 11}) {
    PlayHistory base = MakeDistinct(n);
    PlayHistory out = rot90Clockwise(base);
    ExpectGeometricEquivariance(n, out, base, [n](int h, int w) {
      return std::make_pair(w, n - 1 - h);  // 90-deg clockwise
    });
  }
}

TEST(TaflHelper, MirrorIsGeometricallyEquivariant) {
  for (int n : {5, 7, 11}) {
    PlayHistory base = MakeDistinct(n);
    PlayHistory out = mirrorWidth(base);
    ExpectGeometricEquivariance(n, out, base, [n](int h, int w) {
      return std::make_pair(h, n - 1 - w);  // mirror across width
    });
  }
}

}  // namespace
}  // namespace alphazero::tafl_helper