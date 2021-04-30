#include "connect4_gs.h"

#include "gtest/gtest.h"

namespace alphazero::connect4_gs {
namespace {

// NOLINTNEXTLINE
TEST(Connect4GS, Equals) {
  auto x = Connect4GS{};
  auto y = Connect4GS{};
  EXPECT_EQ(x, y);

  x.play_move(0);
  EXPECT_NE(x, y);

  y.play_move(0);
  EXPECT_EQ(x, y);

  x = Connect4GS{};
  y = Connect4GS{};
  x.play_move(0);
  x.play_move(1);
  x.play_move(2);
  y.play_move(2);
  y.play_move(1);
  y.play_move(0);
  EXPECT_EQ(x, y);
}

// NOLINTNEXTLINE
TEST(Connect4GS, Copy) {
  auto x = Connect4GS{};
  auto y = x.copy();
  EXPECT_EQ(x, *y);

  y->play_move(0);
  y->play_move(1);
  y->play_move(2);
  EXPECT_NE(x, *y);

  auto z = y->copy();
  EXPECT_EQ(*y, *z);
  EXPECT_NE(x, *z);

  x.play_move(2);
  x.play_move(1);
  x.play_move(0);
  EXPECT_EQ(x, *y);
  EXPECT_EQ(x, *z);
}

// NOLINTNEXTLINE
TEST(Connect4GS, ValidMoves) {
  auto x = Connect4GS{};
  auto valids = x.valid_moves();
  auto expected = Vector<uint8_t>{WIDTH};
  expected.setConstant(1);
  EXPECT_EQ(valids, expected);

  auto board = SizedTensor<int8_t, Eigen::Sizes<2, HEIGHT, WIDTH>>{};
  board.setZero();
  board(0, 0, 3) = 1;
  board(1, 0, 5) = 1;

  x = Connect4GS{board, 0, 0};
  valids = x.valid_moves();

  expected(3) = 0;
  expected(5) = 0;
  EXPECT_EQ(valids, expected);
}

// NOLINTNEXTLINE
TEST(Connect4GS, PlayMove) {
  auto board = SizedTensor<int8_t, Eigen::Sizes<2, HEIGHT, WIDTH>>{};
  board.setZero();

  auto x = Connect4GS{};
  auto y = Connect4GS{board, 0, 0};
  EXPECT_EQ(x, y);

  auto i = 0;
  for (auto h = HEIGHT - 1; h >= 1; h -= 2) {
    x.play_move(3);
    board(0, h, 3) = 1;
    y = Connect4GS{board, 1, ++i};
    EXPECT_EQ(x, y);

    x.play_move(3);
    board(1, h - 1, 3) = 1;
    y = Connect4GS{board, 0, ++i};
    EXPECT_EQ(x, y);
  }

  try {
    x.play_move(3);
  } catch (std::exception& e) {
    EXPECT_STREQ("Invalid move: You have a bug in your code.", e.what());
  }
}

// NOLINTNEXTLINE
TEST(Connect4GS, WinState) {
  auto x = Connect4GS{};
  EXPECT_EQ(x.scores(), std::nullopt);

  auto board = SizedTensor<int8_t, Eigen::Sizes<2, HEIGHT, WIDTH>>{};
  auto expected = std::optional<SizedVector<float, 3>>{};

  board.setZero();
  board(0, 3, 0) = 1;
  board(0, 3, 1) = 1;
  board(0, 3, 2) = 1;
  board(0, 3, 3) = 1;
  x = Connect4GS{board, 0, 0};
  expected = {1, 0, 0};
  EXPECT_EQ(x.scores(), expected);

  board(0, 3, 2) = 0;
  x = Connect4GS{board, 0, 0};
  expected = std::nullopt;
  EXPECT_EQ(x.scores(), expected);

  board(1, 1, 2) = 1;
  board(1, 2, 2) = 1;
  board(1, 3, 2) = 1;
  board(1, 4, 2) = 1;
  x = Connect4GS{board, 0, 0};
  expected = {0, 1, 0};
  EXPECT_EQ(x.scores(), expected);

  board(1, 2, 2) = 0;
  x = Connect4GS{board, 0, 0};
  expected = std::nullopt;
  EXPECT_EQ(x.scores(), expected);

  board(0, 1, 1) = 1;
  board(0, 2, 2) = 1;
  board(0, 3, 3) = 1;
  board(0, 4, 4) = 1;
  x = Connect4GS{board, 0, 0};
  expected = {1, 0, 0};
  EXPECT_EQ(x.scores(), expected);

  board(0, 2, 2) = 0;
  x = Connect4GS{board, 0, 0};
  expected = std::nullopt;
  EXPECT_EQ(x.scores(), expected);

  board(1, 1, 3) = 1;
  board(1, 2, 2) = 1;
  board(1, 3, 1) = 1;
  board(1, 4, 0) = 1;
  x = Connect4GS{board, 0, 0};
  expected = {0, 1, 0};
  EXPECT_EQ(x.scores(), expected);

  board(1, 2, 2) = 0;
  x = Connect4GS{board, 0, 0};
  expected = std::nullopt;
  EXPECT_EQ(x.scores(), expected);

  board.setZero();
  for (auto w = 0; w < WIDTH; ++w) {
    board(w % 2, 0, w) = 1;
  }
  x = Connect4GS{board, 0, 0};
  expected = {0, 0, 1};
  EXPECT_EQ(x.scores(), expected);
}

// NOLINTNEXTLINE
TEST(Connect4GS, Canonicalize) {
  auto expected = SizedTensor<float, Eigen::Sizes<4, HEIGHT, WIDTH>>{};
  auto x = Connect4GS{};

  expected.setZero();
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      expected(2, h, w) = 1;
    }
  }
  auto canonical = x.canonicalized();
  for (auto c = 0; c < 4; ++c) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        ASSERT_EQ(canonical(c, h, w), expected(c, h, w));
      }
    }
  }

  x.play_move(0);
  expected.setZero();
  expected(0, HEIGHT - 1, 0) = 1;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      expected(3, h, w) = 1;
    }
  }
  canonical = x.canonicalized();
  for (auto c = 0; c < 4; ++c) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        ASSERT_EQ(canonical(c, h, w), expected(c, h, w));
      }
    }
  }

  x.play_move(0);
  expected.setZero();
  expected(0, HEIGHT - 1, 0) = 1;
  expected(1, HEIGHT - 2, 0) = 1;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      expected(2, h, w) = 1;
    }
  }
  canonical = x.canonicalized();
  for (auto c = 0; c < 4; ++c) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        ASSERT_EQ(canonical(c, h, w), expected(c, h, w));
      }
    }
  }
}

}  // namespace
}  // namespace alphazero::connect4_gs