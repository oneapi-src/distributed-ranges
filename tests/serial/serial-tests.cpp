#include <gtest/gtest.h>

// Demonstrate some basic assertions.
TEST(CpuTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto res = RUN_ALL_TESTS();

  return res;
}
