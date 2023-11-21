#include "torch_xla/csrc/runtime/util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <set>
#include <unordered_map>
#include <vector>

#include "absl/types/span.h"

namespace torch_xla {
namespace runtime {
namespace util {

using ::testing::ElementsAre;

TEST(UtilTest, Range) {
  EXPECT_THAT(Range<int16_t>(0, 10, 2), ElementsAre(0, 2, 4, 6, 8));
  EXPECT_THAT(Range<int16_t>(10, 0, -2), ElementsAre(10, 8, 6, 4, 2));
}

TEST(UtilTest, ToVector) {
  EXPECT_THAT(ToVector<char>(std::string("char")),
              ElementsAre('c', 'h', 'a', 'r'));
}

TEST(UtilTest, Equal) {
  EXPECT_TRUE(Equal(std::string("this"), std::string("this")));
  EXPECT_FALSE(Equal(std::string("this"), std::string("that")));
}

TEST(UtilTest, FindOr) {
  std::unordered_map<int, int> v = {{1, 1}, {2, 2}, {3, 3}};
  EXPECT_EQ(FindOr(v, 1, 7), 1);
  EXPECT_EQ(FindOr(v, 2, 7), 2);
  EXPECT_EQ(FindOr(v, 3, 7), 3);
  EXPECT_EQ(FindOr(v, 10, 7), 7);
}

TEST(UtilTest, MapInsert) {
  std::unordered_map<int, int> v;
  EXPECT_EQ(MapInsert(&v, 1, [] { return 1; }), 1);
  EXPECT_EQ(MapInsert(&v, 1, [] { return 7; }), 1);
  EXPECT_EQ(MapInsert(&v, 1, [] { return 12; }), 1);
}

TEST(UtilTest, Multiply) {
  std::vector<int32_t> t = {1, 2, 3, 4, 5};
  EXPECT_EQ(Multiply<int32_t>(t), 120);
  t.push_back(6);
  EXPECT_EQ(Multiply<int32_t>(t), 720);
}

TEST(UtilTest, MaybeRef) {
  using StringRef = torch_xla::runtime::util::MaybeRef<std::string>;
  std::string storage("String storage");
  StringRef ref_storage(storage);
  EXPECT_FALSE(ref_storage.is_stored());
  EXPECT_EQ(*ref_storage, storage);

  StringRef eff_storage(std::string("Vanishing"));
  EXPECT_TRUE(eff_storage.is_stored());
  EXPECT_EQ(*eff_storage, "Vanishing");
}

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla
