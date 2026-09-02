#include <iostream>
#include <concepts>
#include <source_location>

#define OK "OK"
#define FAIL "FAILED"

template <typename T>
concept IntOrFloat = std::same_as<T, int> || std::same_as<T, float>;

void test_compare(
  IntOrFloat auto x, 
  IntOrFloat auto y,
  const std::source_location& loc = std::source_location::current()) {
  std::cout << loc.function_name() << " -> ";

  if(x == y) {
    std::cout << OK << std::endl;
  } else {
    std::cout << FAIL << std::endl;
  }
}