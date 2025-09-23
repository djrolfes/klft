#include <assert.h>

#include <iomanip>

#include "GLOBAL.hpp"
#include "IndexHelper.hpp"
#include "SUN.hpp"
#include "Spinor.hpp"
using namespace klft;

#define HLINE "=========================================================\n"
struct TagOperatorBase {
  /* data */
};
struct TagHWilson : TagOperatorBase {
  /* data */
};

struct TagWilson : TagOperatorBase {
  /* data */
};

struct Foo {
  TagOperatorBase tag;
  Foo(const TagOperatorBase& tag) : tag(tag) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const TagWilson,
                  const Kokkos::TeamPolicy<>::member_type& team) const {
    printf("Greetings from thread %i of team %i with TagWilson\n",
           team.thread_rank(), team.league_rank());
  }
  KOKKOS_INLINE_FUNCTION
  void operator()(const TagHWilson,
                  const Kokkos::TeamPolicy<>::member_type& team) const {
    printf("Greetings from thread %i of team %i with TagHWilson\n",
           team.thread_rank(), team.league_rank());
  }
  void run() {
    Kokkos::parallel_for(Kokkos::TeamPolicy<>(N, Kokkos::AUTO), foo);
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Foo foo;

    Kokkos::parallel_for("Loop2", Kokkos::TeamPolicy<TagB>(N, Kokkos::AUTO),
                         foo);
  }
  Kokkos::finalize();
  return 0;
}