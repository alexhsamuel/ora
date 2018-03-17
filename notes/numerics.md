In principle, we should be able to use
[fenv](http://en.cppreference.com/w/cpp/header/cfenv) to detect, for instance,
`FE_INVALID` when a conversion from float to int is out of range and results
in an invalid int value.

However, both GCC and
[LLVM](http://lists.llvm.org/pipermail/llvm-dev/2017-May/112918.html) do not
support the `FENV_ACCESS` pragma, and their optimizations may reorder
instructions such that FP flags cannot be tested immediately.

Even if this is resolved, it is not clear that the performance hit will be
acceptable, as the FP flag operations seem to be implemented as full function
calls rather than builtins.

