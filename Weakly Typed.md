Compare with: [[Strongly Typed]]

A **weakly typed** language permits implicit conversions between unrelated types, often coercing values silently to make an expression type-check or evaluate.

Examples:
- JavaScript: `"5" + 3` yields `"53"` (number coerced to string); `"5" - 3` yields `2`.
- C: pointers, integers, and characters convert freely; you can reinterpret bytes via casts.

The convenience can mask bugs: a typo or unexpected value silently becomes a valid-looking result rather than an error. "Weak" vs. "strong" typing is a spectrum and is orthogonal to static vs. dynamic typing.
