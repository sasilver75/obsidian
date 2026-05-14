A **compiler** translates source code in one language (usually a high-level language) into another language (typically machine code, bytecode, or another high-level language) ahead of execution. The translated artifact is then run directly by the target machine or runtime.

Phases typically include: lexing, parsing, semantic analysis, optimization, and code generation.

Examples: GCC and Clang (C/C++), rustc (Rust), javac (Java → JVM bytecode), TypeScript (TS → JS).

Contrast with an interpreter, which executes source code directly without producing a separate compiled artifact. Many modern languages blur the line via JIT compilation.
