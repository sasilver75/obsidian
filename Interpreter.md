An **interpreter** executes source code directly, statement by statement, without first producing a standalone machine-code binary. It reads the program (or an intermediate representation like bytecode) and carries out its instructions at runtime.

Examples: CPython, Ruby's MRI, the original Lisp interpreter, browser JavaScript engines (which today combine interpretation with JIT compilation).

Trade-offs vs. a compiler:
- Faster iteration loop (no separate build step) and easier interactive use (REPLs).
- Generally slower execution than fully compiled code, though JIT compilation closes much of the gap.
