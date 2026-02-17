# Agents

## Cloud-specific instructions

### Overview

This is the **Roc programming language** — a new compiler rewrite in **Zig** lives in `/workspace/src/` with `build.zig` at the repo root. The old Rust/LLVM compiler in `crates/` is legacy. Development targets the **Zig compiler**.

No databases or external services are needed.

### Environment

- **Zig 0.14.0** — installed at `/usr/local/zig`, binary symlinked to `/usr/local/bin/zig`. The `build.zig.zon` specifies `minimum_zig_version = "0.14.0"`.
- **LLVM 18** (optional) — only needed for `zig build -Dllvm`. The default `zig build` does not require LLVM.
- LLVM static lld libraries are NOT available from the Ubuntu `llvm-18-dev` package. Building with `-Dllvm` (without `-Dsystem-llvm`) will download its own LLVM, which takes time on first build. Prefer building without LLVM for fast iteration.

### Key commands

See `src/README.md` for the fast feedback loop. Quick reference:

| Task | Command |
|------|---------|
| Build | `zig build` |
| Build (with LLVM) | `zig build -Dllvm` |
| Fast feedback (no binary) | `zig build -Dno-bin -fincremental --watch` |
| Lint (format check) | `zig build check-fmt` |
| Lint (doc comments) | `./ci/zig_lints.sh` |
| Tests | `zig build test` |
| Snapshot tests | `zig build snapshot -- --debug` |
| Check a .roc file | `./zig-out/bin/roc check path/to/file.roc` |
| Format a .roc file | `./zig-out/bin/roc format path/to/file.roc` |

### Gotchas

- `zig build` (without `-Dllvm`) is the fast default — it builds in ~6 seconds and is sufficient for check, format, and most development.
- CI uses Zig 0.14.1 but 0.14.0 satisfies the `minimum_zig_version` and works fine.
- After running `zig build snapshot -- --debug`, run `git diff --exit-code src/snapshots` to verify no unexpected snapshot changes.
- `roc version`, `roc build`, `roc run`, and `roc repl` are not yet implemented in the new compiler. Use `roc check` and `roc format` to verify the compiler works.
- The `./ci/zig_lints.sh` script checks for `///` doc comments on `pub` declarations and `//!` top-level comments on new files.
