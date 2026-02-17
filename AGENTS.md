# Agents

## Cloud-specific instructions

### Overview

This is the **Roc programming language** compiler — a Rust monorepo. The main binary is `roc` (built from `crates/cli`). No databases or external services are needed.

### Environment

- **Rust 1.77.2** — pinned in `rust-toolchain.toml`; rustup auto-selects it in the workspace.
- **LLVM 18** — must set `LLVM_SYS_180_PREFIX=/usr/lib/llvm-18` (already in `~/.bashrc`).
- **Zig 0.13.0** — installed at `/usr/local/zig`, binary symlinked to `/usr/local/bin/zig`.
- **lld** — configured in `~/.cargo/config.toml` for faster linking (`-fuse-ld=lld`).
- **libstdc++** — a symlink at `/usr/lib/x86_64-linux-gnu/libstdc++.so` was needed on Ubuntu 24.04 because `libstdc++-13-dev` installs only under `/usr/lib/gcc/x86_64-linux-gnu/13/`.

### Key commands

See `CONTRIBUTING.md` for the standard contributor workflow. Quick reference:

| Task | Command |
|------|---------|
| Build (dev) | `cargo build --bin roc` |
| Build (release) | `cargo build --release --bin roc` |
| Lint (format) | `cargo fmt --all -- --check` |
| Lint (clippy) | `cargo clippy --workspace --tests -- --deny warnings` |
| Test (all) | `cargo test --release` |
| Test (specific crate) | `cargo test -p roc_parse --release` |
| REPL | `cargo run -- repl` |
| Check a .roc file | `cargo run -- check path/to/file.roc` |

### Gotchas

- Always export `LLVM_SYS_180_PREFIX=/usr/lib/llvm-18` before building. Without it, `llvm-sys` will fail with "No suitable version of LLVM was found."
- The first full build takes ~2 minutes in debug mode. Subsequent incremental builds are much faster.
- CLI test platform hosts (zig-based) are built automatically by the test harness via `copy_zig_glue::initialize_zig_test_platforms()`. You don't need to build them manually; just run `cargo test -p roc_cli --release`.
- `--build-host` flag for `roc build` may panic with "not implemented" for local zig platforms — this is expected for the dev backend; tests use the LLVM backend with `--optimize`.
- Valgrind tests are skipped if valgrind is not installed. On this VM, valgrind is installed.
