# AGENTS.md for Psi4

## Build and Test Feature
- Run: bash build.sh

## Code Style Guidelines
- **C++**: Follow .clang-format (LLVM style variant). Use auto for declarations, nullptr over NULL/0, std::make_shared, override keyword for virtuals. Print memory in GiB (1024-based). No comments unless necessary.
- **Python**: Follow .style.yapf (PEP8-based). Use snake_case for vars/functions, CapWords for classes. Imports: absolute, grouped (stdlib, third-party, local). Use typing hints. Error handling: raise exceptions, use logging.
- **General**: Mimic existing code. No emojis. Security: Avoid logging secrets. Naming: Descriptive, consistent with codebase (e.g., outfile->Printf for output).
