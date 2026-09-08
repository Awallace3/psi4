.. #
.. # @BEGIN LICENSE
.. #
.. # Psi4: an open-source quantum chemistry software package
.. #
.. # Copyright (c) 2007-2025 The Psi4 Developers.
.. #
.. # The copyrights for code used from other parties are included in
.. # the corresponding files.
.. #
.. # This file is part of Psi4.
.. #
.. # Psi4 is free software; you can redistribute it and/or modify
.. # it under the terms of the GNU Lesser General Public License as published by
.. # the Free Software Foundation, version 3.
.. #
.. # Psi4 is distributed in the hope that it will be useful,
.. # but WITHOUT ANY WARRANTY; without even the implied warranty of
.. # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. # GNU Lesser General Public License for more details.
.. #
.. # You should have received a copy of the GNU Lesser General Public License along
.. # with Psi4; if not, write to the Free Software Foundation, Inc.,
.. # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
.. #
.. # @END LICENSE
.. #

.. include:: autodoc_abbr_options_c.rst

.. _`sec:prog_lsp`:

clangd + Neovim for Psi4 C++ development
========================================

Verified on Arch Linux, Neovim 0.11.6, clangd 22.1.8, Psi4 built against a conda ``p4dev`` environment (gcc 14.3.0).

1. Install clangd
-----------------

Any reasonably recent clangd works; you do **not** need a conda-provided one.

.. code:: bash

   # Arch
   sudo pacman -S clang            # provides /usr/bin/clangd
   # Debian/Ubuntu
   sudo apt install clangd
   # conda, if you prefer to keep it in the environment
   conda install -c conda-forge clang-tools

A system clangd resolves the conda GCC's ``libstdc++`` headers correctly, because ``compile_commands.json`` records the compiler as an absolute, target-prefixed path (``$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++``). clangd infers ``--target=x86_64-conda-linux-gnu --driver-mode=g++`` from that name and walks up from the driver to find the toolchain and sysroot. See §6 if headers still fail.

2. Generate ``compile_commands.json``
-------------------------------------

Configure with ``-DCMAKE_EXPORT_COMPILE_COMMANDS=ON``:

.. code:: bash

   cmake -S . -B objdir -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   cmake --build objdir

Run ``cmake`` from inside the same activated environment you build in. Psi4's ``FindMathOpenMP`` locates Intel-style OpenMP with a bare ``find_library(NAMES iomp5)``, so reconfiguring outside an activated conda environment silently resolves it to ``MathOpenMP_LIBRARIES-NOTFOUND`` and the next link of ``core.*.so`` fails with ``cannot find -lMathOpenMP_LIBRARIES-NOTFOUND``. Reconfiguring inside the environment repairs it.

**Psi4 is a superbuild, so the database is not at the top of the objdir.** It is written only for the core project:

::

   <objdir>/psi4-core-prefix/src/psi4-core-build/compile_commands.json

It covers the ~1200 ``psi4-core`` translation units. Bundled externals under ``external/`` are not included; their installed headers are still found via the ``-isystem $CONDA_PREFIX/include`` flags recorded in the database.

Make the database easy for clangd to find
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After the build creates the core compilation database, expose it at the repository root:

.. code:: bash

   ln -sfn "objdir/psi4-core-prefix/src/psi4-core-build/compile_commands.json" \
           compile_commands.json

Replace ``objdir`` with the build directory passed to ``cmake -B``. CMake updates the database in place as the build configuration changes. Recreate the symlink only when switching to a different build directory.

3. Add a ``.clangd`` to the Psi4 root
-------------------------------------

If the root symlink from §2 exists, use:

.. code:: yaml

   CompileFlags:
     CompilationDatabase: .
   Documentation:
     CommentFormat: Doxygen

Without the symlink, name the directory explicitly:

.. code:: yaml

   CompileFlags:
     CompilationDatabase: <objdir>/psi4-core-prefix/src/psi4-core-build
   Documentation:
     CommentFormat: Doxygen

Relative paths are resolved against the ``.clangd`` file, **not** the editor's working directory, so a relative path survives moving the checkout and can be copied into git worktrees. ``.clangd``, ``compile_commands.json``, ``.cache/``, and ``build*`` are all already covered by Psi4's ``.gitignore``.

4. Configure Neovim (>= 0.11)
-----------------------------

.. code:: lua

   vim.lsp.config("clangd", {
     cmd = {
       "clangd",
       "-j=8",                              -- background indexing threads
       "--background-index-priority=low",   -- ~1200 TUs will otherwise peg every core
       "--header-insertion=never",          -- Psi4 leans hard on transitive includes
     },
     filetypes = { "c", "c.doxygen", "cpp", "cpp.doxygen" },
     root_markers = { ".clangd", "compile_commands.json", ".git" },
   })

   vim.lsp.enable("clangd")

Give ``cmd`` an absolute path to clangd if ``clangd`` is not on ``PATH`` when nvim starts. Keep the extra root markers: with only ``.clangd``, a checkout that lacks the file gets no root directory and clangd runs in single-file mode. (``.git`` is a *file* in git worktrees; marker matching handles that.)

Earlier Neovim versions can use ``nvim-lspconfig`` instead of ``vim.lsp.config``.

Keymaps
~~~~~~~

Neovim 0.11 already binds most of what you want on LSP attach: ``K`` (hover), ``grn`` (rename), ``gra`` (code action), ``grr`` (references), ``gri`` (implementation), ``gO`` (document symbols), and ``[d`` / ``]d`` (previous/next diagnostic). Only add what is missing, or override to taste:

.. code:: lua

   vim.api.nvim_create_autocmd("LspAttach", {
     group = vim.api.nvim_create_augroup("UserLspConfig", { clear = true }),
     callback = function(ev)
       local opts = { buffer = ev.buf }
       vim.keymap.set("n", "gd", vim.lsp.buf.definition,
         vim.tbl_extend("force", opts, { desc = "Go to definition" }))
       vim.keymap.set("n", "gl", vim.diagnostic.open_float,
         vim.tbl_extend("force", opts, { desc = "Show line diagnostics" }))
       vim.keymap.set("n", "dl", vim.diagnostic.setloclist,
         vim.tbl_extend("force", opts, { desc = "Open diagnostic list" }))
     end,
   })

5. Verify before waiting on the index
-------------------------------------

Do not wait several minutes to find out the database was never loaded. Check a single file headlessly:

.. code:: bash

   clangd --check=psi4/src/psi4/libmints/molecule.cc

Look for exactly one line:

::

   Loaded compilation database from .../psi4-core-build/compile_commands.json

``Failed to find compilation database`` means the database was not picked up and everything downstream is guesswork.

**Ignore the error count on the last line.** On a perfectly healthy file ``--check`` reports something like ``All checks completed, 43 errors``; those are all ``tweak: ExtractFunction ==> FAIL`` refactor-availability probes, not compiler diagnostics. Real problems appear as ``error:`` / ``fatal error:`` lines. In nvim, the equivalent check is the ``Loaded compilation database`` line in ``:LspLog``.

Once the database loads, a preamble builds in under two seconds and background indexing populates ``.cache/clangd/index`` at the project root (~19 MB for Psi4). Cross-references and go-to-definition across the tree become complete when indexing finishes; hover, diagnostics, and same-TU navigation work immediately.

6. Troubleshooting
------------------

**Headers not found / ``<vector>`` missing.** clangd could not locate the toolchain from the driver path in the database — usually because the database records a bare ``c++`` or ``cc``. Whitelist the real driver so clangd may execute it to extract system include paths:

.. code:: lua

   "--query-driver=/usr/bin/g++," .. os.getenv("CONDA_PREFIX") .. "/bin/x86_64-conda-linux-gnu-c++",

This is a fallback, not a requirement: with the target-prefixed conda compiler in the database, adding ``--query-driver`` changes nothing.

**clangd crashes or fails to start inside an activated conda environment.** A system clangd will load conda's LLVM if ``LD_LIBRARY_PATH`` points into the environment. Check with ``ldd $(which clangd) | grep LLVM``; if it resolves to ``$CONDA_PREFIX/lib/libLLVM.so.*`` rather than ``/usr/lib``, the two versions must match by luck. Either start nvim outside the environment, or pin the libraries:

.. code:: lua

   cmd = { "env", "-u", "LD_LIBRARY_PATH", "/usr/bin/clangd", "-j=8" },

**Completions look plausible but wrong; diagnostics reference the wrong flags.** Stale database. Confirm the root symlink still resolves (``readlink -f compile_commands.json``) and rebuild if the objdir it names is gone.

**Do not pass a relative ``--compile-commands-dir``.** It is resolved against nvim's working directory, and per clangd's own documentation an invalid path silently falls back to searching parent directories — so a typo looks like it works. Prefer ``.clangd``; note that the flag overrides ``CompilationDatabase`` if both are set.

7. Python completion for the compiled ``psi4.core`` module
----------------------------------------------------------

clangd does not cover the pybind11 extension: ``psi4.core`` is a shared object, so a Python language server sees no methods, signatures, or docstrings for it. The fix is a ``.pyi`` interface generated from the built extension.

Build the stubs as part of the build (preferred)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m pip install pybind11-stubgen basedpyright
   cmake -S . -B objdir -DENABLE_PYTHON_STUBS=ON
   cmake --build objdir

``ENABLE_PYTHON_STUBS`` adds a ``psi4-core-stubs`` target that depends on ``psi4-core`` and writes ``core.pyi`` beside the staged extension in ``<objdir>/stage/lib/psi4/``. Generation takes well under a second and is deterministic, so it re-runs on every build and the interface cannot drift from the compiled module. Because the stub lands inside the staged package, any tool that already resolves that package picks it up with no further configuration, and the existing install rule carries it to the install prefix. Configure fails immediately with a ``pip install`` hint if ``pybind11-stubgen`` is missing from the interpreter CMake selected as ``Python_EXECUTABLE``.

To refresh only the stubs after a build:

.. code:: bash

   cmake --build objdir --target psi4-core-stubs

Or run the generator directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a tree configured without the option:

.. code:: bash

   python devtools/scripts/generate_psi4_core_stubs.py --stage-lib "<objdir>/stage/lib"

Replace ``<objdir>`` with the CMake build directory. The script writes ``.typings/psi4/core.pyi`` by default; pass ``--output-dir`` to place the stub elsewhere.

What the generator covers
~~~~~~~~~~~~~~~~~~~~~~~~~

It deliberately loads only the extension, so generation does not depend on all of Psi4's Python driver dependencies importing successfully. It also adds a typed overlay for the global and ``Wavefunction`` QCVariable helpers (``variable``, ``set_variable``, ``has_variable``, ``del_variable``, and ``variables``) that ``python_helpers.py`` attaches dynamically after import. The overlay also covers the APIs installed by ``molutil.py``, ``writer.py``, ``cubeprop.py``, ``fisapt_proc.py``, and the SCF driver, as well as the NumPy helpers attached to ``Matrix``, ``Vector``, ``Dimension``, and ``CIVector``. The inventory currently covers all 96 unique ``psi4.core`` attributes that local Python code installs dynamically. Some existing bindings expose raw C++ names such as ``psi::Matrix`` in their runtime signatures; the generator replaces those particular annotations with an unknown type while preserving the rest of each signature and its docstring.

Point the Python language server at it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the CMake target, the stub is already inside the staged package, so the staged path is the only thing basedpyright needs:

.. code:: json

   {
     "extraPaths": ["<objdir>/stage/lib"],
     "typeCheckingMode": "basic"
   }

Add ``"stubPath": ".typings"`` as well if you generated stubs with the standalone script and its default output directory.

Enable ``basedpyright`` in Neovim alongside ``clangd``. Restart the Python LSP after a rebuild (``:LspRestart``) so it re-reads the regenerated interface.
