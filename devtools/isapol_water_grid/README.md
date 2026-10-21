# Offline printed-grid utilities

`rounding.py` is the accepted portable e15.8 closed-interval implementation;
`grid.py` separates the read-only token/site parsing from the historical
fixed-geometry `admit_c.py` engine receipt/authentication procedure.
Neither invokes an engine, discovers a runtime, regenerates a grid, supplies
a map or assumes a scratch location. A parsed file is **not** admitted science.

Run `python -B -m unittest -v test_grid` from this directory, with one numerical
thread, AS 8 GiB and core 0. This is a manual/offline developer entrypoint.

Any new comparison must declare C-SLATER (`RELATIVE-RADII SLATER`) and the
frozen 230-entry angular map, and label density model, numerical solve and
refinement selection separately. Never fit a map to the compared fields.
The old run006, C-default arm, 506 weight-interval violations and all historical
HOLDs remain unchanged. The historical `control.py`, `native_grid.py`,
`admit_c.py`, bootstrap runners and fixed engine input decks are retained as
read-only evidence in the source worktree, not copied into ordinary CI.
