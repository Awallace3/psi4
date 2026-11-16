/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2026 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

// Minimal hosting globals for the standalone libpsio round-trip executable.
// libpsio/libpsi4util/libciomr are not standalone archives -- they reference a
// few runtime globals normally defined in core.cc. Supply just enough to link
// without dragging in libdpd or the python bindings TU.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "psi4/libpsi4util/PsiOutStream.h"
#ifdef _WIN32
// A standalone host supplies only one DPD member, not a complete DLL export.
#pragma push_macro("PSI_API")
#undef PSI_API
#define PSI_API
#endif
#include "psi4/libdpd/dpd.h"
#ifdef _WIN32
#pragma pop_macro("PSI_API")
#endif

namespace psi {

std::string restart_id;
char *psi_file_prefix = strdup("psio_test");
std::shared_ptr<PsiOutStream> outfile;

// libciomr's DSYEV_ascending lands in the same unity TU as helpers libpsio
// pulls in; the standalone link references its LAPACK wrapper though the test
// never diagonalizes. A no-op leaf keeps the test self-contained. The signature
// matches the real declaration in libqt/qt.h (returns int).
int C_DSYEV(char, char, int, double *, int, double *, double *, int) { return 0; }

// Exercise real non-DPD hosting, as in SCF/DIIS. Use the actual declaration
// rather than an incompatible fake DPD class. Any attempted cache call must
// fail the round-trip test, including in ordinary non-sanitized builds.
DPD *global_dpd_ = nullptr;
void DPD::file4_cache_del_filenum(std::size_t) { std::abort(); }

}  // namespace psi
