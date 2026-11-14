/*
 * Optional compiler no-alias hint. SPDX-License-Identifier: LGPL-3.0-only
 */
#ifndef PSI4_RESTRICT_H
#define PSI4_RESTRICT_H

// PSI_NO_RESTRICT permits optimized reference builds without no-alias promises.
// Restrict is not standard C++; correctness must not depend on this hint.
#if defined(PSI_NO_RESTRICT)
#define PSI_RESTRICT
#elif defined(_MSC_VER)
#define PSI_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define PSI_RESTRICT __restrict__
#else
#define PSI_RESTRICT
#endif

#endif
