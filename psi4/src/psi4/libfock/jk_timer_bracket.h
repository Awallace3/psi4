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

#ifndef JK_TIMER_BRACKET_H
#define JK_TIMER_BRACKET_H

#include "psi4/libqt/qt.h"

namespace psi {

/*
 * A timer bracket that survives an exception.
 *
 * ``timer_on`` records the label on a stack and ``timer_off`` pops it, so a
 * label left on by an escaping exception is not merely a missing time: the next
 * ``timer_on`` for it throws "is already on" from inside the timer's OpenMP
 * lock, without releasing it, and the ``timer_done`` at interpreter shutdown
 * then spins on that lock forever. JK subclasses legitimately throw out of
 * ``compute_JK`` and out of ``preiterations`` to refuse a request, so every
 * bracket around them has to unwind.
 *
 * ``label`` is borrowed, not copied, and must outlive the bracket; every use in
 * libfock passes a string literal.
 */
class JKTimerBracket {
   public:
    explicit JKTimerBracket(const char* label) : label_(label) { timer_on(label_); }
    ~JKTimerBracket() {
        // Called while an exception may be in flight, so it must not throw.
        try {
            timer_off(label_);
        } catch (...) {
        }
    }
    JKTimerBracket(const JKTimerBracket&) = delete;
    JKTimerBracket& operator=(const JKTimerBracket&) = delete;

   private:
    const char* label_;
};

}  // namespace psi

#endif
