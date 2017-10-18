#!/usr/bin/env python3

import structure

if __name__ == '__main__':
    import energy

    rb = energy.Fourier((1.0, 2.0, 3))
    a1 = structure.Atom(1, 2, 3)
    a2 = structure.Atom(4, 5, 6)
    a3 = structure.Atom(6, 6, 8)
    a4 = structure.Atom(9, 8, 10)

    print(rb.E(a1, a2, a3, a4))
