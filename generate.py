"""Classify surfaces from identification polygons."""

import sys


def raw_pairings(num_edges):
    """Generate pairings representing each polygon.

    Identification polygons with n sides correspond to
    tagged partitions of {1,2,...,n} into blocks of size 2
    by listing the indices of identified edges.

    For example, the polygon with symbol 'aaCbcB' corresponds to
    ((0, 1), (3, 5), (2, 4)) since 'a' occurs at indices 0 and 1,
    'b' occurs at indices 3 and 5, and 'c' occurs at
    indices 2 and 4.

    Furthermore, to account for the fact that edges can be identified
    as either parallel or anti-parallel, we tag each block with either
    True for parallel or false for anti-parallel, thus 'aaCbcB' actually
    is represented as ((0, 1, True), (3, 5, False), (2, 4, False)).

    It then suffices to generate all these tagged partitions. We do so
    in a sorted manner as to only consider unordered partitions
    (corresponding to permutations of the polygon edge labels).
    """
    def subdivide(elems):
        if not elems:
            yield []
            return
        first, *rest = elems
        for second in rest:
            for sub in subdivide([i for i in rest if i != second]):
                yield [(first, second, True), *sub]
                yield [(first, second, False), *sub]

    for sub in subdivide(range(num_edges)):
        yield tuple(sub)


def pairings(num_edges):
    """Return all pairings as in raw_pairings, but removing rotations.

    For a given pairing like

    ((0, 1, True), (2, 3, True), (4, 5, False))

    we know this will be equivalent to

    ((0, 5, False), (1, 2, True), (3, 4, True))

    since it just corresponds to a rotation of the polygon by one edge i,e,
    each index is incremented by 1 modulo 6. We then can filter all these
    redundant rotations from the pairings.
    """
    def rotate(pairing, amount):
        """Rotate the pairing by amount edges."""
        rot = []
        for fst, snd, par in pairing:
            fst_rot = (fst + amount) % num_edges
            snd_rot = (snd + amount) % num_edges
            if fst_rot < snd_rot:
                rot.append((fst_rot, snd_rot, par))
            else:
                rot.append((snd_rot, fst_rot, par))
        return tuple(sorted(rot))

    bad = set()
    for pairing in raw_pairings(num_edges):
        if pairing not in bad:
            yield pairing
            for amount in range(num_edges):
                bad.add(rotate(pairing, amount))


def euler_characteristic(pairing):
    """Determine the Euler characteristic of the corresponding surface.

    The actual algorithm here is a bit hairy, but the general
    idea is just to start at a vertex, use the identification of edges
    to chase arrows and see what other vertices it joins with in the quotient,
    then repeat until all vertices have been partitioned. The
    Euler characteristic is then v - e + f = v - (n / 2) + 1.
    """
    poly_edges = len(pairing) * 2
    is_forward = [None] * poly_edges
    for fst, snd, par in pairing:
        is_forward[fst] = True
        is_forward[snd] = par

    other_idx = {x: y for x, y, _ in pairing}
    other_idx |= {y: x for x, y, _ in pairing}

    def next_vertex(idx, moving_right):
        """Chase the arrow we are currently at to get the next vertex."""
        curr_idx = ((idx + 1) % poly_edges if moving_right
                    else (idx - 1) % poly_edges)
        at_tip = is_forward[curr_idx]
        if moving_right:
            at_tip = not at_tip
        next_idx = other_idx[curr_idx]
        next_moving_right = is_forward[next_idx]
        if not at_tip:
            next_moving_right = not next_moving_right
        return (next_idx, next_moving_right)

    # Repeatedly get the next vertex until a cycle is found, then repeat.
    num_vertices = 1
    unvisited = set(range(0, poly_edges))
    curr = (unvisited.pop(), True)
    while unvisited:
        curr = next_vertex(*curr)
        actual_idx = curr[0] if curr[1] else (curr[0] - 1) % poly_edges
        if actual_idx in unvisited:
            unvisited.remove(actual_idx)
        else:
            curr = (unvisited.pop(), True)
            num_vertices += 1

    # Chi = v - e + f
    return num_vertices - (poly_edges // 2) + 1


def is_orientable(pairing):
    """Determine if pairing is orientable.

    Does a simple check to see if any edge pair is
    identified in a parallel way i,e, if the surface contains
    a Moebius strip.
    """
    return all(map(lambda p: not(p[2]), pairing))


def genus(pairing):
    """Compute the genus of the corresponding surface."""
    euler = euler_characteristic(pairing)
    gen = 2 - euler
    if is_orientable(pairing):
        gen //= 2
    return gen


def classification(pairing):
    """Apply the classification of surfaces."""
    gen = genus(pairing)
    if gen == 0:
        return "S^2"
    if is_orientable(pairing):
        return ("T^2\\#" * gen)[:-2]
    return ("\\mathbb{R}P^2\\#" * gen)[:-2]


def pairing_to_symbol(pairing):
    """Convert our internal representation of a polygon into a symbol.

    Symbols are displayed using capital letters to denote inverses.
    """
    out = [None] * (len(pairing) * 2)
    curr_char = 'a'
    for fst, snd, par in pairing:
        out[fst] = curr_char
        out[snd] = curr_char if par else curr_char.upper()
        curr_char = chr(ord(curr_char) + 1)
    return "".join(out)


def symbol_to_pairing(symbol):
    """Convert a symbol to our internal representation."""
    pairs = {}
    for idx, sym in enumerate(symbol):
        up_sym = sym.upper()
        if up_sym in pairs:
            other_idx = pairs[up_sym]
            pairs[up_sym] = (other_idx, idx,
                             symbol[other_idx] == symbol[idx])
        else:
            pairs[up_sym] = idx
    return tuple(sorted(pairs.values()))


def classify_all_polygons(num_edges):
    """Determine all surfaces producable from polygons with n sides.

    Returns a dictionary of surfaces to the formal strings
    corresponding to those surfaces.
    """
    results = {}
    for pair in pairings(num_edges):
        sym = pairing_to_symbol(pair)
        classify = classification(pair)
        if classify in results:
            results[classify].append(sym)
        else:
            results[classify] = [sym]
    return results


def display_classification(num_edges, show_syms):
    """Display all surfaces producable from polygons with n sides.

    Displays the results of classify_all_polygons in a sorted way,
    with each space an the corresponding symbols listd.
    """
    results = classify_all_polygons(num_edges)
    for res in sorted(results):
        print(f"{res}", end="")
        if show_syms:
            print(": ", end="")
            for sym in sorted(results[res]):
                print(sym, end=" ")
        print("")


if __name__ == '__main__':
    n = int(sys.argv[1])
    show = len(sys.argv) == 3 and sys.argv[2] == "show"
    display_classification(n, show)
