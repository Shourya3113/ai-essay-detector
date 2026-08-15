"""Tiny character trigram language model used as a *measurement* fallback.

This is not a judge. It assigns log-probabilities under a fixed character
trigram estimated from public-domain English. Tests and CPU-only first
runs can compute sentence surprisal without downloading DistilGPT-2.
Live detection prefers DistilGPT-2 when that model is available.
"""

from __future__ import annotations

import math
from collections import defaultdict

# Public-domain English (Gettysburg Address + a Franklin autobiography
# fragment + a short Thoreau paragraph). Used only to estimate trigrams.
_REFERENCE = """
four score and seven years ago our fathers brought forth on this continent
a new nation conceived in liberty and dedicated to the proposition that all
men are created equal. now we are engaged in a great civil war testing
whether that nation or any nation so conceived and so dedicated can long
endure. we are met on a great battlefield of that war. we have come to
dedicate a portion of that field as a final resting place for those who here
gave their lives that that nation might live. the world will little note nor
long remember what we say here but it can never forget what they did here.
it is for us the living rather to be dedicated here to the unfinished work
which they who fought here have thus far so nobly advanced.
i was born in boston in new england and served an apprenticeship in my
brother's printing house. i disliked the trade but I learned to write by
imitating the spectator papers, taking notes, laying them aside, and then
trying to recover the sense in my own words. that exercise taught me method
and a little elegance, though my first drafts were always rougher than I
wished and I struck out many words after I had set them down.
i went to the woods because i wished to live deliberately, to front only the
essential facts of life, and see if i could not learn what it had to teach,
and not, when i came to die, discover that i had not lived. i wanted to live
deep and suck out all the marrow of life, to live so sturdily and spartan
like as to put to rout all that was not life.
"""


def _train(text: str) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    padded = "  " + text.lower()
    for i in range(2, len(padded)):
        ctx, ch = padded[i - 2 : i], padded[i]
        counts[ctx][ch] += 1
    probs: dict[str, dict[str, float]] = {}
    vocab = 96
    for ctx, dist in counts.items():
        total = sum(dist.values())
        probs[ctx] = {ch: (n + 0.1) / (total + 0.1 * vocab) for ch, n in dist.items()}
        probs[ctx]["__unk__"] = 0.1 / (total + 0.1 * vocab)
    return probs


_PROBS = _train(_REFERENCE)
_UNIFORM = 1.0 / 96.0


def char_nll(text: str) -> float:
    """Average negative log2-probability per character under the trigram."""
    if not text:
        return 0.0
    padded = "  " + text.lower()
    total = 0.0
    n = 0
    for i in range(2, len(padded)):
        ctx, ch = padded[i - 2 : i], padded[i]
        dist = _PROBS.get(ctx)
        p = dist.get(ch, dist.get("__unk__", _UNIFORM)) if dist else _UNIFORM
        total += -math.log(max(p, 1e-12), 2)
        n += 1
    return total / max(n, 1)
