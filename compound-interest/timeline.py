"""Shared timing for the compound-interest explainer (used by render.py and music.py)."""

FPS = 30
DUR = 20.0

START = 1000          # dollars
RATE = 0.10           # per year
T0 = 3.0              # chart starts drawing
TY = 1.35             # seconds per simulated year (year 10 lands at 16.5 s)

HERO_DROP = 0.2       # coin drops in
TIP_T = 2.5           # coin tips from standing to flat
TIP_D = 0.7
CANOPY_Y = 4.3        # simulated year when the tree sprouts

CAPTIONS = [
    (0.7, 3.0, "Start with $1,000."),
    (3.0, 7.0, "Each year it earns 10% interest."),
    (7.0, 11.5, "Next year, the interest earns interest too."),
    (11.5, 16.5, "Small gains snowball into a tree."),
    (16.5, 19.9, "$1,000 becomes $2,594. Time does the work."),
]


def year_at(t):
    return max(0.0, min(10.0, (t - T0) / TY))


def tick_time(y):
    return T0 + TY * y


def balance(y):
    return START * (1 + RATE) ** y


def simple(y):
    return START * (1 + RATE * y)


def coins_at_year(y):
    return round(1 + 10 * ((1 + RATE) ** y - 1) / ((1 + RATE) ** 10 - 1))


def coin_times():
    """Arrival time of coins 1..10 (coin 0 is the hero coin)."""
    out = []
    prev = 1
    for y in range(1, 11):
        n = coins_at_year(y)
        for j in range(n - prev):
            out.append(tick_time(y) + 0.14 * j)
        prev = n
    return out


CANOPY_T = tick_time(CANOPY_Y)
END_T = tick_time(10)
