"""Shared timing for the avocado toast video: 128 BPM, 32 beats = exactly 15.0 s.
render.py (picture) and music.py (sound) both import this so the cuts, the number bounces and the sound
effects land on the same beats. All values are in beats."""

BPM = 128
B = 60 / BPM            # seconds per beat (0.46875)
TOTAL_BEATS = 32
FPS = 30

S = [4, 10, 16, 22]     # start of step 1..4 (each step lasts 6 beats)
OUT = 28                # outro start

# intro
T_PLATE_IN = 0.3
T_TITLE1, T_TITLE2, T_PILL = -0.1, 0.35, 1.2
T_TITLE_OUT = 3.3

# step 1: toast it
T_BREAD_IN = S[0] + 0.5
T_TOAST_A, T_TOAST_B = S[0] + 1.4, S[0] + 3.0
T_DING = S[0] + 3.1
T_PLATE_OUT1 = S[0] + 5.1

# step 2: mash it
T_AVO_IN = S[1] + 0.4
T_BOWL_IN = S[1] + 0.6
T_PIT = S[1] + 1.8
T_FLIP = S[1] + 2.8
T_PLOP = S[1] + 3.5
T_FORK_IN = S[1] + 3.1
T_MASH_A, T_MASH_B = S[1] + 3.7, S[1] + 5.3
T_FORK_OUT = S[1] + 5.2
T_BOWL_UP = S[1] + 5.2

# step 3: spread it
T_PLATE_IN2 = S[2]
T_TILT = S[2] + 0.6
T_DOLLOP = [S[2] + 1.0, S[2] + 1.2, S[2] + 1.4]
T_SPREAD_A, T_SPREAD_B = S[2] + 2.1, S[2] + 4.3
T_SHINE = S[2] + 4.5

# step 4: top it
T_EGG = S[3] + 0.6
T_CHILI = S[3] + 1.7
T_PEPPER = S[3] + 2.8
T_LEAF = S[3] + 3.6
T_LEMON = S[3] + 4.3

# outro
T_CONFETTI = OUT + 0.1
T_CTA = OUT + 1.6
