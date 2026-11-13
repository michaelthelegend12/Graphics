"""Shared timeline for the CPU cycle explainer: render.py (picture) and sound.py (audio)
both read it so the two stay in sync."""

FPS = 30
DUR = 15.0

# (name, start, end) of each stage of the cycle
STAGES = [("FETCH", 1.2, 4.8), ("DECODE", 4.8, 8.4), ("EXECUTE", 8.4, 12.6)]
LOOP_T = 12.6  # after this the cycle "repeats" faster and faster


def _steps():
    out, t, dt = [], LOOP_T, 0.5
    while t < 14.8:
        out.append(t)
        dt = max(0.13, dt * 0.86)
        t += dt
    return out


STEPS = _steps()

# data packets: (id, trace, label, t_start, t_end)
PACKETS = [
    ("addr", "addr", "0x0040", 1.4, 2.3),         # PC -> memory
    ("instr", "data", "ADD R1,R2", 2.9, 3.9),     # memory -> instruction register
    ("ir2dec", "ir2dec", "ADD R1,R2", 5.0, 5.7),  # IR -> decoder
    ("op", "dec2alu", "ADD", 7.0, 7.6),           # decoder -> ALU (opcode)
    ("rq1", "dec2reg", "R1", 7.0, 7.9),           # decoder -> registers (operands)
    ("rq2", "dec2reg", "R2", 7.25, 8.15),
    ("v1", "reg2alu_a", "5", 8.5, 9.0),           # registers -> ALU (values)
    ("v2", "reg2alu_b", "3", 8.6, 9.1),
    ("res", "alu2res", "8", 10.5, 11.0),          # ALU -> result
    ("wb", "wb", "8", 11.2, 12.2),                # write-back to registers
]

# when each block of the chip is lit: name -> [(start, end)]
BLOCKS = {
    "pc": [(1.0, 1.8), (12.6, 13.8)],
    "ram": [(2.2, 3.0)],
    "ir": [(3.9, 5.0)],
    "dec": [(5.7, 7.6)],
    "reg": [(7.9, 8.9), (12.2, 13.0)],
    "alu": [(7.6, 8.0), (9.1, 10.5)],
    "res": [(11.0, 12.2)],
}

# trace-log lines typed into the console panel: (time, text)
CONSOLE = [
    (1.4, "> FETCH  PC=0x0040"),
    (3.9, "> IR <- ADD R1,R2"),
    (5.7, "> DECODE op=ADD"),
    (6.3, "  src=R1,R2"),
    (9.1, "> EXEC   5 + 3"),
    (11.0, "> RESULT = 8"),
    (12.2, "> WRITE  R1 <- 8"),
    (12.9, "> PC <- 0x0044"),
]

TITLE_1 = "HOW A CPU RUNS ONE INSTRUCTION"
TITLE_2 = "AND THEN IT REPEATS. BILLIONS OF TIMES A SECOND."
