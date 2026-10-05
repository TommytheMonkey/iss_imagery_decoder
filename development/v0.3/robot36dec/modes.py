"""Mode constants. Robot 36 is the only decoded mode; the VIS table is for identification/reporting only."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Robot36:
    name: str = "Robot 36"
    vis: int = 8
    width: int = 320
    height: int = 240
    line_ms: float = 150.0
    sync_ms: float = 9.0          # 1200 Hz
    porch_ms: float = 3.0         # 1500 Hz
    y_ms: float = 88.0            # 320 px
    sep_ms: float = 4.5           # 1500 Hz even / 2300 Hz odd
    porch2_ms: float = 1.5        # 1900 Hz
    c_ms: float = 44.0            # 320 px, R-Y even / B-Y odd
    f_sync: float = 1200.0
    f_black: float = 1500.0
    f_white: float = 2300.0
    @property
    def y_start_ms(self): return self.sync_ms + self.porch_ms                       # 12.0
    @property
    def sep_start_ms(self): return self.y_start_ms + self.y_ms                       # 100.0
    @property
    def c_start_ms(self): return self.sep_start_ms + self.sep_ms + self.porch2_ms    # 106.0

ROBOT36 = Robot36()

VIS_TABLE = {8: "Robot 36", 12: "Robot 72", 4: "Robot 24", 0: "Robot 12", 40: "Martin 2", 44: "Martin 1",
             56: "Scottie 2", 60: "Scottie 1", 76: "Scottie DX", 95: "PD120", 96: "PD180", 99: "PD90",
             98: "PD160", 97: "PD240", 94: "PD290", 93: "PD50", 55: "Wraase SC2-180"}

# VIS header layout (ms): (frequency, duration)
VIS_LEADER = [(1900.0, 300.0), (1200.0, 10.0), (1900.0, 300.0), (1200.0, 30.0)]
VIS_BIT_MS = 30.0
VIS_TOTAL_MS = 300 + 10 + 300 + 30 + 8 * 30 + 30     # 910 ms from first leader start to end of stop bit
