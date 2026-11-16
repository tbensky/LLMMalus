"""The apparatus: knobs in, one signal out.

This is the only module that knows what the experiment physically is. The
tools the model uses (tools.py) talk to the `Apparatus` interface below and
nothing else, so swapping the simulation for the real bench means writing
one new subclass, not touching the tools, the conversation loop or the
scoring.

The bench:

    bulb (intensity controller 0..100)
      -> polarizer 1 (angle t1, rotates freely)
      -> polarizer 2 (angle t2, rotates freely)
      -> photodetector (voltage)

    I = I0 * cos^2(t2 - t1)        Malus's law

The model is never told any of this. It sees three knobs named A, B and C:
A goes from 0 to 100, B and C rotate freely and are set in degrees.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnobSpec:
    """One knob, as the model is allowed to know it."""

    name: str
    description: str
    minimum: float | None = None   # None: no limit (a freely rotating knob)
    maximum: float | None = None
    units: str = ""


class ApparatusError(RuntimeError):
    """A request the apparatus cannot carry out (bad knob, out of range...).

    The message is shown to the model, so keep it plain and physical.
    """


class Apparatus(ABC):
    """What every apparatus, simulated or real, must provide.

    A hardware implementation needs exactly these: move a knob to a
    setting, report where the knobs are, and read the detector.
    """

    signal_name: str = "voltage"
    signal_units: str = "V"

    @property
    @abstractmethod
    def knobs(self) -> tuple[KnobSpec, ...]:
        """The knobs the model can turn, in the order they are described."""

    @abstractmethod
    def set_knob(self, name: str, value: float) -> float:
        """Move one knob and return where it actually ended up.

        Raise ApparatusError for an unknown knob or a value it cannot take.
        A real knob may land slightly off the request (motor steps,
        backlash): return the actual position.
        """

    @abstractmethod
    def get_knobs(self) -> dict[str, float]:
        """Current position of every knob."""

    @abstractmethod
    def read_signal(self) -> float:
        """One reading of the detector, in signal_units."""

    def knob_spec(self, name: str) -> KnobSpec:
        for spec in self.knobs:
            if spec.name == name:
                return spec
        names = ", ".join(spec.name for spec in self.knobs)
        raise ApparatusError(f"There is no knob {name!r}. The knobs are {names}.")

    def close(self) -> None:
        """Release hardware (turn the bulb off, park motors). No-op by default."""


# The three knobs, described as plainly as the task prompt describes them.
# Which physical part each one drives is the apparatus's business only.
KNOB_A = KnobSpec("A", "Turns from 0 to 100.", minimum=0.0, maximum=100.0)
KNOB_B = KnobSpec("B", "Rotates continuously; set it to any angle in degrees.", units="degrees")
KNOB_C = KnobSpec("C", "Rotates continuously; set it to any angle in degrees.", units="degrees")


@dataclass
class SimulationConfig:
    """How realistic the simulated bench is. Defaults are a clean bench.

    full_scale_volts   detector voltage with the bulb at 100 and the two
                       polarizers aligned (I0 at full brightness)
    dark_volts         detector offset with no light
    noise_volts        standard deviation of each reading
    extinction         fraction of light that still leaks through when the
                       polarizers are crossed (0 = perfect polarizers)
    angle_offset_deg   hidden angle between knob B's zero mark and knob C's
                       (real polarizer zero marks are never aligned)
    bulb_exponent      detector light ~ (A/100)**bulb_exponent; 1 is a
                       linear intensity controller, a real dimmed bulb is not
    saturation_volts   detector clips at this voltage (None: never)
    """

    full_scale_volts: float = 5.0
    dark_volts: float = 0.0
    noise_volts: float = 0.005
    extinction: float = 0.0
    angle_offset_deg: float = 0.0
    bulb_exponent: float = 1.0
    saturation_volts: float | None = None

    @classmethod
    def randomized(cls, rng: random.Random, **fixed: float) -> "SimulationConfig":
        """A bench with unknown calibration, so no constant can be memorized.

        The law stays the same; only the constants a student would have to
        measure change from run to run: the full-scale voltage and the dark
        offset. The angle offset is left alone (0 unless given): real
        polarizers carry a hash mark on the transmission axis, and a student
        lines those up as the zero, so the dials agree on "aligned".
        """
        config = cls(
            full_scale_volts=round(rng.uniform(2.0, 9.0), 3),
            dark_volts=round(rng.uniform(0.0, 0.05), 4),
        )
        for key, value in fixed.items():
            setattr(config, key, value)
        return config


class SimulatedPolarizerApparatus(Apparatus):
    """Bulb, two polarizers and a photodetector, obeying Malus's law."""

    def __init__(self, config: SimulationConfig | None = None, seed: int | None = None):
        self.config = config or SimulationConfig()
        self._rng = random.Random(seed)
        self._settings: dict[str, float] = {"A": 0.0, "B": 0.0, "C": 0.0}

    @property
    def knobs(self) -> tuple[KnobSpec, ...]:
        return (KNOB_A, KNOB_B, KNOB_C)

    def set_knob(self, name: str, value: float) -> float:
        spec = self.knob_spec(name)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ApparatusError(f"Knob {name} needs a number, not {value!r}.") from None
        if not math.isfinite(value):
            raise ApparatusError(f"Knob {name} needs a finite number.")
        if spec.minimum is not None and value < spec.minimum:
            raise ApparatusError(f"Knob {name} cannot go below {spec.minimum:g}.")
        if spec.maximum is not None and value > spec.maximum:
            raise ApparatusError(f"Knob {name} cannot go above {spec.maximum:g}.")
        self._settings[name] = value
        return value

    def get_knobs(self) -> dict[str, float]:
        return dict(self._settings)

    def true_signal(self, settings: dict[str, float] | None = None) -> float:
        """The noiseless detector voltage (simulation only)."""
        s = settings or self._settings
        c = self.config
        light = (s["A"] / 100.0) ** c.bulb_exponent
        delta = math.radians(s["C"] - s["B"] + c.angle_offset_deg)
        transmitted = c.extinction + (1.0 - c.extinction) * math.cos(delta) ** 2
        volts = c.dark_volts + c.full_scale_volts * light * transmitted
        if c.saturation_volts is not None:
            volts = min(volts, c.saturation_volts)
        return volts

    def read_signal(self) -> float:
        volts = self.true_signal() + self._rng.gauss(0.0, self.config.noise_volts)
        if self.config.saturation_volts is not None:
            volts = min(volts, self.config.saturation_volts)
        return volts


class HardwareApparatus(Apparatus):
    """Placeholder for the real bench. Fill in the four methods below.

    Suggested mapping (the model never sees it):
        A -> bulb intensity controller (0 = off, 100 = full)
        B -> rotation stage under polarizer 1, degrees
        C -> rotation stage under polarizer 2, degrees
        signal -> photodetector voltage from the ADC

    Keep set_knob blocking until the knob has settled, and return the
    position the hardware actually reached. Raise ApparatusError with a
    plain message for anything the model asked for that cannot be done;
    it is shown to the model as the tool result.
    """

    def __init__(self, **connection: object):
        self.connection = connection
        raise NotImplementedError(
            "HardwareApparatus is a stub: implement set_knob, get_knobs and "
            "read_signal for the real bulb controller, rotation stages and ADC."
        )

    @property
    def knobs(self) -> tuple[KnobSpec, ...]:
        return (KNOB_A, KNOB_B, KNOB_C)

    def set_knob(self, name: str, value: float) -> float:  # pragma: no cover - stub
        raise NotImplementedError

    def get_knobs(self) -> dict[str, float]:  # pragma: no cover - stub
        raise NotImplementedError

    def read_signal(self) -> float:  # pragma: no cover - stub
        raise NotImplementedError
