from typing import TypedDict
# TODO
# Make Conifg Dict Types ?
# Extend NerveBlock configuration accordingly
# Plots to visualize things


class MonophasicPulseConfig(TypedDict):
    amp: float
    pw: float
    delay: float
    offset: float


class SinusoidPulseConfig(TypedDict):
    amp: float
    freq: float
    delay: float
    offset: float


class BiphasicPulseConfig(TypedDict):
    amp1: float
    pw1: float
    amp2: float
    pw2: float
    delay: float
    offset: float

