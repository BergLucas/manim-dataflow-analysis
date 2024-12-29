from manim.mobject.mobject import Mobject
from manim import config


def scale_mobject(mobject: Mobject, max_x: float, max_y: float) -> None:
    if mobject.width >= mobject.height:
        if max_y / max_x >= mobject.height / mobject.width:
            mobject.scale(max_x / mobject.width)
        else:
            mobject.scale(max_y / mobject.height)
    elif max_x / max_y >= mobject.width / mobject.height:
        mobject.scale(max_y / mobject.height)
    else:
        mobject.scale(max_x / mobject.width)


def fw(scale_w: float):
    return scale_w * config.frame_width


def fh(scale_y: float):
    return scale_y * config.frame_height
