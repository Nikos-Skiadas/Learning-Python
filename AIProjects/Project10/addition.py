# addition.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
Run python autograder.py
"""


from __future__ import annotations

import typing


class SupportsAdd(typing.Protocol):

    def __add__(self, other: typing.Self, /) -> typing.Self:
        ...


def add(a: float, b: float) -> float:
    """Add two real numbers.

    Args:
        a: Lefthand operand.
        b: Righthand operand.

    Returns:
        a + b
    """
    return a + b
