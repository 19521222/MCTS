"""
This file is a part of My-PyChess application.
In this file, we define heuristic constants required for the python chess
engine.
"""
occupancies = {
    0: [
        True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True
    ],
    1: [
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True
    ],
    2: [
        True, True, True, True, True, True, True, True,
        True, True, True, True, True, True, True, True,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
        False, False, False, False, False, False, False, False,
    ]
}


mvv_lva =(
    (105, 205, 305, 405, 505, 605),
	(104, 204, 304, 404, 504, 604),
	(103, 203, 303, 403, 503, 603),
	(102, 202, 302, 402, 502, 602),
	(101, 201, 301, 401, 501, 601),
	(100, 200, 300, 400, 500, 600)
)

pawnEvalBlack = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    (2.0, 2.0, 3.0, 5.0, 5.0, 3.0, 2.0, 2.0),
    (0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5),
    (0.0, 0.0, 0.5, 2.0, 2.0, 0.5, 0.0, 0.0),
    (0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5),
    (0.5, 1.0, 0.5, -2.0, -2.0, 0.5, 1.0, 0.5),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)

pawnEvalWhite = tuple(reversed(pawnEvalBlack))

knightEval = (
    (-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0),
    (-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0),
    (-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0),
    (-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0),
    (-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0),
    (-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0),
    (-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0),
    (-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0),
)

bishopEvalBlack = (
    (-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0),
    (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0),
    (-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0),
    (-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0),
    (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0),
    (-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0),
    (-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0),
)

bishopEvalWhite = tuple(reversed(bishopEvalBlack))

rookEvalBlack = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5),
    (0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0),
)

rookEvalWhite = tuple(reversed(rookEvalBlack))

queenEval = (
    (-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0),
    (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0),
    (-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5),
    (0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5),
    (-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0),
    (-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0),
    (-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0),
)

kingEvalBlack = (
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0),
    (-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0),
    (-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0),
    (2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0),
    (2.0, 3.0, 3.0, 0.0, 0.0, 1.0, 3.0, 2.0),
)

kingEvalWhite = tuple(reversed(kingEvalBlack))