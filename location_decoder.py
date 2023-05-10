import multiprocessing

import numpy as np
from sympy import *
import math
from scipy import optimize as op


class Location:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z


base1 = Location(1, 0, 0, 0)
base2 = Location(2, 500, 0, 66)
base3 = Location(3, 500, 483, 0)
base4 = Location(4, 0, 483, 66)
bases = [base1, base2, base3, base4]


def decode_location_scipy(dis1, dis2, dis3, dis4) -> list:
    def f0(K):
        return [
            # (K[0] - base1.x) ** 2 + (K[1] - base1.y) ** 2 + (K[2] - base1.z) ** 2 - dis1 ** 2,
            (K[0] - base2.x) ** 2 + (K[1] - base2.y) ** 2 + (K[2] - base2.z) ** 2 - dis2 ** 2,
            (K[0] - base3.x) ** 2 + (K[1] - base3.y) ** 2 + (K[2] - base3.z) ** 2 - dis3 ** 2,
            (K[0] - base4.x) ** 2 + (K[1] - base4.y) ** 2 + (K[2] - base4.z) ** 2 - dis4 ** 2,
        ]

    def f1(K):
        return [
            (K[0] - base1.x) ** 2 + (K[1] - base1.y) ** 2 + (K[2] - base1.z) ** 2 - dis1 ** 2,
            # (K[0] - base2.x) ** 2 + (K[1] - base2.y) ** 2 + (K[2] - base2.z) ** 2 - dis2 ** 2,
            (K[0] - base3.x) ** 2 + (K[1] - base3.y) ** 2 + (K[2] - base3.z) ** 2 - dis3 ** 2,
            (K[0] - base4.x) ** 2 + (K[1] - base4.y) ** 2 + (K[2] - base4.z) ** 2 - dis4 ** 2,
        ]

    def f2(K):
        return [
            (K[0] - base1.x) ** 2 + (K[1] - base1.y) ** 2 + (K[2] - base1.z) ** 2 - dis1 ** 2,
            (K[0] - base2.x) ** 2 + (K[1] - base2.y) ** 2 + (K[2] - base2.z) ** 2 - dis2 ** 2,
            # (K[0] - base3.x) ** 2 + (K[1] - base3.y) ** 2 + (K[2] - base3.z) ** 2 - dis3 ** 2,
            (K[0] - base4.x) ** 2 + (K[1] - base4.y) ** 2 + (K[2] - base4.z) ** 2 - dis4 ** 2,
        ]

    def f3(K):
        return [
            (K[0] - base1.x) ** 2 + (K[1] - base1.y) ** 2 + (K[2] - base1.z) ** 2 - dis1 ** 2,
            (K[0] - base2.x) ** 2 + (K[1] - base2.y) ** 2 + (K[2] - base2.z) ** 2 - dis2 ** 2,
            (K[0] - base3.x) ** 2 + (K[1] - base3.y) ** 2 + (K[2] - base3.z) ** 2 - dis3 ** 2,
            # (K[0] - base4.x) ** 2 + (K[1] - base4.y) ** 2 + (K[2] - base4.z) ** 2 - dis4 ** 2,
        ]
    dists = [dis1, dis2, dis3, dis4]
    P = []
    for i in range(4):
        P.append([bases[i].x, bases[i].y, bases[i].z])
    results = []
    r1 = op.fsolve(f0, P[0])
    r2 = op.fsolve(f1, P[1])
    r3 = op.fsolve(f2, P[2])
    r4 = op.fsolve(f3, P[3])
    # print(r1, r2, r3, r4)
    results = [r1, r2, r3, r4]
    INF_LOC = [10000, 10000, 10000]
    for i in range(4):
        if get_distance(results[i], P[i]) < 1e-5:
            results[i] = INF_LOC
    # print(results)
    min_var = 0xfffffff
    min_loc = []
    # print(results)
    for i in range(4):
        results[i] = [j for j in results[i]]
        va = get_variance(i, bases, dists, results[i])
        if va < min_var:
            min_var = va
            min_loc = results[i]
    if min_var > 200 or min_var < -200:
        return []
    print(min_var)
    return min_loc


def decode_location(dis1, dis2, dis3, dis4) -> list:
    dists = [dis1, dis2, dis3, dis4]

    ux, uy, uz = symbols('ux uy uz', real=True)
    eqs = [
        (ux - base1.x) ** 2 + (uy - base1.y) ** 2 + (uz - base1.z) ** 2 - dis1 ** 2,
        (ux - base2.x) ** 2 + (uy - base2.y) ** 2 + (uz - base2.z) ** 2 - dis2 ** 2,
        (ux - base3.x) ** 2 + (uy - base3.y) ** 2 + (uz - base3.z) ** 2 - dis3 ** 2,
        (ux - base4.x) ** 2 + (uy - base4.y) ** 2 + (uz - base4.z) ** 2 - dis4 ** 2,
    ]

    # p1 = solve_eq_concurrent([eqs[1], eqs[2], eqs[3]], [ux, uy, uz], result1)
    # p2 = solve_eq_concurrent([eqs[0], eqs[2], eqs[3]], [ux, uy, uz], result2)
    # p3 = solve_eq_concurrent([eqs[0], eqs[1], eqs[3]], [ux, uy, uz], result3)
    # p4 = solve_eq_concurrent([eqs[0], eqs[1], eqs[2]], [ux, uy, uz], result4)
    result1 = nonlinsolve([eqs[1], eqs[2], eqs[3]], [ux, uy, uz])
    result2 = nonlinsolve([eqs[0], eqs[2], eqs[3]], [ux, uy, uz])
    result3 = nonlinsolve([eqs[0], eqs[1], eqs[3]], [ux, uy, uz])
    result4 = nonlinsolve([eqs[0], eqs[1], eqs[2]], [ux, uy, uz])
    # for p in [p1, p2, p3, p4]:
    #     p.join()
    results = [result1, result2, result3, result4]
    # print(handle_solve_result(result1))
    # print(handle_solve_result(result2))
    # print(handle_solve_result(result3))
    # print(handle_solve_result(result4))
    min_var = 0xfffffff
    min_loc = []
    for i in range(4):
        results[i] = handle_solve_result(results[i])
        for locs in results[i]:
            va = get_variance(i, bases, dists, locs)
            if va < min_var:
                min_var = va
                min_loc = locs
    if min_var > 200 or min_var < -200:
        return []
    print(min_var)
    return min_loc


def solve_eq_concurrent(eqs: list, symbs: list, result: list) -> multiprocessing.Process:
    manager = multiprocessing.Manager()
    p = multiprocessing.Process(target=_solve, args=(eqs, symbs, result,))
    p.start()
    # p.join()
    return p


def _solve(eqs: list, symbs: list, ret_list: list):
    ret_list.append(nonlinsolve(eqs, symbs))


def handle_solve_result(result: list) -> list:
    locs = []
    for loc in result:
        l = []
        complex_flag = False
        for i in list(loc):
            if str(i).__contains__("I"):
                complex_flag = True
                break
            l.append(float(i))
        if not complex_flag:
            locs.append(l)
    return locs


def get_variance(eqs_num, bases: list, dists: list, solve_locs: list) -> float:
    __bound__ = 200
    ux = solve_locs[0]
    uy = solve_locs[1]
    uz = solve_locs[2]
    variance = (ux - bases[eqs_num].x) ** 2 \
               + (uy - bases[eqs_num].y) ** 2 \
               + (uz - bases[eqs_num].z) ** 2 \
               - dists[eqs_num] ** 2
    return math.sqrt(abs(variance))


def get_distance(loc1_3d: list, loc2_3d: list) -> float:
    return math.sqrt(
        (loc1_3d[0] - loc2_3d[0]) ** 2 +
        (loc1_3d[1] - loc2_3d[1]) ** 2 +
        (loc1_3d[2] - loc2_3d[2]) ** 2
    )