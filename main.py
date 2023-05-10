from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Any

import pandas as pd
from matplotlib import pyplot as plt

import filter
from location_decoder import *


def read_from_xlsx(path: str, sheet_name: str) -> list:
    sheet = pd.read_excel(path, sheet_name=sheet_name)
    lines = []
    # print(sheet.index)
    for i in range(sheet.index.start, sheet.index.stop):  # 实际上应该是sheet.index.stop
        lines.append(list(sheet.loc[i].values))
    return lines


def get_movement(data_path: str, sheet_name: str) -> list:
    moves = read_from_xlsx(data_path, sheet_name)
    move_trail = compute(moves, [])
    return move_trail


def get_movement_concurrent_mac_m1(data_path: str, sheet_name: str) -> list:
    move_trail = []
    moves = read_from_xlsx(data_path, sheet_name)
    m1_cpu_cores = 20
    step = math.ceil(len(moves) / m1_cpu_cores)
    processes = []
    manager = multiprocessing.Manager()
    ret_list = manager.list()
    for i in range(0, len(moves), step):
        # print(i)
        ms = moves[i:min(len(moves), i + step)]
        # t = threading.Thread(target=compute, args=(ms,)).start()
        p = multiprocessing.Process(target=compute, args=(ms, ret_list))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    for th_res in ret_list:
        for i in th_res:
            move_trail.append(i)
    return move_trail


def get_movement_concurrent_windows(data_path: str, sheet_name: str) -> list:
    move_trail = []
    moves = read_from_xlsx(data_path, sheet_name)
    i7_cpu_cores = 20
    pool = ThreadPoolExecutor(max_workers=i7_cpu_cores)
    step = math.ceil(len(moves) / i7_cpu_cores)
    futures = []
    for i in range(0, len(moves), step):
        ms = moves[i:min(len(moves), i + step)]
        futures.append(pool.submit(compute, ms, []))

    for f in futures:
        move_trail.append(r for r in f.result())
    pool.shutdown()
    return move_trail


def compute(moves: list, ret_list: list):
    locs = []
    for i in range(len(moves)):
        m = moves[i]
        print(i, m)
        loc = decode_location_scipy(m[0], m[1], m[2], m[3])
        if len(loc) != 0:
            locs.append(loc)
    ret_list.append(locs)
    return locs


def draw_3d_trial(locs: list, step: int = 1):
    fig = plt.figure(figsize=(34, 25))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    x, y, z = [], [], []
    for loc in locs:
        x.append(loc[0])
        y.append(loc[1])
        z.append(loc[2])
    ax1.plot3D(x[::step], y[::step], z[::step])
    ax1.view_init(80, 210)
    ax2.scatter3D(x[::step], y[::step], z[::step], c='gray', s=4.5)
    ax2.view_init(30, 210)
    ax3.plot3D(x[::step], y[::step], z[::step])
    ax3.view_init(30, 210)
    ax4.plot3D(x[::step], y[::step], z[::step])
    ax4.view_init(0, 210)
    plt.show()


def handle_movements_with_kalman_filter_2d(raw_movements: list, filter_nums: int = 1) -> list:
    moves_with_kalman = raw_movements
    for i in range(filter_nums):
        x, y, z = get_xyz(moves_with_kalman)
        moves_with_kalman = []
        predicts_xy = filter.kalman_filter_2d(x, y)
        predicts_xz = filter.kalman_filter_2d(x, z)
        predicts_yz = filter.kalman_filter_2d(y, z)
        for i in range(min(len(predicts_xy), len(predicts_xz), len(predicts_yz))):
            x_avg = (predicts_xy[i][0] + predicts_xz[i][0]) / 2
            y_avg = (predicts_xy[i][1] + predicts_yz[i][0]) / 2
            z_avg = (predicts_xz[i][1] + predicts_yz[i][1]) / 2
            const_invalid_bound = 210
            if x_avg < const_invalid_bound or y_avg < const_invalid_bound:
                continue
            moves_with_kalman.append([x_avg, y_avg, z_avg])
    return moves_with_kalman


def handle_movements_with_kalman_filter_3d(raw_movements: list, filter_nums: int = 1) -> list:
    moves_with_kalman = raw_movements
    for i in range(filter_nums):
        x, y, z = get_xyz(moves_with_kalman)
        moves_with_kalman = filter.kalman_filter_3d(x, y, z)
    return moves_with_kalman


def get_xyz(coordinates: list) -> tuple[list[Any], list[Any], list[Any]]:
    x, y, z = [], [], []
    for c in coordinates:
        x.append(c[0])
        y.append(c[1])
        z.append(c[2])
    return x, y, z


if __name__ == '__main__':
    movements = get_movement("data/3.xlsx", 'Sheet1')
    # print(movements)
    print(len(movements))
    draw_3d_trial(movements)
    # draw_3d_trial(handle_movements_with_kalman_filter_2d(movements, filter_nums=5), step=1)
    # draw_3d_trial(handle_movements_with_kalman_filter_3d(movements, filter_nums=80), step=1)
    # print(decode_location_scipy(271, 338, 400, 380))
    # print(decode_location(271, 338, 400, 380))
    _3d = handle_movements_with_kalman_filter_3d(movements, filter_nums=400)
    print(_3d)
    draw_3d_trial(_3d)
    draw_3d_trial(handle_movements_with_kalman_filter_2d(_3d, filter_nums=2))