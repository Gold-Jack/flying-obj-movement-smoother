import cv2
import numpy as np


def kalman_filter_2d(axis1: list, axis2: list) -> list:
    kf = cv2.KalmanFilter(4, 2)
    # 设置测量矩阵
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # 设置转移矩阵
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # 设置过程噪声协方差矩阵
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    last_measurement = current_measurement = np.array([[np.float32(axis1[0])], [np.float32(axis2[0])]], np.float32)
    last_prediction = current_prediction = np.array([[np.float32(axis1[0]) + 0.1], [np.float32(axis2[0]) + 0.1]], np.float32)
    predicts = []
    for i in range(len(axis1)):
        # 初始化
        last_measurement = current_measurement
        last_prediction = current_prediction
        # 传递当前测量坐标值
        current_measurement = np.array([[np.float32(axis1[i])],
                                        [np.float32(axis2[i])]])
        # 用来修正卡尔曼滤波的预测结果
        kf.correct(current_measurement)
        # 调用kalman这个类的predict方法得到状态的预测值矩阵，用来估算目标位置
        current_prediction = kf.predict()
        # 上一次测量值
        lmx, lmy = last_measurement[0], last_measurement[1]
        # 当前测量值
        cmx, cmy = current_measurement[0], current_measurement[1]
        # 上一次预测值
        lpx, lpy = last_prediction[0], last_prediction[1]
        # 当前预测值
        cpx, cpy = current_prediction[0], current_prediction[1]
        __bound__ = 260
        if cpx > __bound__ or cpx < 180:
            cpx = cmx
            current_prediction[0] = cpx
        if cpy > __bound__ or cpy < 180:
            cpy = cmy
            current_prediction[1] = cpy
        predicts.append([float(cmx + cpx) / 2, float(cmy + cpy) / 2])
    return predicts


def kalman_filter_3d(axis1: list, axis2: list, axis3: list) -> list:
    kf = cv2.KalmanFilter(9, 3, 0)

    x0, y0, z0 = axis1[0], axis2[0], axis3[0]
    kf.measurementMatrix = np.array([
        [x0, 0, 0, y0, 0, 0, z0, 0, 0],
        [0, x0, 0, 0, y0, 0, 0, z0, 0],
        [0, 0, x0, 0, 0, y0, 0, 0, z0]
    ], np.float32)

    # kf.transitionMatrix = np.array([
    #     [x0, 0, 0, y0, 0, 0, z0, 0, 0],
    #     [0, x0, 0, 0, y0, 0, 0, z0, 0],
    #     [0, 0, x0, 0, 0, y0, 0, 0, z0],
    #     [0, 0, 0, x0, 0, 0, y0, 0, 0],
    #     [0, 0, 0, 0, x0, 0, 0, y0, 0],
    #     [0, 0, 0, 0, 0, x0, 0, 0, y0],
    #     [0, 0, 0, 0, 0, 0, x0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, x0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, x0]
    # ], np.float32)

    kf.transitionMatrix = np.array([
        [1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32)

    kf.processNoiseCov = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32) * 0.007

    kf.measurementNoiseCov = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], np.float32) * 0.2

    # last_measurement = current_measurement = np.array([[np.float32(x0)], [np.float32(y0)], [np.float32(z0)]], np.float32)
    # last_prediction = current_prediction = np.array([[np.float32(x0) + 0.1],
    #                                                  [np.float32(y0) + 0.02],
    #                                                  [np.float32(z0) + 0.1]], np.float32)
    last_measurement = current_measurement = np.array((3, 1), np.float32)
    last_prediction = current_prediction = np.zeros((3, 1), np.float32)
    predicts = []
    for i in range(1, len(axis1)):
        # 初始化
        last_measurement = current_measurement
        last_prediction = current_prediction
        # 传递当前测量坐标值
        current_measurement = np.array([[np.float32(axis1[i])],
                                        [np.float32(axis2[i])],
                                        [np.float32(axis3[i])]])
        # 用来修正卡尔曼滤波的预测结果
        kf.correct(current_measurement)
        # 调用kalman这个类的predict方法得到状态的预测值矩阵，用来估算目标位置
        current_prediction = kf.predict()
        # lmx, lmy, lmz = last_measurement[0], last_measurement[1], last_measurement[2]
        # lpx, lpy, lpz = last_prediction[0], last_prediction[1], last_prediction[2]
        # 当前测量值
        cmx, cmy, cmz = current_measurement[0], current_measurement[1], current_measurement[2]
        # 当前预测值
        cpx, cpy, cpz = current_prediction[0], current_prediction[1], current_prediction[2]
        __upper_bound__ = 260
        __lower_bound__ = 180
        if cpx > 260 or cpx < 180:
            cpx = cmx
            current_prediction[0] = cpx
        if cpy > 260 or cpy < 180:
            cpy = cmy
            current_prediction[1] = cpy
        if cpz > 100 or cpz < -100:
            cpz = cmz
            current_prediction[2] = cpz
        predicts.append([float(cmx + cpx) / 2, float(cmy + cpy) / 2, float(cmz + cpz) / 2])
        # predicts.append([float(cpx), float(cpy), float(cpz)])
    return predicts