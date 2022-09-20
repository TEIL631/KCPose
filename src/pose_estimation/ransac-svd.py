import numpy as np
import matplotlib.pyplot as plt

# Please specify your camera intrinsic parameter
def backward_projection(u, v, depth):
    depth = depth
    x_3d = depth * (u - 318.155) / 381.29
    y_3d = depth * (v - 238.028) / 381.29 
    d_3d = depth
    return [x_3d, y_3d, d_3d]


def RANSAC_SVD(raw_origin_points, raw_detected_points):
    detected_points = raw_detected_points.copy()
    origin_points = raw_origin_points.copy()

    detected_points = detected_points.astype('float64')
    origin_points = origin_points.astype('float64')

    detected_centroid = np.average(detected_points, axis=0)
    origin_centroid = np.average(origin_points, axis=0)

    
    detected_points -= detected_centroid
    origin_points -= origin_centroid

    h = detected_points.T @ origin_points
    u, _, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    r = v @ e @ u.T

    # calculate T
    origin_points += origin_centroid
    trans_points = origin_points @ r
    trans_centroid = np.average(trans_points, axis=0)

    t = detected_centroid - trans_centroid
    estimate_points = trans_points + t

    return estimate_points, r, t
    
def calculate_add(estimated_points, gt_points):
    add = 0
    for i in range(estimated_points.shape[0]):
        e = estimated_points[i]
        g = gt_points[i]
        add += np.sqrt((e[0] - g[0]) ** 2 + (e[1] - g[1])**2 + (e[2] - g[2]) ** 2)
    return add / estimated_points.shape[0]

def calculate_adds(estimated_points, gt_points):
    adds = 0
    for i in range(estimated_points.shape[0]):
        min_dist = np.inf
        for j in range(gt_points.shape[0]):
            e = estimated_points[i]
            g = gt_points[j]
            dist = np.sqrt((e[0] - g[0]) ** 2 + (e[1] - g[1])**2 + (e[2] - g[2]) ** 2)
            if dist < min_dist:
                min_dist = dist
        adds += min_dist
    return adds / estimated_points.shape[0]