import numpy as np, scipy as sp, matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
from util import Line
import util
import math

def min_vol_ellipsoid(points, tolerance = 1e-3):
    #Khachiyan Algorithm
    #https://stackoverflow.com/questions/1768197/bounding-ellipse
    #https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
    #https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
    P = np.copy(np.asarray(points))
    hull = ConvexHull(P)
    P = P[hull.vertices]
    P = P.T

    #P = np.copy(np.asarray(points)).T

    d = P.shape[0]
    N = P.shape[1]
    Q = np.ones((d+1,N))
    Q[:-1,:] = P
    err = tolerance + 1
    u = np.ones((N,)) * (1/N)
    while err > tolerance:
        X = Q @ np.diag(u) @ Q.T
        M = np.diag( Q.T @ np.linalg.inv(X) @ Q )
        j = M.argmax()
        maximum = M[j]
        step_size = (maximum - d - 1) / ( (d + 1) * (maximum - 1) )
        new_u = u * (1 - step_size)
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    U = np.diag(u)
    c = P.dot(u)
    A = (1/d) * np.linalg.inv(P @ U @ P.T - np.multiply.outer(c,c))
    return A,c

def mvee(points, tol = 0.001):
    """
    Find the minimum volume ellipse.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    """
    la = np.linalg
    points = np.asmatrix(points)
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = Q * np.diag(u) * Q.T
        M = np.diag(Q.T * la.inv(X) * Q)
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = u*points
    A = la.inv(points.T*np.diag(u)*points - c.T*c)/d
    return np.asarray(A), np.squeeze(np.asarray(c))

def ellipse_to_plot_matr(A, c):
    u,s,vh = np.linalg.svd(A)
    t = np.linspace(0, 2*np.pi, 100)
    Ell = np.array(np.expand_dims(1/np.sqrt(s),1) * [np.cos(t) , np.sin(t)])
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(vh.T,Ell[:,i])
    Ell_rot += np.expand_dims(c,1)
    return Ell_rot[0,:], Ell_rot[1,:]

def ellipse_vol_matr(A):
    u,s,vh = np.linalg.svd(A)
    dim = len(s)
    coef = (np.pi ** (dim/2)) / math.gamma(dim/2 + 1)
    return np.prod(1/np.sqrt(s)) * coef

def ellipse_area(width, height):
    return width * height * np.pi / 4

def ellipsoid_vol(radii):
    dim = len(radii)
    coef = (np.pi ** (dim/2)) / math.gamma(dim/2 + 1)
    return coef * np.prod(radii)

def Petunin_ellipse(points):
    """
    Строит эллипс Петунина
    :param points: shape (n_points, 2)
    :return: c, width, height, alpha
        c - центр эллипса
        width - диаметр по оси Ox
        height -диаметр по оси Oy
        alpha - угол поворота
    """
    x = np.asarray(points)
    points = x
    hull = ConvexHull(points)
    v = hull.vertices
    vl = v.shape[0]
    dists = np.array([[np.linalg.norm(x[v[i]] - x[v[j]]), (v[i],v[j])]
                    for i in range(vl)
                    for j in range(i)])
    #x[k] и x[l] лежат на диаметре
    k,l = dists[np.argmax(dists[:,0]),1]

    #прямая a*x_1 + b*x_2 + c = 0 по точкам x[k] и x[l]
    L = Line(points=[x[k],x[l]])

    #x[r] и x[q] наиболее отдалены от прямой
    rel_dists_line = np.array([[L.rel_dist(x[ind]), ind]
                                  for ind in v])
    rel_dists_line[np.isclose(rel_dists_line, np.zeros_like(rel_dists_line), atol=1e-04)]  = 0.
    dists_line = np.abs(rel_dists_line)
    r = int(dists_line[np.argmax(dists_line[:,0]), 1])
    r_sign = np.sign(L.rel_dist(x[r]))
    rel_dists_other_side = rel_dists_line[np.sign(rel_dists_line[:,0]) != r_sign]
    dists_other_side = np.abs(rel_dists_other_side)
    q = int(dists_other_side[np.argmax(dists_other_side[:,0]), 1])

    L1 = L.build_parallel(x[r])
    L2 = L.build_parallel(x[q])
    L3 = L.build_perpend(x[k])
    L4 = L.build_perpend(x[l])

    #прямоугольник ABCD, ограниченный прямыми L1, L2, L3, L4
    A = L1.intersection(L3)
    B = L1.intersection(L4)
    C = L2.intersection(L4)
    D = L2.intersection(L3)
    #a = np.linalg.norm(A - B)
    #b = np.linalg.norm(B - C)


    #plt.plot(points[:,0], points[:,1], 'ro')
    #plt.plot([x[k][0], x[l][0]], [x[k][1],x[l][1]], 'b-')
    #plt.plot(x[r][0], x[r][1], 'b^')
    #plt.plot(x[q][0], x[q][1], 'b^')
    #plt.plot(A[0], A[1], 'y+')
    #plt.plot(B[0], B[1], 'y+')
    #plt.plot(C[0], C[1], 'y+')
    #plt.plot(D[0], D[1], 'y+')
    #plt.show()

    #ищем самую левою (если несколько - левую нижнюю) точку прямоугольника
    v_rect = np.array([A,B,C,D])
    left_x = np.min(v_rect[:,0])
    bot_y = np.min(v_rect[v_rect[:,0]==left_x,1])
    #ind = np.argmax(v_rect[:] == np.array([left_x,bot_y]))
    ind = np.where(np.all(v_rect == np.array([left_x,bot_y]),axis=1))[0][0]
    vert = v_rect[ind]
    #ищем прямую, которая перейдёт в ось Ox, и прямую, которая перейдёт в Oy
    lines_adj_to_vert = np.array([[L1,L3],[L1,L4],[L2,L4],[L2,L3]])[ind]
    if lines_adj_to_vert[0].angle_coef() <= 0:
        L_Ox = lines_adj_to_vert[0]
    else:
        L_Ox = lines_adj_to_vert[1]

    #Перенос vert в начало координат
    new_points = points - vert
    alpha = -np.arctan2(-L_Ox.a, L_Ox.b) % np.pi

    #матрица поворота
    rot = np.array([[np.cos(alpha), -np.sin(alpha)],
                 [np.sin(alpha), np.cos(alpha)]])
    new_points = rot.dot(new_points.T).T

    a, b = np.max(new_points,axis=0)
    #помещаем точки в квадрат с размером стороны a
    coef = np.array([1, a/b])
    new_points *= coef

    #center = np.array([a/2, a/2])
    center = np.sum(new_points,axis=0) / new_points.shape[0]
    dists_center = np.linalg.norm(new_points - center, axis = 1)
    R = np.max(dists_center)

    #применяем обратное превращение к оуркжности радиуса R с центром center
    c = center
    c /= coef
    c = rot.T.dot(c)
    c += vert
    width = 2 * R
    height = 2 * R * b/a
    alpha = - alpha
    alpha %= np.pi
    return c, width, height, alpha

def Petunin_ellipsoid(points):
    """
    Строит эллипсоид Петунина
    :param points: shape = (n_points, dim)
    :return: c, radii, rot, alphas
    c - центр эллипсоида
    R - массив относительных расстояний до каждой точки входного массива (не отсортирован)
    coef - коеффициенты для получения полуосей эллипсоидов; Полуоси эллипсоида, отвечающего i-ой
           входной точке, являются R[i] * coef
    rot - матрица поворота
    alphas - shape = (dim-1,), углы поворотов, где alphas[i] - угол поворота в плоскости x_{i}, x_{i+1};
             применять от 0 до dim-1
    """
    x = np.asarray(points)
    x = np.copy(x)
    d = x.shape[1]
    hull = ConvexHull(x)
    v = hull.vertices
    vl = v.shape[0]
    dists = np.array([[np.linalg.norm(x[v[i]] - x[v[j]]), (v[i],v[j])]
                    for i in range(vl)
                    for j in range(i)])
    #x[k] и x[l] лежат на диаметре
    k,l = dists[np.argmax(dists[:,0]),1]

    #выбираем точку с меньшей первой координатой, она перейдёт в начало координат
    if x[k][0] <= x[l][0]:
        x0 = x[k]
        x1 = x[l]
    else:
        x0 = x[l]
        x1 = x[k]

    new_x = x - x0

    direction = x1 - x0
    #поворачиваем direction так, чтобы вектор лежал на первой координатной оси (с положительной стороны)
    rot = np.eye(d)
    alphas = np.zeros((d-1,))
    for i in range(d - 1, 0, -1):
        if direction[i] == 0:
            continue
        alpha = -np.arctan2(direction[i], direction[i-1])
        alphas[i-1] = alpha
        cur_rot = util.rotation_matrix(d,i-1, i, alpha)
        direction = cur_rot.dot(direction)
        rot = cur_rot @ rot

    new_x = rot.dot(new_x.T).T

    #строим параллелепипед наименьшего объема; ищем нижние и верхние границы области по всем координатам
    lower = np.min(new_x[v],axis=0)
    upper = np.max(new_x[v],axis=0)

    #размеры сторон параллелепипеда
    size = upper - lower

    #смещаем параллелепипед и растягиваем его до гиперкуба со стороной size[0]
    new_x -= lower
    coef = size[0] / size
    new_x *= coef

    center = np.sum(new_x, axis=0) / new_x.shape[0]
    dists_center = np.linalg.norm(new_x - center, axis = 1)
    #dtype = [("dist", float), ("index", int)]
    #to_sort = np.array([(dist, k) for k, dist in enumerate(dists_center)], dtype=dtype)
    #sorted_dists = np.sort(to_sort, order="dist")
    #order = sorted_dists["index"]
    #R = sorted_dists["dist"]
    R = dists_center
    #R = np.max(dists_center)

    #обратные превращения
    c = center / coef
    c += lower
    c = rot.T.dot(c)
    c += x0

    #radii = R[-1] / coef

    return c, R, 1 / coef, rot.T, -alphas

def peeling_concentric_rel_dists(R):
    """
    Статистический концентрический пилинг
    :param R: shape = (n_samples,) - относительные расстояния, вычисленные при построении эллипсоида Петунина
    :return: order, shape = (n_samples,) - массив индексов точек, отсортированных в порядке возрастания площади эллипса
    """
    dtype = [("dist", float), ("index", int)]
    to_sort = np.array([(dist, k) for k, dist in enumerate(R)], dtype=dtype)
    sorted_dists = np.sort(to_sort, order="dist")
    order = sorted_dists["index"]
    return order

def peeling_concentric(points):
    """
    Статистический концентрический пилинг
    :param points: shape = (n_samples, dim)
    :return: order, shape = (n_samples,) - массив индексов точек, отсортированных в порядке возрастания площади эллипса
    """
    c, R, coef, rot, alphas = Petunin_ellipsoid(points)
    return concentric_peeling_rel_dists(R)

def peeling_entrywise(points):
    """
    Статистический поэлементный пилинг
    :param points: shape = (n_samples, dim)
    :return: order, shape = (n_samples,) - массив индексов точек, отсортированных в порядке возрастания площади эллипса
    """
    n = len(points)
    order = np.zeros((n,), dtype=int)
    cur_points = np.copy(np.asarray(points))
    cur_indeces = np.array([k for k in range(n)])
    for i in range(n - 1, 1, -1):
        c, R, coef, rot, alphas = Petunin_ellipsoid(cur_points)
        dtype = [("dist", float), ("index", int)]
        to_sort = np.array([(dist, k) for k, dist in zip(cur_indeces, R)], dtype=dtype)
        sorted_dists = np.sort(to_sort, order="dist")
        ind = sorted_dists["index"][-1]
        order[i] = ind
        not_to_del = cur_indeces != ind
        cur_points = cur_points[not_to_del]
        cur_indeces = cur_indeces[not_to_del]
    order[0:2] = sorted_dists["index"][0:2]
    return order
