# coding=utf-8
# from scipy.optimize import minimize
# import numpy as np
#
# # demo 1
# # 计算1/x + x 得最小值
# # def fun(args):
# #     a = args
# #     v = lambda x: a/x[0] + x[0]
# #     return v
#
# # demo 2
# # minimize with constraints
# # 计算 (2+x1)/(1+x2) - 3x1 + 4x3  得最小值， x的范围是[0.1, 0.9]
# def fun(args):
#     a, b, c, d = args  # args should be tuple
#     v = lambda x: (a+x[0])/(b+x[1]) - c*x[0] + d*x[2]
#     return v
#
# def con(args):
#     # constraints only used for 'SLSQP' and 'L-BFGS-B'
#     # constraints conditions include eq and ineq
#     # eq 表示函数结果为零， ineq表示函数结构大于等于零
#     x1lim, x1max, x2lim, x2max, x3lim, x3max = args
#     cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1lim},
#             {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
#             {'type': 'ineq', 'fun': lambda x: x[1] - x2lim},
#             {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
#             {'type': 'ineq', 'fun': lambda x: x[2] - x3lim},
#             {'type': 'ineq', 'fun': lambda x: -x[2] + x3max},
#             )
#     return cons
# '''
# # 约束也可以是函数, 例如：
# def con(args):
#     a, b, i = args
#     def v(x):
#         return np.log2(1+x[i]*a/b) -5
#
#     return v
#
# cons = ({'type': 'ineq', 'fun': con(args)},
#         {'type': 'ineq', 'fun': con(args)}
#         )
# '''
#
# if __name__ == '__main__':
#
#     # args = (1)  # value of a
#     # x0 = np.asarray((2))  # give the initial valie
#     # # res = minimize(fun(args), x0, method='SLSQP')  # Jacobian is required for Newton-CG method
#     # res = minimize(fun(args), x0, method='BFGS')  # more accurate than 'SLSQP'
#     args = (2, 1, 3, 4)
#     args1 = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
#     cons = con(args1)
#     x0 = np.asarray((0.5, 0.5, 0.5))
#     res = minimize(fun(args), x0, method='SLSQP', constraints=cons)
#     print(res.fun)  # the minimal function value
#     print(res.success)  # ob reach  the minimize
#     print(res.x)  # solution of x by the minimal function value
#
import numpy as np

from scipy import interpolate

a = np.asarray((1,2,3,4)).reshape(2,2)
print(a)
H, W = np.shape(a)
b = np.zeros((2*H, 2*W))
b[::2, ::2] = a
b[b==0] = np.nan
print(b)

x = np.arange(0, b.shape[1])
y = np.arange(0, b.shape[0])

array = np.ma.masked_invalid(b)

xx, yy = np.meshgrid(x, y)
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]

GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='cubic', fill_value=0)
print(GD1)