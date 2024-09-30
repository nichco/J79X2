import numpy as np
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import pickle

_FT2M = 0.3048

_LBF2N = 4.4482216

_SCALE = 1E-3

# Note in the data that Mach varies fastest (the first 10 datapoints correspond to Alt=0)
# Altitude is given in ft and thrust is given in lbf

mach = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])

h = np.array([0.0, 5.0E3, 10.0E3, 15.0E3, 20.0E3,
                25.0E3, 30.0E3, 40.0E3, 50.0E3, 70.0E3]) * _SCALE#* _FT2M

thrust = np.array([30210., 26880.064, 28242.384, 31584.864, 34915.024,
                    36960., 37166.544, 35701.024, 33449.424, 32017.344,

                    #  alt=5000
                    28391.175, 25005.861467, 25144.153572, 27434.067627, 30723.757952,
                    34081.516875, 36795.774732, 38375.099867, 38548.198632, 37263.915387,

                    # alt=10000
                    24464.8, 22128.759472, 22005.577152, 23722.970032,
                    26812.239232, 30708.27, 34749.531712, 38178.077872, 40139.546112, 39683.158192,

                    #  alt=15000
                    19553.925, 18777.500827, 18952.033332, 20404.355787, 23187.984112,
                    27083.116875, 31596.635292, 35962.103227, 39139.767192, 39816.556347,

                    #  alt=20000
                    14554.8, 15375.527552, 16080.162432, 17434.192512,
                    19854.691712, 23410.32, 27821.323392, 32459.533952,
                    36348.369792, 38162.835072,

                    #  alt=25000
                    10136.875, 12240.980875, 13457.8665, 14771.630875, 16812.244,
                    19855.546875, 23823.2515, 28282.940875, 32448.069, 35177.960875,

                    #  alt=30000
                    6742.8,  9586.701232, 11124.309312, 12379.004592,
                    14056.705792, 16545.87, 19917.492672, 23925.107632,
                    28004.787072, 31275.141552,

                    #  alt=40000
                    3662.8,  6043.800832,  7336.374912,  8268.808192, 9371.531392,
                    10977.12, 13220.294272, 16037.919232, 19169.004672, 22154.705152,


                    #  alt=50000
                    4320., 4343.534, 4454.904, 4865.934, 5691.344,
                    6948.75, 8558.664, 10344.494, 12032.544, 13252.014,

                    #  alt=70000
                    # 100.0,     200.0,   400.0,   800.0,  1100.0,
                    # 1400.0,   1700.0,  2200.0,  2900.0,  3100.0
                    -5277.2, -3566.331728, -1933.530048,  -513.881168, 609.260032,
                    1404.27,  1891.256512,  2142.058672, 2280.246912,  2481.122992
                    ]).reshape((10, 10)) * _SCALE#* _LBF2N




# x = np.arange(-5.01, 5.01, 0.25)        # the grid is an outer product
# y = np.arange(-5.01, 7.51, 0.25)        # of x and y arrays

# xx, yy = np.meshgrid(x, y, indexing='ij')
# z = np.sin(xx**2 + 2.*yy**2)            # z array needs to be 2-D

func = RectBivariateSpline(h, mach, thrust, s=0)

xnew = np.linspace(min(h), max(h), 100)
ynew = np.linspace(min(mach), max(mach), 30)
znew = func(xnew, ynew)


dx = func.partial_derivative(dx=1, dy=0)
dy = func.partial_derivative(dx=0, dy=1)
# dnew = der(xnew, ynew)

# plt.imshow(znew)
# plt.colorbar()
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

X, Y = np.meshgrid(xnew, ynew)
X_OLD, Y_OLD = np.meshgrid(h, mach)
p1 = ax1.contourf(X, Y, znew.T)
p2 = ax2.contourf(X_OLD, Y_OLD, thrust.T)


plt.colorbar(p1, shrink=1, ax=ax1,)
plt.colorbar(p2, shrink=1, ax=ax2,)
plt.show()


# print(X.shape, Y.shape, znew.shape)
# zz = func(70000, 0.2)
# print(zz)
# dd = der(70000, 0.2)
# print(dd)

# plt.contourf(X, Y, dnew.T)
# plt.colorbar()
# plt.show()
# exit()

hh = np.linspace(min(h), max(h), 500)
data = np.zeros((len(hh)))
der = np.zeros((len(hh)))
for i, alt in enumerate(hh):
    data[i] = func(alt, 0.7)
    der[i] = dx(alt, 0.7)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(hh, data)
ax2.plot(hh, der)
plt.show()
# exit()



dict = {'func': func, 'dx': dx, 'dy': dy}

with open('data.pkl', 'wb') as f:
    pickle.dump(dict, f)