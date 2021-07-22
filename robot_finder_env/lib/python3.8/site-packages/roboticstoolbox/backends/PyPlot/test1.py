import matplotlib.pyplot as plt

fig = plt.figure()

# ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection='3d')

ax = fig.add_subplot(projection='3d')

# ax.plot([0], [0], [0], '*')
ax.plot(0, 0)


plt.show(block=True)