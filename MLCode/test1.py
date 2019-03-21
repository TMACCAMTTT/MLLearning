#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
plt.show()

# y = 5
# print(y**2)