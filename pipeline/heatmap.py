import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import data_preparation

data = data_preparation.prep_data('train')

print(data.corr().status)
plt.figure(figsize=(16, 6))
ax = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
plt.show()