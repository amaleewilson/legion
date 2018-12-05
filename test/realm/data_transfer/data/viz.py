import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv
import pdb
import sys
import matplotlib.pylab as plt2


dfs = []

# dfs.append(pd.read_csv('modified_og_transpose.csv'))
# dfs.append(pd.read_csv('perfdata_kernel_transpose1.csv'))		
# dfs.append(pd.read_csv('modified_shared_transpose.csv'))		
# dfs.append(pd.read_csv('perfdata_kernel_transpose2.csv'))
# dfs.append(pd.read_csv('perfdata_kernel_share_transpose.csv'))	
# dfs.append(pd.read_csv('perfdata_kernel_no_transpose.csv'))		
# dfs.append(pd.read_csv('perfdata_memcpy_no_transpose.csv'))	
# dfs.append(pd.read_csv('perfdata_memcpy_transpose1.csv'))
# dfs.append(pd.read_csv('shared_mem_vs_non.csv'))

dfs.append(pd.read_csv('data_201812041831.csv'))



df = pd.concat(dfs, ignore_index=True, sort=False)
df["mb"] = round(df["bytes"]/1e+6,1); 
df["gb/s"] = df["bytesPerNano"]; 

# g = sns.catplot("mb", col="fieldIDcount", y="gb/s", hue="method", kind="bar", col_wrap=1,  data=df)

g = sns.barplot(x="mb", y="gb/s", hue="method", data=df[df.bytes > 131072])

g.set(ylim=(df[df.bytes > 131072]["gb/s"].min() - 1,df[df.bytes > 131072]["gb/s"].max() + 1))

# g.set_xticklabels(rotation=45)
# g.fig.get_axes()[0].set_yscale('log')
plt.show()

# dfs = []

# dfs.append(pd.read_csv('perfdata_kernel_transpose201811301122.csv'))
# dfs.append(pd.read_csv('perfdata_kernel_transpose201811301120.csv'))

# df = pd.concat(dfs, ignore_index=True, sort=False)
# df["mb"] = round(df["bytes"]/1e+6,3); 
# df["gb/s"] = df["bytesPerNano"]; 

# g = sns.catplot("mb", col="fieldIDcount", y="gb/s", hue="method", kind="bar", col_wrap=1,  data=df)

# g.set_xticklabels(rotation=45)
# g.fig.get_axes()[0].set_yscale('log')
# plt.show()