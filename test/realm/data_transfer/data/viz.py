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

dfs.append(pd.read_csv('data_201812122234.csv'))



df = pd.concat(dfs, ignore_index=True, sort=False)
df["mb"] = round(df["bytes"]/1e+6,1); 
df["gb/s"] = df["bytesPerNano"];



#
#hmd_t1 = pd.DataFrame(df[(df["method"] == "kernel_transpose_multi_batch")])
#hmd_t1.block_count = pd.to_numeric(hmd_t1.block_count)
#hmd_t1.copy_count = pd.to_numeric(hmd_t1.copy_count)
#hmd_t1.drop(columns=["machine", "mb", "fieldIDcount", "bytesPerNano", "bytes", "elemcount", "block_size"])
#hmd_t1 = pd.pivot_table(hmd_t1, index="block_count", columns="copy_count", values="gb/s")
#
#ax = sns.heatmap(hmd_t1, annot=True)
#
#ax.set_title("kernel_trans1_multi_batch")
#plt.show()
#
#
#hmd_t2 = pd.DataFrame(df[(df["method"] == "kernel_trans2_multi_batch")])
#hmd_t2.block_count = pd.to_numeric(hmd_t2.block_count)
#hmd_t2.copy_count = pd.to_numeric(hmd_t2.copy_count)
#hmd_t2.drop(columns=["machine", "mb", "fieldIDcount", "bytesPerNano", "bytes", "elemcount", "block_size"])
#hmd_t2 = pd.pivot_table(hmd_t2, index="block_count", columns="copy_count", values="gb/s")
#
#ax = sns.heatmap(hmd_t2, annot=True)
#
#ax.set_title("kernel_trans2_multi_batch")
#plt.show()


df["trans_method"] = df["method"] + df["copy_count"]

scd = pd.DataFrame(df)
scd = scd[(scd["copy_count"] == "8") | (scd["copy_count"] == "na")]
scd["method"] = scd["method"] + "_cc_" + scd["copy_count"]
scd2 = pd.DataFrame(df)
scd2 = scd2[(scd2["method"] != "kernel_trans2_multi_batch") & (scd2["method"] != "kernel_transpose_multi_batch")]

data = pd.concat([scd, scd2])


g = sns.catplot("block_count", col="fieldIDcount", y="gb/s", hue="method", kind="point", col_wrap=1,  data=data)

#g = sns.barplot(x="block_count", y="gb/s", hue="method", data=df[df.bytes > 131072])

#g.set(ylim=(df[df.bytes > 131072]["gb/s"].min() - 1,df[df.bytes > 131072]["gb/s"].max() + 1))

g.set_xticklabels(rotation=45)
# g.fig.get_axes()[0].set_yscale('log')
plt.show()






# dfs = []

# dfs.append(pd.read_csv('perfdata_kernel_transpose201811301122.csv'))
# dfs.append(pd.read_csv('perfdata_kernel_transpose201811301120.csv'))

# df = pd.concat(dfs, ignore_index=True, sort=False)
# df["mb"] = round(df["bytes"]/1e+6,3); 
# df["gb/s"] = df["bytesPerNano"]; 

# g = sns.catplot("mb", col="fieldIDcount", y="gb/s", hue="method", kind="line", col_wrap=1,  data=df)

# g.set_xticklabels(rotation=45)
# g.fig.get_axes()[0].set_yscale('log')
# plt.show()
