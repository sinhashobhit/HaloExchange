'''
Code for the Plot generation
Authors : Shobhit Sinha, Mayank Bansal
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys


def plotter(dfx,name,P_val):
	fig, ax = plt.subplots(figsize=(15,8)) 
	ax.set_yscale('log')
	ax = sns.boxplot(x="N", y="T", hue="Legend",ax = ax, data=dfx) 	
	# for patch in ax.artists:
	# 	r, g, b, a = patch.get_facecolor()
	# 	patch.set_facecolor((r, g, b, .7))
	#ax = sns.lineplot(x="N", y="T", hue="Legend",ax = ax, data=dfx) 	
	d_copy = dfx.copy()
	d_new = d_copy.sort_values(by=['N','Legend','T'])
	d_copy = d_new.iloc[2::5, :]
	ax = sns.pointplot(x="N", y="T", hue="Legend",ax = ax, data=d_copy)
	plt.grid(True)
	plt.xlabel("On x-axis : Value of N, the dimension of square data matrix") 
	plt.ylabel("On y-axis : Time on logarithmic scale in s ") 
	plt.title('Time in s Vs N (dimension of square data matrix)for P = '+str(P_val))

	plt.savefig(name)
	#plt.show()
	
df = pd.read_csv(sys.argv[1],header=None, names=["P", "N","Legend","T"])
df1 = df[df['P'] == 16]
df2 = df[df['P'] == 36]
df3 = df[df['P'] == 49]
df4 = df[df['P'] == 64]
pd.set_option("display.max_rows", None, "display.max_columns", None)

plotter(df1, "P16boxplot.png",16)
plotter(df2, "P36boxplot.png",36)
plotter(df3, "P49boxplot.png",49)
plotter(df4, "P64boxplot.png",64)

