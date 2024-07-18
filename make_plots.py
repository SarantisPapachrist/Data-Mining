import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.api import qqplot 
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_activity(activity, df,data_num):
    custom_x_values = range(data_num, data_num*2 )
    dataplot = df[df['label'] == activity][['back_x', 'back_y', 'back_z','thigh_x','thigh_y','thigh_z']][0:data_num]
    dataplot.index = custom_x_values
    fig1, ax1 = plt.subplots(figsize=(16, 4))
    dataplot.iloc[:, :3].plot(ax=ax1, title='back')
    ax1.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    
    # Plot the remaining three features in another figure
    fig2, ax2 = plt.subplots(figsize=(16, 4))
    dataplot.iloc[:, 3:].plot(ax=ax2, title='thigh')
    ax2.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    
    plt.show()

def create_corr_matrix(df):
    cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    df_plot=df[cols]
    plt.figure(figsize=(30,15))
    sns.heatmap(df_plot.corr().abs().round(2),annot=True, cmap = "BuGn")
    plt.show()

def cluster3D(data,labels):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
        ax.set_title('HAR-Clustering in 3D')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()
def cluster2D(data,labels):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # XY Plane
    axes[0].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    axes[0].set_title('X-Y Plane')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # XZ Plane
    axes[1].scatter(data[:, 0], data[:, 2], c=labels, cmap='viridis')
    axes[1].set_title('X-Z Plane')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 3')

    # YZ Plane
    axes[2].scatter(data[:, 1], data[:, 2], c=labels, cmap='viridis')
    axes[2].set_title('Y-Z Plane')
    axes[2].set_xlabel('Principal Component 2')
    axes[2].set_ylabel('Principal Component 3')

    plt.tight_layout()
    plt.show()


def plot_histograms_qqplots(df):
    colors = ['#FF4C4C', '#8CD790', '#4D7EA8', '#E97451',  '#F2B134']

# Create a 2x4 grid for histograms and Q-Q plots
    fig1 = plt.figure(figsize=(17, 3))
    fig1.set_facecolor('#131516')

    gs1 = gridspec.GridSpec(1, 6)

    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_facecolor('whitesmoke')
    ax1.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['back_x'], bins=100, color=colors[0], zorder=4)
    ax1.set_title('back_x', color='white')
    ax1.tick_params(axis='both', colors='white')  


    ax2 = plt.subplot(gs1[0, 1])
    ax2.set_facecolor('whitesmoke')
    ax2.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['back_y'], bins=100, color=colors[1], zorder=4)
    ax2.set_title('back_y', color='white')
    ax2.tick_params(axis='both', colors='white')


    ax3 = plt.subplot(gs1[0, 2])
    ax3.set_facecolor('whitesmoke')
    ax3.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['back_z'], bins=100, color=colors[2], zorder=4)
    ax3.set_title('back_z', color='white')
    ax3.tick_params(axis='both', colors='white')

    ax4 = plt.subplot(gs1[0, 3])
    ax4.set_facecolor('whitesmoke')
    ax4.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['thigh_x'], bins=100, color=colors[3], zorder=4)
    ax4.set_title('thigh_x', color='white')
    ax4.tick_params(axis='both', colors='white')

    ax5 = plt.subplot(gs1[0, 4])
    ax5.set_facecolor('whitesmoke')
    ax5.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['thigh_y'], bins=100, color=colors[4], zorder=4)
    ax5.set_title('thigh_y', color='white')
    ax5.tick_params(axis='both', colors='white')

    ax6 = plt.subplot(gs1[0, 5])
    ax6.set_facecolor('whitesmoke')
    ax6.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    plt.hist(df['thigh_z'], bins=100, color=colors[4], zorder=4)
    ax6.set_title('thigh_z', color='white')
    ax6.tick_params(axis='both', colors='white')



    fig2 = plt.figure(figsize=(12, 3))
    fig2.set_facecolor('#131516')
    gs2 = gridspec.GridSpec(1, 6)

    # # QQ-Plots
    ax7 = plt.subplot(gs2[0, 0])
    ax7.set_facecolor('whitesmoke')
    ax7.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['back_x'], line='s', ax=ax7, color=colors[0], zorder=2)
    ax7.set_title('back_x QQ-Plot', color='white')
    ax7.tick_params(axis='both', colors='white')
    ax7.set_ylabel('Sample Quantiles', color='white')
    ax7.set_xlabel('Theoretical Quantiles', color='white')

    ax8 = plt.subplot(gs2[0, 1])
    ax8.set_facecolor('whitesmoke')
    ax8.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['back_y'], line='s', ax=ax8, color=colors[1], zorder=2)
    ax8.set_title('back_y QQ-Plot', color='white')
    ax8.tick_params(axis='both', colors='white')
    ax8.set_ylabel('Sample Quantiles', color='white')
    ax8.set_xlabel('Theoretical Quantiles', color='white')

    ax9 = plt.subplot(gs2[0, 2])
    ax9.set_facecolor('whitesmoke')
    ax9.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['back_z'], line='s', ax=ax9, color=colors[2], zorder=2)
    ax9.set_title('back_z QQ-Plot', color='white')
    ax9.tick_params(axis='both', colors='white')
    ax9.set_ylabel('Sample Quantiles', color='white')
    ax9.set_xlabel('Theoretical Quantiles', color='white')

    ax10 = plt.subplot(gs2[0, 3])
    ax10.set_facecolor('whitesmoke')
    ax10.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['thigh_x'], line='s', ax=ax10, color=colors[3], zorder=2)
    ax10.set_title('thigh_x QQ-Plot', color='white')
    ax10.tick_params(axis='both', colors='white')
    ax10.set_ylabel('Sample Quantiles', color='white')
    ax10.set_xlabel('Theoretical Quantiles', color='white')

    ax11 = plt.subplot(gs2[0, 4])
    ax11.set_facecolor('whitesmoke')
    ax11.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['thigh_y'], line='s', ax=ax11, color=colors[4], zorder=2)
    ax11.set_title('thigh_y QQ-Plot', color='white')
    ax11.tick_params(axis='both', colors='white')
    ax11.set_ylabel('Sample Quantiles', color='white')
    ax11.set_xlabel('Theoretical Quantiles', color='white')

    ax12 = plt.subplot(gs2[0, 5])
    ax12.set_facecolor('whitesmoke')
    ax12.grid(True, linestyle='-', linewidth=0.8, color='lightgrey')
    qqplot(df['thigh_z'], line='s', ax=ax12, color=colors[4], zorder=2)
    ax12.set_title('thigh_z QQ-Plot', color='white')
    ax12.tick_params(axis='both', colors='white')
    ax12.set_ylabel('Sample Quantiles', color='white')
    ax12.set_xlabel('Theoretical Quantiles', color='white')

def boxplot(df,n):
    dup=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']

    for column in dup:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])#[0:n])
        plt.title(f'Box plot of {column}')
        plt.show()


def mean_and_std(data):
    df=data[['back_x', 'back_y', 'back_z','thigh_x','thigh_y','thigh_z']].copy()
    mean = df.mean()

    std_dev = df.std()

    plt.figure(figsize=(12, 8))
    plt.bar(mean.index, mean, yerr=std_dev, capsize=5, color='skyblue', alpha=0.7, label='Mean')
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.title('Mean and Standard Deviation of Dataset Features')
    plt.legend()
    plt.show()
