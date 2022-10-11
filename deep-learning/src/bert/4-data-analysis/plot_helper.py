
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,
                               PercentFormatter)


def set_size(fig):
    """Setting the plot size."""
    fig.set_size_inches(8, 4)
    plt.tight_layout()


def set_style():
    """Setting the seaborn plot style."""
    sns.set_context("paper")
    sns.set(style="ticks", color_codes=True)


def set_labels(size=12, ylabel=""):
    """Setting axis labels."""
    plt.xlabel("", fontsize=size)
    plt.xticks(fontsize=size)
    plt.ylabel(ylabel, fontsize=size)
    plt.yticks(fontsize=size)


def plot_box(data, x, y):
    """Plotting a boxplot chart"""
    fig, ax = plt.subplots()

    set_style()
    set_size(fig)

    properties = {
        'boxprops': {'facecolor': 'white', 'edgecolor': 'darkgray'},
        'medianprops': {'color': 'darkgray'},
        'whiskerprops': {'color': 'darkgray'},
        'capprops': {'color': 'darkgray'}
    }

    sns.despine(offset=5)
    sns.boxplot(data=data,
                x=x,
                y=y,
                palette="Set2",
                ax=ax,
                **properties)

    #set_labels(ylabel='Balanced accuracy\n(50 folds)')

    ax.set_ylim(.4, .65)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    return fig, ax


def get_colors():
    return np.array([
        [0.7, 0.7, 0.7],
        [0.9, 0.9, 0.9],
        [0.984375, 0.7265625, 0],  
        [1, 1, 0.9],
        [1, 0.7, 0.3],  
        [0.6, 0.6, 0.6]                
    ])
    


def color_bars(axes, colors, best_classifier_bert, best_classifier_tfidf):
    for i in range(5):
        if best_classifier_bert == i:
            dark_color  = colors[2]
            light_color = colors[3]
        else:
            dark_color  = colors[0]
            light_color = colors[1]

        p1, p2 = axes[i].patches
        p1.set_color(dark_color)
        # p2.set_color(light_color)
        # p2.set_edgecolor(dark_color)
        # p2.set_hatch('////')

    for i in range(5):
        if best_classifier_tfidf == i:
            dark_color   = colors[2]
            light_color  = colors[3]
            border_color = colors[4] 
        else:
            dark_color   = colors[0]
            light_color  = colors[1]
            border_color = colors[5]

        p1, p2 = axes[i].patches
        # p1.set_color(dark_color)
        p2.set_color(light_color)
        p2.set_edgecolor(border_color)
        p2.set_hatch('////')


def make_labels(fig, axes, project, labels):
    for i, ax in enumerate(axes):
        ax.set_xticks([-.2, .2])
        ax.set_xticklabels(["bert", "tf-idf"], fontdict={'fontsize': 11})
        ax.set_xlabel(labels[i], fontsize=11)
        ax.set_ylabel("")
        ax.set_title("")
        for p in ax.patches:
            width = p.get_width()
            height = p.get_height()
            x, y = p.get_xy()
            ax.annotate(f'{height:.1%}', (x + width/2,
                                          y + height*1.02), ha='center')

    axes.flat[0].set_ylabel("balanced accuracy", fontsize=11)

    ticks_loc = axes[0].get_yticks().tolist()
    axes[0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    axes[0].set_yticklabels(['{:.0f}%'.format(x*100)
                             for x in axes[0].get_yticks()], fontdict={'fontsize': 11})

    axes[1].axes.yaxis.set_visible(False)
    axes[2].axes.yaxis.set_visible(False)
    axes[3].axes.yaxis.set_visible(False)
    axes[4].axes.yaxis.set_visible(False)

    sns.despine(offset=5)
    sns.despine(ax=axes[1], offset=5, left=True)
    sns.despine(ax=axes[2], offset=5, left=True)
    sns.despine(ax=axes[3], offset=5, left=True)
    sns.despine(ax=axes[4], offset=5, left=True)

    # fig.suptitle(
    #    f"Long-lived bug prediction performance for {project.capitalize()}: BERT versus TF-IDF features")


def plot_stacked_bar(data, classifiers):
 
    g = sns.FacetGrid(
        data,
        col="classifier",
        col_order=classifiers,
        sharex=False
    )

    g.map(
        sns.barplot,
        "classifier", "balanced_acc", "feature_extraction",
        hue_order=["bert", "tf-idf"],
        linewidth=1.25
    )

    axes = np.array(g.axes.flat)
    for ax in axes:
        ax.hlines(0.5, -0.5, 0.5, linestyle='--', linewidth=1, color="black")
        ax.set_ylim(0.2, 0.7)

    return plt.gcf(), axes


def plot_bar(data, best_position):
    plt.rcParams['axes.linewidth'] = 0.5
    sns.set_context("paper")
    sns.set(style="ticks", color_codes=True)

    data = data[['classifier', 'balanced_acc']].melt(id_vars=['classifier'])

    palete = get_colors()
    colors = []
    edges  = []
    for i in range(5):
        if i == best_position:
            colors.append(palete[2])
            edges.append(palete[2])
        else:
            colors.append(palete[0])
            edges.append(palete[0])

    data.sort_values('classifier').plot(
        'classifier', 'value', label="Balanced Accuracy", 
        kind="bar", legend=None, color=colors, edgecolor=edges
    )

    sns.despine(offset=5)

    ax = plt.axes()

    for p in ax.patches:
        width  = p.get_width()
        height = p.get_height()
        x, y   = p.get_xy()
        ax.annotate(f'{height:.1%}', (x + width/2,
                                      y + height*1.02), ha='center')

    plt.xlabel("", fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.ylabel('balanced accuracy', fontsize=12)
    plt.ylim(0.2, 0.7)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100)
                               for x in plt.gca().get_yticks()])
    left, right = plt.xlim()
    ax.hlines(0.5, left, right, linestyle='--', linewidth=1, color="black")
    #plt.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    return plt.gcf(), ax


def plot_pyramid(project, data, column, n, extractor='bert'):
    SMALL_SIZE = 12
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    top_g = data.groupby(column)[column].agg(['count'])
    top_s = top_g.sort_values('count', ascending=False).head(n)
    top_n = data.loc[data[column].isin(top_s.index)].groupby(
        column)['prediction_status'].value_counts(normalize=True).unstack()
    top_n.fillna(0, inplace=True)

    top_n.reset_index(inplace=True)
    top_n = top_n.sort_values('False Negative', ascending=True)


    fig, axes = plt.subplots(figsize=(10, 10), ncols=2, sharey=False, gridspec_kw={
                             'wspace': 0, 'hspace': 0})

    plt.rcParams["axes.grid.axis"] = "x"
    plt.rcParams["axes.grid"] = True
    #plt.grid(linestyle='--', linewidth=0.5, alpha=0.5)

    sns.despine(offset=0.5)

    colors = get_colors()
    if extractor == 'bert':
        fn_color = colors[0]
        tp_color = colors[2]
    else:
        fn_color = colors[0]
        tp_color = colors[2]

    bar_fn = sns.barplot(x="False Negative", y=column,
                         color=fn_color, ax=axes[0], data=top_n)
    bar_tp = sns.barplot(x="True Positive", y=column,
                         color=tp_color, ax=axes[1], data=top_n)

    if extractor == 'tfidf':
        for b in bar_fn.patches:
            b.set_hatch('////')

        for b in bar_tp.patches:
            # p2.set_color(light_color)
            # p2.set_edgecolor(dark_color)
            b.set_hatch('////')

    axes[0].set(xlim=(0, 1))
    axes[0].invert_xaxis()
    axes[0].spines['left'].set_visible(False)
    axes[0].tick_params(axis="x", labelsize=MEDIUM_SIZE)
    axes[0].tick_params(axis="y", labelsize=MEDIUM_SIZE)
    axes[0].set_ylabel(column.title().replace("_", " ").replace(
        "Category", "Level"), fontsize=MEDIUM_SIZE)
    axes[0].set_xlabel("False Negative", fontsize=MEDIUM_SIZE)
    axes[0].set_xticklabels(['{:,.0%}'.format(x)
                             for x in axes[0].get_xticks()])

    axes[1].set(xlim=(0, 1))
    axes[1].set_yticks([])
    axes[1].set(ylabel="")
    axes[1].tick_params(axis="x", labelsize=MEDIUM_SIZE)
    axes[1].set_xlabel("True Positive", fontsize=MEDIUM_SIZE)
    axes[1].set_xticklabels(['{:,.0%}'.format(x)
                             for x in axes[1].get_xticks()])

    # sns.despine(offset=0.5)
    s = axes[1].twinx()
    labels = [i.get_text() for i in axes[0].get_yticklabels()]
    labels_new = [top_s.loc[top_s.index == i, 'count'].values[0]
                  for i in labels]
    s.set_yticks(axes[0].get_yticks())
    s.set_ylim(axes[0].get_ylim())
    s.set_yticklabels(labels_new)
    s.tick_params(axis="y", labelsize=MEDIUM_SIZE)

    s.spines['top'].set_visible(False)
    s.spines['bottom'].set_visible(False)
    s.spines['right'].set_visible(False)
    s.spines['left'].set_visible(False)

    return top_n
'''
    Created by @myrthings 2/4/20
'''

import pandas as pd
import matplotlib.pyplot as plt


def catscatter(df,colx,coly,cols, topsx, color=['goldenrod', 'grey','black'],ratio=10,font='Helvetica',save=False,save_name='Default'):
    '''
    Goal: This function create an scatter plot for categorical variables. It's useful to compare two lists with elements in common.
    Input:
        - df: required. pandas DataFrame with at least two columns with categorical variables you want to relate, and the value of both (if it's just an adjacent matrix write 1)
        - colx: required. The name of the column to display horizontaly
        - coly: required. The name of the column to display vertically
        - cols: required. The name of the column with the value between the two variables
        - color: optional. Colors to display in the visualization, the length can be two or three. The two first are the colors for the lines in the matrix, the last one the font color and markers color.
            default ['grey','black']
        - ratio: optional. A ratio for controlling the relative size of the markers.
            default 10
        - font: optional. The font for the ticks on the matrix.
            default 'Helvetica'
        - save: optional. True for saving as an image in the same path as the code.
            default False
        - save_name: optional. The name used for saving the image (then the code ads .png)
            default: "Default"
    Output:
        No output. Matplotlib object is not shown by default to be able to add more changes.
    '''
    # Create a dict to encode the categeories into numbers (sorted)
    colx_codes=dict(zip(df[colx].sort_values().unique(),range(len(df[colx].unique()))))
    coly_codes=dict(zip(df[coly].sort_values(ascending=False).unique(),range(len(df[coly].unique()))))
    
    # Apply the encoding
    df[colx]=df[colx].apply(lambda x: colx_codes[x])
    df[coly]=df[coly].apply(lambda x: coly_codes[x])
    
    # Prepare the aspect of the plot

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(colx.replace("_", " ").title(), fontsize=20)
    plt.ylabel(coly.replace("_", " ").title(), fontsize=20)
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
    #plt.rcParams['font.sans-serif']=font
    plt.rcParams['xtick.color']=color[-1]
    plt.rcParams['ytick.color']=color[-1]
    plt.box(True)

    
    # Plot all the lines for the background
    for num in range(len(coly_codes)):
        plt.hlines(num,-1,len(colx_codes)+1,linestyle='dashed',linewidth=1,color=color[num%2+1],alpha=0.5)
    for num in range(len(colx_codes)):
        plt.vlines(num,-1,len(coly_codes)+1,linestyle='dashed',linewidth=1,color=color[num%2+1],alpha=0.5)
        
    # Plot the scatter plot with the numbers
    #plt.scatter(df[colx],
    #           df[coly],
    #           s=df[cols]*ratio,
    #           zorder=2,
    #           color=color[-1])
    
    key_list = list(colx_codes.keys())
    val_list = list(colx_codes.values())
    for x, y, s in zip(df[colx], df[coly], df[cols]):
            value  = '@'
            if x in val_list:   
                position  = val_list.index(x)
                value     = key_list[position]
        
            if value in topsx:
                plt.plot(x, y, color=color[0], alpha=.5, marker='*', markersize=s*ratio, linestyle='-', linewidth=5)
            else:
                plt.plot(x, y, color=color[1], alpha=.5, marker='o', markersize=s*ratio, linestyle='-', linewidth=5)
            
            plt.text(x-0.1, y-0.1, s, fontweight='bold')

    # Change the ticks numbers to categories and limit them
    plt.xticks(ticks=list(colx_codes.values()),labels=colx_codes.keys(),rotation=90)
    plt.yticks(ticks=list(coly_codes.values()),labels=coly_codes.keys())
    plt.xlim(xmin=-1,xmax=len(colx_codes))
    plt.ylim(ymin=-1,ymax=len(coly_codes))
    
    # Save if wanted
    if save:
        plt.savefig(save_name+'.png')


