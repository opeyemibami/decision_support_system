import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

def visualize_confidence_level(data, label, ylabel, title):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    grad_percentage = pd.DataFrame(
        data=data, columns=['Percentage'], index=label)
    ax = grad_percentage.plot(kind='barh', figsize=(
        7, 4.8), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off",
                   labelbottom="on", left="off", right="off", labelleft="on")

    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed',
                   alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel("Model Confidence Level(%)",
                  labelpad=2, weight='bold', size=12)
    ax.set_ylabel(ylabel, labelpad=10, weight='bold', size=12)
    ax.set_title(title, fontdict=None,
                 loc='center', pad=None, weight='bold')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    return


def pieChart(data, label, colors, title, startangle):
    y = data
    x = label
    percent = y

    predicted_class = np.argmax(data, axis=0)
    explode_list = [0]*len(x)
    explode_list[predicted_class] = 0.1
    explode = tuple(explode_list)

    # , 'grey', 'violet', 'magenta', 'cyan'
    patches, texts = plt.pie(
        y, explode=explode, colors=colors, shadow=True, startangle=startangle, radius=1.2)
    labels = ['{0} --> {1:1.2f} %'.format(i, j) for i, j in zip(x, percent)]

    # sort_legend = True
    # if sort_legend:
    #     patches, labels, dummy = zip(
    #         *sorted(zip(patches, x, y), key=lambda x: x[2], reverse=True))

    plt.legend(patches, labels, loc='center right',
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    plt.title(title, loc='center', pad=None, weight='bold')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    return
