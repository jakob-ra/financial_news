import matplotlib.pyplot as plt

# plotting functions

def plot_barh(df, title, xlabel, ylabel='', fontsize=12, extend_x_axis=2):
    df.plot(kind='barh', figsize=(10, 7), fontsize=fontsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax = plt.gca()
    ax.bar_label(ax.containers[0], fmt='%.1f', label_type='edge', padding=3, fontsize=12)
    # extend x-axis to make labels visible
    ax.set_xlim(right=df.max() + extend_x_axis)
    plt.show()

# set rcParams

def rcParams():
    # reset rcParams
    plt.rcParams.update(plt.rcParamsDefault)

    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['font.size'] = 12

    # set theme
    plt.style.use('seaborn-darkgrid')
    # plt.style.use('ggplot')

    # tight layout
    plt.rcParams['figure.autolayout'] = True

    # set font
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Tahoma'

    # set color palette
    plt.rcParams['image.cmap'] = 'Set1'
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set1.colors)

rcParams()