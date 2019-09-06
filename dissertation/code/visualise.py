import pandas as pd
import numpy as np
import subprocess
import matplotlib
import json

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import patches as mpatches
from time import time

activity_labels = [
    'Lying',
    'Sitting',
    'Standing',
    'Walking',
    'Running',
    'Cycling',
    "Nor' walking",
    "Asc' stairs",
    "Des' stairs",
    'Vacuuming',
    'Ironing',
    'Rope jump'
]
text_width = 5.777
std_height = 5.777 / 2

output_folder = '/Users/gregwalsh/Google Drive/Study/Data Science Masters/Modules/Project/Output/PAMAP2'
# Linux Biolinum: https://www.dafont.com/linux-biolinum.font
# After downloading, update matplotlibs fonts by running:
# matplotlib.font_manager._rebuild()

matplotlib.rcParams['font.sans-serif'] = ['Linux Biolinum']
matplotlib.rcParams['mathtext.fontset'] = 'cm'

font_size = 9
dark_grey = '#3d3d3d'
light_grey = '#999999'
blue = '#83bbe5'
green = '#5abaa7'
red = '#ff8f80'
yellow = '#ffdf71'
pink = '#f5b5c8'

def plot_training_score(experiment_id, metric_key, double_col=False, height=std_height):
    with open('{}/{}/{}.json'.format(output_folder, 'scores', experiment_id)) as f:
        score_data = json.load(fp=f)
    colours = {'validation':blue, 'test':red, 'train': light_grey}
    y_axis_labels = {'f1m': r'$F_m$ Score', 'f1w': 'Weighted F1 Score', 'acc': 'Accuracy'}

    width = text_width if double_col else text_width / 2
    plt.figure(figsize=(width, height), frameon=False, dpi=133)
    for dataset_key, score_dict in score_data.items():
        y = score_dict[metric_key]
        x = range(1, len(y) + 1)
        plt.plot(x, y, label=dataset_key, color=colours[dataset_key], lw=1)

    axes = plt.gca()
    lims = axes.get_ylim()
    plt.plot([142, 142], [0, lims[1]], ':k', lw=1)
    axes.set_ylim(lims)

    plt.tick_params(labelsize=font_size-1)
    plt.xlabel('Training Epoch', fontsize=font_size)
    plt.ylabel(y_axis_labels[metric_key], fontsize=font_size)


    plt.legend(frameon=False, fontsize=font_size, loc=8)

    outpath = '{}/{}/{}.pdf'.format(output_folder, 'figures/training', experiment_id)
    plt.savefig(
        fname=outpath,
        format='pdf',
        bbox_inches='tight',
        dpi=133
    )
    subprocess.call(['open', outpath])

def plot_confusion_matrices(experiment_id, metric_key, labels):
    colourmap = LinearSegmentedColormap.from_list('blue_cmap', ['white', '#0c7cba'])

    for dataset_key in ['test']:#, 'train', 'validation']:
        confmat_data = np.loadtxt(
            fname='{}/{}/{}/{}_{}.csv'.format(
                output_folder,
                'confusion_matrices',
                metric_key,
                experiment_id,
                dataset_key
            ),
            delimiter=' '
        )

        # Normalise data
        confmat_data = confmat_data.astype('float') / confmat_data.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(text_width / 2, text_width / 2), frameon=False, dpi=133)
        plt.imshow(confmat_data, interpolation='nearest', cmap=colourmap)

        cutoff = 0.5
        for i in range(confmat_data.shape[0]):
            for j in range(confmat_data.shape[1]):
                if confmat_data[i, j] > cutoff:
                    colour = 'white'
                elif confmat_data[i, j] >= 0.05:
                    colour = "black"
                else:
                    colour = light_grey
                plt.text(
                    j,
                    i,
                    "{:0.1f}".format(confmat_data[i, j]) if confmat_data[i, j] >= 0.05 else 0,
                    fontsize = font_size,
                    horizontalalignment="center",
                    verticalalignment='center',
                    color=colour
                 )

        plt.xticks(np.arange(len(labels)), labels, rotation=-45, horizontalalignment='left')
        plt.yticks(np.arange(len(labels)), labels, rotation=-45, verticalalignment='bottom')
        plt.tick_params(labelsize=font_size-2)

        plt.ylabel('True Activity', fontweight='bold', fontsize=font_size-2)
        plt.xlabel('Predicted Activity', fontweight='bold', fontsize=font_size-2)

        outpath = '{}/{}/{}_{}_{}.pdf'.format(
            output_folder,
            'figures/confmat',
            experiment_id,
            dataset_key,
            metric_key
        )
        plt.savefig(
            fname=outpath,
            format='pdf',
            bbox_inches='tight',
            dpi=133
        )
        subprocess.call(['open', outpath])


def plot_cv_histogram(experiment_ids, metric_key, dataset, double_col=False, height=std_height):

    # Get top experiments
    df_all_results = pd.read_csv(
        filepath_or_buffer='{}/results/results.txt'.format(output_folder),
        sep='\t',
        usecols=['id', 'model', 'data hyperparams'],
        index_col=['id']
    )

    metric_col = '{} - {}'.format(dataset, metric_key)

    experiment_dfs = []
    for experiment_id in experiment_ids:
        experiment_model = df_all_results.loc[experiment_id, 'model']
        experiment_data_hyperparams = df_all_results.loc[experiment_id, 'data hyperparams']
        if "standardisation_method': 'minmax'" in experiment_data_hyperparams:
            std_method = 'minmax'
        elif "standardisation_method': 'meanstd'" in experiment_data_hyperparams or (
                not "standardisation_method':" in experiment_data_hyperparams
                and "ranked_standardisation': False" in experiment_data_hyperparams
        ):
            std_method = 'meanstd'
        elif "standardisation_method': 'ranked'" in experiment_data_hyperparams or (
                not "standardisation_method':" in experiment_data_hyperparams
                and "ranked_standardisation': True" in experiment_data_hyperparams
        ):
            std_method = 'ranked'

        filepath = '{}/results/{}/{}_{}.txt'.format(output_folder, std_method, experiment_id, experiment_model)

        df_experiment_results = pd.read_csv(
            filepath_or_buffer=filepath,
            sep='\t',
            usecols=['model', metric_col]
        )

        df_experiment_results['experiment_id'] = experiment_id
        df_experiment_results['std_method'] = std_method
        experiment_dfs.append(df_experiment_results)

    df_all_experiments = pd.concat(experiment_dfs)

    width = (text_width) if double_col else text_width / 2
    plt.figure(figsize=(width, height), frameon=False, dpi=133)

    group_by_cols = ['model', 'std_method']
    cmap = matplotlib.cm.get_cmap('viridis')
    grouped_dfs = df_all_experiments.groupby(group_by_cols)
    colours = [cmap(x) for x in np.linspace(start=0, stop=1, num=grouped_dfs.ngroups)]
    for i, ((model, std_method), df) in enumerate(grouped_dfs):
        label = '{} - {}'.format(model, std_method)
        plt.hist(x=df[metric_col], bins=20, density=True, range=(0, 1), label=label, color=colours[i], alpha=0.33, lw=0)

    x_axis_labels = {'f1m': 'Mean F1 Score', 'f1w': 'Weighted F1 Score', 'acc': 'Accuracy'}
    plt.xlabel(x_axis_labels[metric_key], fontweight='bold', fontsize=font_size)
    plt.ylabel('Frequency Density', fontweight='bold', fontsize=font_size)
    plt.tick_params(labelsize=font_size - 1)
    plt.legend(frameon=False)

    outpath = '{}/{}/{}.pdf'.format(output_folder, 'figures/crossval', int(time()))
    plt.savefig(
        fname=outpath,
        format='pdf',
        bbox_inches='tight',
        dpi=133
    )
    subprocess.call(['open', outpath])

def plot_timeseries(activity_ids, subject_ids, num_timesteps):

    start_timestep = 2500
    colours = [blue, red, yellow]
    df = pd.read_pickle('/Users/gregwalsh/project_data/PAMAP2_Dataset/Protocol/standardised.pk')
    nrows = len(subject_ids)
    ncols = len(activity_ids)
    fig, axes = plt.subplots(sharey=True, nrows=nrows, ncols=ncols, figsize=(text_width, 4/3 * nrows))


    for i, subject_id in enumerate(subject_ids):
        df_subject = df.loc[df['subject_id'] == subject_id, :]
        for j, activity_id in enumerate(activity_ids):
            # Get a user's data
            df_activity = df_subject.loc[df_subject['activity_id'] == activity_id, :]
            df_activity = df_activity.iloc[start_timestep:start_timestep + num_timesteps, :]
            ax = axes[i][j]
            ax.plot(
                range(0, num_timesteps),
                df_activity['ankle_acc6_2'],
                c=colours[i]
            )
            if i == 2 and j == 1:
                ax.set_xlabel('Timestep (ms)', fontsize=font_size)
            if i == 0:
                ax.set_title(activity_labels[activity_id], fontweight='bold', fontsize=font_size)
            if i == 1 and j == 0:
                ax.set_ylabel('Normalised Acceleration', fontsize=font_size)

            ax.tick_params(labelsize=font_size - 1)

    fig.subplots_adjust(hspace=0)

    patches = [mpatches.Patch(color=colour, label='Subject {}'.format(subject_id)) for subject_id, colour in zip(subject_ids, colours)]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.4), ncol=nrows, frameon=False, fontsize=font_size)
    outpath = '{}/{}/{}.pdf'.format(output_folder, 'figures/timeseries', int(time()))
    fig.savefig(
        outpath,
        format='pdf',
        bbox_inches='tight',
        dpi=100
    )
    subprocess.call(['open', outpath])



def plot_sensor_dist_histogram(activity_id, subject_ids, sensor_names):

        start_timestep = 2500
        colours = [blue, green, yellow]
        df = pd.read_pickle('/Users/gregwalsh/project_data/PAMAP2_Dataset/Protocol/imputed.pk')
        nrows = len(subject_ids)
        ncols = len(sensor_names)
        fig, axes = plt.subplots(sharey=True, nrows=nrows, ncols=ncols, figsize=(text_width, 4 / 3 * nrows))

        for i, subject_id in enumerate(subject_ids):
            df_subject = df.loc[(df['subject_id'] == subject_id) & (df['activity_id'] == activity_id), :]
            for j, sensor_name in enumerate(sensor_names):
                # Get a user's data
                df_activity = df_subject[sensor_name]
                ax = axes[i][j]
                ax.hist(
                    x=df_activity,
                    bins=[range(-34,23),range(-25,30),range(-25,25)][j],
                    color=colours[i],
                    edgecolor=colours[i],
                    alpha=1,
                    density=True
                )
                if i in [0, 1]:
                    ax.get_xaxis().set_visible(False)
                if i == 2 and j == 1:
                    ax.set_xlabel(r'Acceleration Readings ($m/s^2$)', fontsize=font_size)
                if i == 0:
                    title = {'hand_acc6_1':'Hand Acceleration X', 'hand_acc6_2':'Hand Acceleration Y', 'hand_acc6_3':'Hand Acceleration Z'}[sensor_name]
                    ax.set_title(title, fontweight='bold', fontsize=font_size)
                if i == 1 and j == 0:
                    ax.set_ylabel('Frequency Density', fontsize=font_size)
                ax.tick_params(labelsize=font_size - 1)

        fig.subplots_adjust(hspace=0)


        patches = [mpatches.Patch(color=colour, label='Subject {}'.format(subject_id)) for subject_id, colour in
                   zip(subject_ids, colours)]

        plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.45), ncol=nrows, frameon=False, fontsize=font_size)
        outpath = '{}/{}/{}.pdf'.format(output_folder, 'figures/sensor_hist', int(time()))
        fig.savefig(
            outpath,
            format='pdf',
            bbox_inches='tight',
            dpi=100
        )
        subprocess.call(['open', outpath])



def plot_f1m_histogram(double_col=True):

    # Get top experiments
    df = pd.read_csv(
        filepath_or_buffer='{}/results/processed_results.csv'.format(output_folder),
        sep=',',
        usecols=['model', 'std method', 'Is LSTM', 'test - f1m'],
        dtype={'model': str, 'std method': str}
    )

    df = df.loc[-df['Is LSTM'], :]
    df = df.loc[df['test - f1m'] > 0.05]
    colours = [green, yellow, red]
    rnn_colours = [blue, pink]
    width = (text_width) if double_col else text_width / 2
    fig, axes = plt.subplots(sharey=True, nrows=3, ncols=4, figsize=(text_width, text_width))

    ff_methods = ['popmeanstd', 'meanstd', 'ranked']
    rnn_methods = ['popminmaxsymmetric', 'ranked']
    std_methods = {'MLP':ff_methods, 'CNN':ff_methods, 'RNN':rnn_methods, 'CRNN':ff_methods}

    for i, model in enumerate(['MLP', 'CNN', 'CRNN', 'RNN']):
        for j, method in enumerate(std_methods[model]):

            ax = axes[j][i]
            ser = df['test - f1m'][(df['model'] == model) & (df['std method'] == method)]
            ax.hist(x=ser, bins=20, density=True, range=(0, 1), label='x',
                    color=colours[j] if model != 'RNN' else rnn_colours[j],
                    edgecolor=colours[j] if model != 'RNN' else rnn_colours[j],
                    alpha=1
            )

            # if j == 2 and i == 1:
            #     ax.set_xlabel('F1m Score', fontweight='bold', fontsize=font_size)

            ax.tick_params(labelsize=font_size - 1)
            if j == 0:
                ax.set_title(model, fontweight='bold', fontsize=font_size)
            if j == 1 and i == 0:
                ax.set_ylabel('Frequency Density', fontsize=font_size)

    axes[-1][-1].axis('off')
    fig.subplots_adjust(hspace=0)

    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.xlabel(r"$F_m$ Scores", fontsize=font_size)

    patches = [mpatches.Patch(color=colour, label='{}'.format(subject_id)) for subject_id, colour in
               zip(['population-normalisation', 'subject-normalisation', 'subject-normalisation-via-ranking'], colours)]
    patches += [mpatches.Patch(color=colour, label='{}'.format(subject_id)) for subject_id, colour in
               zip(['population-min-max-scaling', 'subject-scaling-via-ranking'], rnn_colours)]

    plt.legend(handles=patches, bbox_to_anchor=(0.90, -0.1), ncol=2, fontsize = font_size - 1, frameon=False)

    outpath = '{}/{}/{}.pdf'.format(output_folder, 'figures/f1m_hist', int(time()))
    plt.savefig(
        fname=outpath,
        format='pdf',
        bbox_inches='tight',
        dpi=133
    )
    subprocess.call(['open', outpath])

def plot_f1m_boxplots(normalise):

    # Get top experiments
    df = pd.read_excel(
        io='{}/results/f1m_scores_by_participant.xlsx'.format(output_folder),
        header=0

    )

    colours = [green, yellow, red]
    rnn_colours = [blue, pink]
    fig, axes = plt.subplots(sharey=True, nrows=1, ncols=4, figsize=(text_width, text_width * 0.5))

    ff_methods = ['population-normalisation', 'subject-normalisation', 'subject-normalisation-via-ranking']
    rnn_methods = ['population-min-max-scaling', 'subject-scaling-via-ranking']
    std_methods = {'MLP':ff_methods, 'CNN':ff_methods, 'RNN':rnn_methods, 'CRNN':ff_methods}

    for i, model in enumerate(['MLP', 'CNN', 'CRNN', 'RNN']):
        ax = axes[i]
        if normalise:
            ser = df.ix[:, 2:][(df['Architecture'] == model)] / df.ix[:, 2:].max()
        else:
            ser = df.ix[:, 2:][(df['Architecture'] == model)]
        bp = ax.boxplot(x=ser.values.T,
                        widths=0.1 * len(std_methods[model]),
                        showmeans=False,
                        patch_artist=True)

        means = np.mean(ser.values, axis=1)
        ax.plot([1 + x for x in range(len(std_methods[model]))], means, 'kx')
        ax.tick_params(labelsize=font_size - 1)
        for j, method in enumerate(std_methods[model]):
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                if len(bp[element]) > 0:
                    if element in ['whiskers', 'caps']:
                        plt.setp(bp[element][j * 2], color=colours[j] if model != 'RNN' else rnn_colours[j])
                        plt.setp(bp[element][(j* 2) + 1], color=colours[j] if model != 'RNN' else rnn_colours[j])
                    else:
                        plt.setp(bp[element][j], color=colours[j] if model != 'RNN' else rnn_colours[j])
                    if element == 'boxes':
                        bp[element][j].set_facecolor(colours[j] if model != 'RNN' else rnn_colours[j])
                        bp[element][j].set_alpha(0.5)

        ax.set_title(model, fontweight='bold', fontsize=font_size)
        if i == 0:
            ax.set_ylabel(r'Adjusted $F_m$ Score', fontsize=font_size)

        ax.tick_params(axis='x', bottom='off', labelcolor='none')

    fig.subplots_adjust(wspace=0)

    patches = [mpatches.Patch(color=colour, label='{}'.format(subject_id)) for subject_id, colour in
               zip(ff_methods, colours)]
    patches += [mpatches.Patch(color=colour, label='{}'.format(subject_id)) for subject_id, colour in
               zip(rnn_methods, rnn_colours)]

    plt.legend(handles=patches, bbox_to_anchor=(0.65, 0), ncol=2, fontsize = font_size-1, frameon=False)

    outpath = '{}/{}/{}_{}.pdf'.format(output_folder, 'figures/f1m_box', int(time()), str(normalise))
    plt.savefig(
        fname=outpath,
        format='pdf',
        bbox_inches='tight',
        dpi=133
    )
    subprocess.call(['open', outpath])

# plot_training_score('1535131664_660', 'f1m', double_col=True)
# for cof_matrix_id in ['1535370589_309', '1534777362_629']:
#     plot_confusion_matrices(experiment_id=cof_matrix_id, metric_key='f1m', labels=activity_labels)
# plot_cv_histogram(
#     experiment_ids=[
#         '1533326561_6', '1533336968_904', '1533345959_498', '1533342164_322', '1533362011_538',
#         '1533334609_300', '1533332741_227', '1533353665_431', '1533327924_624', '1533342987_585',
#     ],
#     metric_key='f1m',
#     dataset='val'
# )
#
# plot_f1m_boxplots(normalise=False)
plot_f1m_boxplots(normalise=True)


# plot_timeseries(activity_ids=[3, 4, 5], subject_ids=[5, 6, 7], num_timesteps=300)
# plot_sensor_dist_histogram(activity_id=4, subject_ids=[2, 5, 6], sensor_names=['hand_acc6_1', 'hand_acc6_2', 'hand_acc6_3'])
#

# plot_f1m_histogram()