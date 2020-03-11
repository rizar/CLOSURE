import os
import json
from matplotlib import pyplot
import pandas
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)

def load_log(root, file_, data_train, data_val, args, parts):
     slurmid = file_[:-8]
     path = os.path.join(root, file_)
     log = json.load(open(path))

     args[root][slurmid] = log['args']

     for i, t in enumerate(log['train_losses_ts']):
         data_train['root'].append(root)
         data_train['slurmid'].append(slurmid)
         data_train['step'].append(t)
         data_train['train_loss'].append(log['train_losses'][i])
         data_train['ts'].append(log['train_losses_ts'][i])
         data_train['entropy'].append(log['entropy'][i] if i < len(log.get('entropy', [])) else 0)
         data_train['prog_acc'].append(log['prog_acc'][i] if i < len(log.get('prog_acc', [])) else 0)

     part_keys = ['{}_accs'.format(part) for part in parts]
     for key in part_keys:
        if key in log:
            if len(log['val_accs_ts']) > len(log[key]):
                logger.warning('numbers of timesteps and values dont match for {}, {}, {}; attempt to fix'.format(
                    root, file_, key))
                log[key] = [None] * (len(log['val_accs_ts']) - len(log[key])) + log[key]
            if len(log['val_accs_ts']) < len(log[key]):
                raise ValueError()

     for i, t in enumerate(log['val_accs_ts']):
         data_val['root'].append(root)
         data_val['slurmid'].append(slurmid)
         data_val['step'].append(t)
         for key in part_keys:
             data_val[key[:-1]].append(log[key][i] if key in log else None)

def load_logs(root, data_train, data_val, args, parts=['train', 'val']):
    for root, dirs, files in os.walk(root):
        for file_ in files:
            if file_.endswith('pt.json'):
                load_log(root, file_, data_train, data_val, args, parts)


def plot_average(df, train_quantity='train_acc', val_quantity='val_acc', window=1, plot_interval=False):
    for root, df_root in df.groupby('root'):
        min_progress = min([df_slurmid['step'].max() for _, df_slurmid in df_root.groupby('slurmid')])
        df_root = df_root[df_root['step'] <= min_progress]
        df_agg = df_root.groupby(['step']).agg(['mean', 'std'])

        # Plot train
        train_values = df_agg[train_quantity]['mean']
        train_values = train_values.rolling(window).mean()
        train_lines = pyplot.plot(df_agg.index,
                                  train_values,
                                  label=root + ' train',
                                  linestyle='dotted')

        # Plot validation
        n_seeds = len(df_root['slurmid'].unique())
        if val_quantity:
            val_values = df_agg[val_quantity]['mean']
            val_std = df_agg[val_quantity]['std']
            val_values = val_values.rolling(window).mean()
            val_std = val_std.rolling(window).mean()
            width = val_std * stats.t.ppf(0.975, n_seeds - 1) / (n_seeds ** 0.5)
            pyplot.plot(df_agg.index,
                        val_values,
                        label=root + " val",
                        color=train_lines[0].get_color())
            if plot_interval:
                pyplot.fill_between(df_agg.index,
                                    val_values - width, val_values + width,
                                    color=train_lines[0].get_color(),
                                    alpha=0.5)

        # Count number of successes
        n_train_successes = 0
        n_val_successes = 0
        for slurmid, df_slurmid in df_root.groupby('slurmid'):
            slurmid_values = df_slurmid[train_quantity].rolling(window).mean()
            if slurmid_values.iloc[-1] > 0.99:
                n_train_successes += 1
            if val_quantity:
                slurmid_values = df_slurmid[val_quantity].rolling(window).mean()
                if slurmid_values.iloc[-1] > 0.99:
                    n_val_successes += 1
        success_report = "{} out of {}".format(n_train_successes, n_seeds)

        # Print
        to_print = ["{} ({} steps)".format(root, str(min_progress)),
                    success_report, "({:.1f})".format(100 * train_values.iloc[-1])]
        if val_quantity:
            to_print.append("{} out of {}".format(n_val_successes, n_seeds))
            to_print.append("({:.1f}+-{:.1f})".format(100 * val_values.iloc[-1], 100 * width.iloc[-1]))
        print(*to_print)

    pyplot.legend()
    quantities = ([train_quantity] if train_quantity else []) + ([val_quantity] if val_quantity else [])
    pyplot.title(", ".join(quantities))

def plot_all_runs(df, train_quantity='train_acc', val_quantity='val_acc', color=None, window=1, verbose=True):
    kwargs = {}
    if color:
        kwargs['color'] = color
    legend = []
    for (root, slurmid), df_run in df.groupby(['root', 'slurmid']):
        path = root + ' ' + slurmid
        train_lines = pyplot.plot(df_run['step'],
                                  df_run[train_quantity].rolling(window).mean(),
                                  label=path + ' train',

                                  linestyle='dotted',
                                  **kwargs)
        legend.append(slurmid)
        if val_quantity:
            pyplot.plot(df_run['step'],
                        df_run[val_quantity].rolling(window).mean(),
                        label=path + ' val',
                        color=train_lines[0].get_color())
            legend.append(slurmid)
        to_print = [path, df_run['step'].iloc[-1], df_run[train_quantity].iloc[-1]]
        if val_quantity:
            to_print.append(df_run[val_quantity].iloc[-1].mean())
        if verbose:
            print(*to_print)

    pyplot.legend(legend)
    quantities = ([train_quantity] if train_quantity else []) + ([val_quantity] if val_quantity else [])
    pyplot.title(", ".join(quantities))
