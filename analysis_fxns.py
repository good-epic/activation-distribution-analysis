import numpy as np
import pandas as pd
import seaborn as sns
import ast
import scipy.stats as st
import matplotlib.pyplot as plt
import gc
import ast


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def predict_categories(model, le, X_new):
    """
    Make predictions with the fitted model
    
    Parameters:
    -----------
    model : LogisticRegression
        Fitted model from fit_multinomial_logistic
    le : LabelEncoder
        Fitted label encoder from fit_multinomial_logistic
    X_new : pandas.DataFrame
        New data with predictor columns
        
    Returns:
    --------
    predictions : array
        Predicted category combinations
    probabilities : array
        Probability estimates for each class
    """
    # Get predictions and probabilities
    pred_encoded = model.predict(X_new)
    pred_probs = model.predict_proba(X_new)
    
    # Convert predictions back to category combinations
    predictions = le.inverse_transform(pred_encoded)
    
def fit_subj_attr_predictor(df, num_pred_cols, cat_pred_cols, response_col, model_type="multinomial", impute=True):
    """
    Fit a multinomial logistic regression where the response is the cross of two categorical variables
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data
    predictor_columns : list of str
        Names of columns to use as predictors
    row_category : str
        Name of first categorical variable to cross
    col_category : str
        Name of second categorical variable to cross
    
    Returns:
    --------
    model : LogisticRegression
        Fitted model
    le : LabelEncoder
        Fitted label encoder for response variable
    """
    
    # Encode the response variable
    le = LabelEncoder()
    y = le.fit_transform(df[response_col])
    
    # Get predictor matrix
    X = pd.get_dummies(df[num_pred_cols + cat_pred_cols], columns=cat_pred_cols)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    if impute:
        # Get predictor matrix and one-hot encode categorical variables
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
    
    if model_type == "multinomial":
        # Fit the model
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        #model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,     # number of trees
            max_depth=None)       # max tree depth
            #random_state=42)      # for reproducibility
        model.fit(X_train, y_train)


    return model, le, X_train, y_train, X_test, y_test


def analyze_distribution_types(dfs: list[pd.DataFrame],  # Expecting like [gpt2s, gpt2m, p160, p410, p1b]
                               df_names: list[str]):     # Expecting like ['GPT2-Small', 'GPT2-Medium', 'Pythia-160M', 'Pythia-410M', 'Pythia-1B']
    # Distribution counts by model and layer
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, df in enumerate(dfs):
        sns.countplot(data=df, x='layer_num', hue='dist_name', ax=axes[i])
        axes[i].set_title(f"Distribution Types by Layer - {df_names[i]}")
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()



def create_heatmap(df, row_category, col_category, value_column, 
                   figsize=(10, 8), cmap='YlOrRd', normalize=True,
                   fmt='.2f', title=None, abs_val=False):
    """
    Create a heatmap from a DataFrame where values are aggregated by mean.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing the data
    row_category : str
        Column name for categories that will appear as rows
    col_category : str
        Column name for categories that will appear as columns
    value_column : str
        Column name containing the values to be plotted
    figsize : tuple, optional
        Figure size as (width, height)
    cmap : str, optional
        Color map to use for the heatmap
    fmt : str, optional
        Format string for cell values
    title : str, optional
        Title for the heatmap
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the heatmap
    ax : matplotlib.axes.Axes
        The axes containing the heatmap
    """
    
    # Pivot the data to create the heatmap matrix
    pivot_table = df.pivot_table(
        values=value_column,
        index=row_category,
        columns=col_category,
        aggfunc='mean'
    )
    if normalize:
        pivot_table /= df[value_column].std()
    
    if abs_val:
        pivot_table = pivot_table.abs()
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the heatmap
    sns.heatmap(
        pivot_table,
        annot=True,      # Show values in cells
        fmt=fmt,         # Format for values
        cmap=cmap,       # Color scheme
        cbar=True,       # Show color bar
        ax=ax
    )
    
    # Set title if provided
    if title:
        plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig, ax



def left_right_line_plots_with_variation(dfs, df_names, group_var, left_var, right_var, is_loc, title, 
                                       left_title, right_title, variation_type='bar_iqr', alpha=0.3,
                                       mean_condition_fxn=None, rotate_xticks=0):
    """
    variation_type options:
    - 'bar_iqr': Side-by-side bar chart with transparent IQR region
    - 'error_bars': Standard error bars showing IQR
    - 'ribbon': Continuous ribbon showing IQR between lines
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    max_v = -np.inf
    min_v = np.inf
    all_means = np.array([])
    
    # Calculate bar positions
    n_groups = len(dfs)
    bar_width = 0.8 / n_groups  # Bars will take up 80% of the space
    
    for ax, var in [(ax1, left_var), (ax2, right_var)]:
        # Get all unique x values across all dataframes
        all_x_values = set()
        for df in dfs:
            all_x_values.update(df[group_var].unique())
        x_positions = np.arange(len(all_x_values))
        x_values = sorted(list(all_x_values))
        
        for i, (df, label, color) in enumerate(zip(dfs, df_names, plt.cm.tab10(np.linspace(0, 1, len(dfs))))):
            grouped = df.groupby(group_var)[var]
            means = grouped.mean()
            q1 = grouped.quantile(0.25)
            q3 = grouped.quantile(0.75)
            
            # Calculate positions for this group's bars
            group_positions = x_positions + (i - n_groups/2 + 0.5) * bar_width
            
            if variation_type == 'bar_iqr':
                # Create dictionary to map x values to their positions
                x_to_pos = {x_val: pos for pos, x_val in zip(group_positions, x_values)}
                
                # Plot bars only for x values present in this dataset
                present_x = sorted(df[group_var].unique())
                bar_positions = [x_to_pos[x] for x in present_x]
                bar_means = [means[x] for x in present_x]
                
                # Create bars
                bars = ax.bar(bar_positions, bar_means, 
                            width=bar_width, label=label, alpha=0.7, color=color)
                
                # Add IQR as lighter colored extensions
                for x, mean in zip(present_x, bar_means):
                    pos = x_to_pos[x]
                    iqr_height = q3[x] - q1[x]
                    if mean >= 0:
                        ax.bar(pos, iqr_height, bottom=mean,
                              alpha=alpha, color=color, width=bar_width)
                    else:
                        ax.bar(pos, iqr_height, bottom=q3[x],
                              alpha=alpha, color=color, width=bar_width)
                
            elif variation_type == 'error_bars':
                present_x = sorted(df[group_var].unique())
                bar_positions = [x_to_pos[x] for x in present_x]
                bar_means = [means[x] for x in present_x]
                yerr_low = [means[x] - q1[x] for x in present_x]
                yerr_high = [q3[x] - means[x] for x in present_x]
                
                ax.errorbar(bar_positions, bar_means, 
                          yerr=[yerr_low, yerr_high],
                          label=label, color=color, fmt='o-', capsize=5)
                
            elif variation_type == 'ribbon':
                present_x = sorted(df[group_var].unique())
                bar_positions = [x_to_pos[x] for x in present_x]
                bar_means = [means[x] for x in present_x]
                q1_values = [q1[x] for x in present_x]
                q3_values = [q3[x] for x in present_x]
                
                ax.plot(bar_positions, bar_means, label=label, color=color, marker='o')
                ax.fill_between(bar_positions, q1_values, q3_values, 
                              alpha=alpha, color=color)
            
            # Collect values for y-axis limits
            if mean_condition_fxn is not None:
                valid_means = np.array([x for x, q1_val, q3_val in zip(means.values, q1.values, q3.values) 
                                      if mean_condition_fxn(x) and not np.isnan(x) and not np.isnan(q1_val) and not np.isnan(q3_val)])
                valid_q1 = np.array([x for x in q1.values if not np.isnan(x)])
                valid_q3 = np.array([x for x in q3.values if not np.isnan(x)])
                all_means = np.concatenate([all_means, valid_means, valid_q1, valid_q3])
            else:
                all_means = np.concatenate([all_means, 
                                          means.values[~np.isnan(means.values)],
                                          q1.values[~np.isnan(q1.values)],
                                          q3.values[~np.isnan(q3.values)]])
        
        # Set x-ticks to be in the middle of each group
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_values, rotation=rotate_xticks)
    
    # Set y-axis limits
    if not is_loc:
        max_v = np.abs(all_means).max()
        ax1.set_ylim(0, 1.05 * max_v)
        ax2.set_ylim(0, 1.05 * max_v)
    else:
        max_v = np.abs(all_means).max()
        min_v = np.abs(all_means).min()
        ax1.set_ylim(-0.95 * min_v, -1.05 * max_v)
        ax2.set_ylim(0.95 * min_v, 1.05 * max_v)
    
    ax1.set_title(left_title)
    ax2.set_title(right_title)
    for ax in [ax1, ax2]:
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()



def left_right_line_plots(dfs, df_names, group_var, left_var, right_var, is_loc, title, left_title, right_title, 
                         mean_condition_fxn=None, rotate_xticks=0, filter_condition=None):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(20, 7))
    
    max_v = -np.inf
    min_v = np.inf
    all_means = np.array([])
    for df, label in zip(dfs, df_names):
        # Apply filter if provided
        plot_df = df if filter_condition is None else df[filter_condition(df)]
        
        means_left = plot_df.groupby(group_var)[left_var].mean()
        means_right = plot_df.groupby(group_var)[right_var].mean()
        ax1.plot(means_left.index, means_left.values, label=label, marker='o')
        ax2.plot(means_right.index, means_right.values, label=label, marker='o')
        if mean_condition_fxn is not None:
            all_means = np.concatenate([all_means, 
                                        np.array([x for x in means_left.values if mean_condition_fxn(x) and not np.isnan(x)]),
                                        np.array([x for x in means_right.values if mean_condition_fxn(x) and not np.isnan(x)])])
        else:
            all_means = np.concatenate([all_means,
                                        np.array([x for x in means_left.values if not np.isnan(x)]),
                                        np.array([x for x in means_right.values if not np.isnan(x)])])
    
    if not is_loc:
        max_v = np.abs(all_means).max()
        ax1.set_ylim(0, 1.05 * max_v)
        ax2.set_ylim(0, 1.05 * max_v)
    else:
        max_v = np.abs(all_means).max()
        min_v = np.abs(all_means).min()
        ax1.set_ylim(-0.95 * min_v, -1.05 * max_v)
        ax2.set_ylim(0.95 * min_v, 1.05 * max_v)
    
    ax1.tick_params(axis='x', rotation=rotate_xticks)
    ax2.tick_params(axis='x', rotation=rotate_xticks)
    
    ax1.set_title(left_title)
    ax2.set_title(right_title)
    for ax in [ax1, ax2]:
        ax.legend()
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()



def analyze_modes(dfs: list[pd.DataFrame],  # Expecting like [gpt2s, gpt2m, p160, p410, p1b]
                  df_names: list[str]):     # Expecting like ['GPT2-Small', 'GPT2-Medium', 'Pythia-160M', 'Pythia-410M', 'Pythia-1B']
    # Number of modes by layer and model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Get max y value across all models for both left and right modes
    max_y = max([
        max(df.groupby('layer_num')['n_left_modes'].mean().max(),
            df.groupby('layer_num')['n_right_modes'].mean().max())
        for df in dfs
    ])

    for df, label in zip(dfs, df_names):
        means_left = df.groupby('layer_num')['n_left_modes'].mean()
        means_right = df.groupby('layer_num')['n_right_modes'].mean()
        
        ax1.plot(means_left.index, means_left.values, label=label, marker='o')
        ax2.plot(means_right.index, means_right.values, label=label, marker='o')
    
    # Set same y axis limits for both plots
    ax1.set_ylim(0, max_y)
    ax2.set_ylim(0, max_y)
    ax1.set_title('Average Number of Left Modes by Layer')
    ax2.set_title('Average Number of Right Modes by Layer')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    ###############################
    #### Loc/Spread Line Plots ####
    ###############################
    left_vars = ['left_most_mode_height', 'left_most_mode_loc', 'left_most_mode_volume', 
                 '2nd_left_most_mode_height', '2nd_left_most_mode_loc', '2nd_left_most_mode_volume',
                 '3rd_left_most_mode_height', '3rd_left_most_mode_loc']
    right_vars = ['right_most_mode_height', 'right_most_mode_loc', 'right_most_mode_volume', 
                  '2nd_right_most_mode_height', '2nd_right_most_mode_loc', '2nd_right_most_mode_volume',
                  '3rd_right_most_mode_height', '3rd_right_most_mode_loc']
    is_locs = [False, True, False, False, True, False, False, True]
    titles = ['First Most Mode Heights', 'First Most Mode Locations', 'First Most Mode Volumes',
              'Second Most Mode Heights', 'Second Most Mode Locations', 'Second Most Mode Volumes',
              'Third Most Mode Heights', 'Third Most Mode Locations']
    left_titles = ['Left Most Mode Height', 'Left Most Mode Locations', 'Left Most Mode Volumes',
                   'Second Left Most Mode Height', 'Second Left Most Mode Locations', 'Second Left Most Mode Volumes',
                   'Third Left Most Mode Height', 'Third Left Most Mode Locations']
    right_titles = ['Right Most Mode Height', 'Right Most Mode Locations', 'Right Most Mode Volumes',
                    'Second Right Most Mode Height', 'Second Right Most Mode Locations', 'Second Right Most Mode Volumes',
                    'Third Right Most Mode Height', 'Third Right Most Mode Locations']
    mcfs = [lambda x: x < 0.5, None, lambda x: x < 0.05, None, None, None, None, None]
    for left_var, right_var, is_loc, left_title, right_title, mcf in zip(left_vars, right_vars, is_locs, titles, left_titles, right_titles, mcfs):
        left_right_line_plots(dfs, df_names, group_var='layer_num', left_var=left_var, right_var=right_var,
                              is_loc=is_loc, title=title, left_title=left_title, right_title=right_title, mean_condition_fxn=mcf)
    
    ######################
    ##### Box Plots ######
    ######################
    # Mode heights and distances
    # Focus on first left and right modes for visualization
    max_height = [max(df['left_most_mode_height'][df['left_most_mode_height'] < 0.5].max(), df['right_most_mode_height'].max()) for df in dfs]
    max_abs_dist = [max(df['left_most_mode_loc'].abs().max(), df['right_most_mode_loc'].abs().max()) for df in dfs]
    min_abs_dist = [min(df['left_most_mode_loc'].abs().min(), df['right_most_mode_loc'].abs().min()) for df in dfs]

    max_height_2 = [max(df['2nd_left_most_mode_height'].max(), df['2nd_right_most_mode_height'].max()) for df in dfs]
    max_abs_dist_2 = [max(df['2nd_left_most_mode_loc'].abs().max(), df['2nd_right_most_mode_loc'].abs().max()) for df in dfs]
    min_abs_dist_2 = [min(df['2nd_left_most_mode_loc'].abs().min(), df['2nd_right_most_mode_loc'].abs().min()) for df in dfs]

    max_height_3 = [max(df['3rd_left_most_mode_height'].max(), df['3rd_right_most_mode_height'].max()) for df in dfs]
    max_abs_dist_3 = [max(df['3rd_left_most_mode_loc'].abs().max(), df['3rd_right_most_mode_loc'].abs().max()) for df in dfs]
    min_abs_dist_3 = [min(df['3rd_left_most_mode_loc'].abs().min(), df['3rd_right_most_mode_loc'].abs().min()) for df in dfs]
    
    for df, label, max_h, min_abs_d, max_abs_d, max_h_2, min_abs_d_2, max_abs_d_2, max_h_3, min_abs_d_3, max_abs_d_3 in \
            zip(dfs, df_names,
                max_height, min_abs_dist, max_abs_dist,
                max_height_2, min_abs_dist_2, max_abs_dist_2,
                max_height_3, min_abs_dist_3, max_abs_dist_3):
        
        # Heights of most modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='layer_num', y='left_most_mode_height', ax=axes[0])
        sns.boxplot(data=df, x='layer_num', y='right_most_mode_height', ax=axes[1])
        
        fig.suptitle(f'Most Mode Heights - {label}')
        axes[0].set_title('Left Most Mode Heights')
        axes[1].set_title('Right Most Mode Heights')
        
        if label == "Pythia-160M":
            max_h = 0.007
        axes[0].set_ylim(0, max_h)
        axes[1].set_ylim(0, max_h)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # Distances of most modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='layer_num', y='left_most_mode_loc', ax=axes[0])
        sns.boxplot(data=df, x='layer_num', y='right_most_mode_loc', ax=axes[1])
        
        fig.suptitle(f'Most Mode Distances - {label}')
        axes[0].set_title('Left Most Mode Distances')
        axes[1].set_title('Right Most Mode Distances')
        
        axes[0].set_ylim(-min_abs_d, -max_abs_d)
        axes[1].set_ylim(min_abs_d, max_abs_d)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # And similar for 2nd and 3rd modes...
        # Heights of 2nd most modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='layer_num', y='2nd_left_most_mode_height', ax=axes[0])
        sns.boxplot(data=df, x='layer_num', y='2nd_right_most_mode_height', ax=axes[1])
        
        fig.suptitle(f'2nd Most Mode Heights - {label}')
        axes[0].set_title('2nd Left Most Mode Heights')
        axes[1].set_title('2nd Right Most Mode Heights')
        
        axes[0].set_ylim(0, max_h_2)
        axes[1].set_ylim(0, max_h_2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # Distances of 2nd most modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='layer_num', y='2nd_left_most_mode_loc', ax=axes[0])
        sns.boxplot(data=df, x='layer_num', y='2nd_right_most_mode_loc', ax=axes[1])
        
        fig.suptitle(f'2nd Most Mode Distances - {label}')
        axes[0].set_title('2nd Left Most Mode Distances')
        axes[1].set_title('2nd Right Most Mode Distances')
        
        axes[0].set_ylim(-min_abs_d_2, -max_abs_d_2)
        axes[1].set_ylim(min_abs_d_2, max_abs_d_2)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # Heights of 3rd most modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='layer_num', y='3rd_left_most_mode_height', ax=axes[0])
        sns.boxplot(data=df, x='layer_num', y='3rd_right_most_mode_height', ax=axes[1])
        
        fig.suptitle(f'3rd Most Mode Heights - {label}')
        axes[0].set_title('3rd Left Most Mode Heights')
        axes[1].set_title('3rd Right Most Mode Heights')
        
        axes[0].set_ylim(0, max_h_3)
        axes[1].set_ylim(0, max_h_3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

        # Distances of 3rd most modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.boxplot(data=df, x='layer_num', y='3rd_left_most_mode_loc', ax=axes[0])
        sns.boxplot(data=df, x='layer_num', y='3rd_right_most_mode_loc', ax=axes[1])
        
        fig.suptitle(f'3rd Most Mode Distances - {label}')
        axes[0].set_title('3rd Left Most Mode Distances')
        axes[1].set_title('3rd Right Most Mode Distances')
        
        axes[0].set_ylim(-min_abs_d_3, -max_abs_d_3)
        axes[1].set_ylim(min_abs_d_3, max_abs_d_3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()



def analyze_with_distances_singles(dstn: pd.DataFrame,         # Distribution analysis results
                                   ccs:  list[pd.DataFrame]):  # Class correlation distances/attraction matrix, subject x attribute
    # Plot heatmaps for each layer
    for i, heat in enumerate(ccs):
        plt.figure(figsize=(10,8))
        sns.heatmap(heat, annot=True, cmap='Reds', center=0)
        plt.title(f'Layer {i}')
        plt.tight_layout()
        plt.show()
    
    # Analyze relationships between distances and distribution characteristics by layer
    # Get mode counts for each subject-attribute pair, by layer
    correlations = []
    for layer in range(dstn['layer_num'].max() + 1):
        layer_data = dstn[dstn['layer_num'] == layer]
        mode_counts = layer_data.groupby(['subject', 'attribute'])[['n_left_modes', 'n_right_modes']].mean()
        
        # Reshape mode counts to match heatmap shape
        mode_matrix_left = pd.pivot_table(mode_counts.reset_index(), 
                                          values='n_left_modes', 
                                          index='subject', 
                                          columns='attribute')
        
        # Correlation between distances and mode counts for this layer
        correlation = np.corrcoef(ccs[layer].values.flatten(), \
                                  mode_matrix_left.values.flatten())[0,1]
        correlations.append(correlation)
    
    # Plot correlations across layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(12), correlations, marker='o')
    plt.title('Correlation between Distances and Left Mode Counts by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.show()
    
    # Print strongest correlations
    print("\nStrongest correlations between distances and mode counts:")
    for layer, corr in enumerate(correlations):
        print(f"Layer {layer}: {corr:.3f}")



def analyze_with_distances_grid(dstn:     pd.DataFrame,         # Distribution analysis results
                                cc_heat:  list[pd.DataFrame]):  # Class correlation distances/attraction matrix, subject x attribute
    # Plot heatmaps for each layer
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, heat in enumerate(cc_heat):
        sns.heatmap(heat, annot=False, cmap='Reds', center=0, ax=axes[i])
        axes[i].set_title(f'Layer {i}')
    
    plt.tight_layout()
    plt.show()
    
    # Analyze relationships between distances and distribution characteristics by layer
    # Get mode counts for each subject-attribute pair, by layer
    correlations_left_n  = []
    correlations_right_n = []
    for layer in range(12):
        layer_data = gpt2s[gpt2s['layer_num'] == layer]
        mode_counts = layer_data.groupby(['subject', 'attribute'])[['n_left_modes', 'n_right_modes']].mean()
        
        # Reshape mode counts to match heatmap shape
        mode_matrix_left = pd.pivot_table(mode_counts.reset_index(), 
                                        values='n_left_modes', 
                                        index='subject', 
                                        columns='attribute')
        mode_matrix_right = pd.pivot_table(mode_counts.reset_index(), 
                                           values='n_right_modes', 
                                           index='subject', 
                                           columns='attribute')
        
        # Now calculate correlation
        df_temp_l = pd.DataFrame({'cc' : cc_heat[layer].values.flatten(),
                                  'mm' : mode_matrix_left.values.flatten()})
        df_clean_l = df_temp_l.dropna()
        correlation_left = df_clean_l['cc'].corr(df_clean_l['mm'])

        df_temp_r = pd.DataFrame({'cc' : cc_heat[layer].values.flatten(),
                                  'mm' : mode_matrix_right.values.flatten()})
        df_clean_r = df_temp_r.dropna()
        correlation_right = df_clean_r['cc'].corr(df_clean_r['mm'])

        correlations_left_n.append(correlation_left)
        correlations_right_n.append(correlation_right)
    
    # Plot correlations across layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(12), correlations_left_n, marker='o', label="N Left Modes Correlation")
    plt.plot(range(12), correlations_right_n, marker='o', label="N Right Modes Correlation")
    plt.title('Correlation between Distances and Mode Counts by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    
    # Analyze relationships between distances and distribution characteristics by layer
    # Get mode counts for each subject-attribute pair, by layer
    correlations_left_h  = []
    correlations_right_h = []
    for layer in range(12):
        layer_data = gpt2s[gpt2s['layer_num'] == layer]
        mode_heights = layer_data.groupby(['subject', 'attribute'])[['left_most_mode_height', 'right_most_mode_height']].mean()
        
        # Reshape mode counts to match heatmap shape
        mode_matrix_left = pd.pivot_table(mode_heights.reset_index(), 
                                        values='left_most_mode_height', 
                                        index='subject', 
                                        columns='attribute')
        mode_matrix_right = pd.pivot_table(mode_heights.reset_index(), 
                                           values='right_most_mode_height', 
                                           index='subject', 
                                           columns='attribute')
        # Now calculate correlation
        df_temp_l = pd.DataFrame({'cc' : cc_heat[layer].values.flatten(),
                                  'mm' : mode_matrix_left.values.flatten()})
        df_clean_l = df_temp_l.dropna()
        correlation_left = df_clean_l['cc'].corr(df_clean_l['mm'])

        df_temp_r = pd.DataFrame({'cc' : cc_heat[layer].values.flatten(),
                                  'mm' : mode_matrix_right.values.flatten()})
        df_clean_r = df_temp_r.dropna()
        correlation_right = df_clean_r['cc'].corr(df_clean_r['mm'])

        correlations_left_h.append(correlation_left)
        correlations_right_h.append(correlation_right)
    
    # Plot correlations across layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(12), correlations_left_h, marker='o', label="Left Most Mode Height Correlation")
    plt.plot(range(12), correlations_right_h, marker='o', label="Right Most Mode Height Correlation")
    plt.title('Correlation between Distances and Mode Heights by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()
    
    # Analyze relationships between distances and distribution characteristics by layer
    # Get mode counts for each subject-attribute pair, by layer
    correlations_left_l  = []
    correlations_right_l = []
    for layer in range(12):
        layer_data = gpt2s[gpt2s['layer_num'] == layer]
        mode_locs = layer_data.groupby(['subject', 'attribute'])[['left_most_mode_loc', 'right_most_mode_loc']].mean()
        
        # Reshape mode counts to match heatmap shape
        mode_matrix_left = pd.pivot_table(mode_locs.reset_index(), 
                                        values='left_most_mode_loc', 
                                        index='subject', 
                                        columns='attribute')
        mode_matrix_right = pd.pivot_table(mode_locs.reset_index(), 
                                           values='right_most_mode_loc', 
                                           index='subject', 
                                           columns='attribute')
        
        # Now calculate correlation. Some groups all NaN, so drop
        df_temp_l = pd.DataFrame({'cc' : cc_heat[layer].values.flatten(),
                                  'mm' : mode_matrix_left.values.flatten()})
        df_clean_l = df_temp_l.dropna()
        correlation_left = df_clean_l['cc'].corr(df_clean_l['mm'])

        df_temp_r = pd.DataFrame({'cc' : cc_heat[layer].values.flatten(),
                                  'mm' : mode_matrix_right.values.flatten()})
        df_clean_r = df_temp_r.dropna()
        correlation_right = df_clean_r['cc'].corr(df_clean_r['mm'])

        correlations_left_l.append(correlation_left)
        correlations_right_l.append(correlation_right)
    
    # Plot correlations across layers
    plt.figure(figsize=(10, 6))
    plt.plot(range(12), correlations_left_l, marker='o', label="Left Most Mode Location Correlation")
    plt.plot(range(12), correlations_right_l, marker='o', label="Right Most Mode Location Correlation")
    plt.title('Correlation between Distances and Mode Locationts by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()




def analyze_layer_evolution(dfs: list[pd.DataFrame],  # Expecting like [gpt2s, gpt2m, p160, p410, p1b]
                            df_names: list[str]):     # Expecting like ['GPT2-Small', 'GPT2-Medium', 'Pythia-160M', 'Pythia-410M', 'Pythia-1B']
    # Track statistical moments across layers
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    for df, label in zip(dfs, df_names):
        mean_std = df.groupby('layer_num')['std_hat'].mean()
        mean_skew = df.groupby('layer_num')['main_mode_density'].mean()  # Using density as proxy for skewness
        
        axes[0,0].plot(mean_std.index, mean_std.values, label=label, marker='o')
        axes[0,1].plot(mean_skew.index, mean_skew.values, label=label, marker='o')
        
    axes[0,0].set_title('Standard Deviation Evolution')
    axes[0,1].set_title('Main Mode Density Evolution')
    axes[0,0].legend()
    axes[0,1].legend()
    
    # Plot evolution of percentage beyond 2 and 3 standard deviations
    for df, label in zip(dfs, df_names):
        
        mean_p2 = df.groupby('layer_num')['pct_gt_p2std'].mean()
        mean_p3 = df.groupby('layer_num')['pct_gt_p3std'].mean()
        
        axes[1,0].plot(mean_p2.index, mean_p2.values, label=label, marker='o')
        axes[1,1].plot(mean_p3.index, mean_p3.values, label=label, marker='o')
    
    axes[1,0].set_title('Evolution of % > 2σ')
    axes[1,1].set_title('Evolution of % > 3σ')
    axes[1,0].legend()
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()
