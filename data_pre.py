import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import scipy.stats as stats
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

cols=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
def create_dataframe():
    folder_path='harth/'
    files = os.listdir(folder_path)
    dfs = []
    for file in files:
        df = pd.read_csv( os.path.join(folder_path, file))
        participant_id = os.path.splitext(file)[0]
        df['Participant_ID'] = participant_id
        if participant_id=='S020' or participant_id=='S006':
            df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda ts: ts.normalize())
            milliseconds_to_add = 20  
            for i in range(len(df)):
                df.at[i, 'timestamp'] = df.at[i, 'timestamp'] + pd.Timedelta(milliseconds=milliseconds_to_add * (i + 1))
    
        dfs.append(df)
    dataset=pd.concat(dfs)
    dataset.reset_index(drop=True, inplace=True)
    dataset.loc[dataset['label'] == 13, 'label'] = 9
    dataset.loc[dataset['label'] == 14, 'label'] = 10
    dataset.loc[dataset['label'] == 130, 'label'] = 11
    dataset.loc[dataset['label'] == 140, 'label'] = 12   
    return dataset


def window_data(time_step,overlap,df):
    
    step = time_step - overlap
    windows = []
    labels = []
    for i in range(0, len(df) - time_step, step):
        back_xs = df['back_x'].values[i: i + time_step]
        back_ys = df['back_y'].values[i: i + time_step]
        back_zs = df['back_z'].values[i: i + time_step]
        thigh_xs = df['thigh_x'].values[i: i + time_step]
        thigh_ys = df['thigh_y'].values[i: i + time_step]
        thigh_zs = df['thigh_z'].values[i: i + time_step]
        label = stats.mode(df['label'][i: i + time_step])[0]
        #print(f"{label} in lines --> {i}-{i+time_step}")
        windows.append([back_xs, back_ys, back_zs,thigh_xs,thigh_ys,thigh_zs])
        labels.append(label)
    return windows,labels    

def normal_data_Standard(df):
    scaler = StandardScaler()
    newdf=df.copy()
    newdf[cols] = scaler.fit_transform(newdf[cols])
    return newdf

def normal_data_MinMax(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    newdf=df.copy()
    newdf[cols] = scaler.fit_transform(newdf[cols])
    return newdf


def rolling_means(df,k=20):
    newdf=df.copy()
    newdf[cols]=newdf[cols].rolling(k).mean()
    new_df=newdf[k-1:].reset_index(drop=True)
    return new_df



def pca_test(df,components=3):
    pca = PCA(n_components=components)
    principal_components = pca.fit_transform(df)
    return principal_components

def get_balanced_dataset(df,k=7000):
    balanced_data = pd.DataFrame()
    for i in range(1,13):
        activity = df[df['label']== i].head(k).copy()
        balanced_data=pd.concat([balanced_data, activity])
   
    print(balanced_data.shape)
    return balanced_data


def correlation_analysis(df):
    print(f"{'RANDOM VARIABLES':<25}{'corr':<10}{'p-value':<10}\n")
    # Iterate over continuous features list
    for i in range(len(cols)):
        # Set output variable
        y_output = cols[i]
        # Loop through continuous features list again
        for j in range(len(cols)):
            # Set input variable
            x_input = cols[j]
            # Check for equality, if equal, skip iteration
            if y_output == x_input:
                continue
            # If not equal, grab coefficient and p-value
            else:
                corr, p_val = pearsonr(df[x_input], df[y_output])
                
                print(f"{y_output + '-vs-' + x_input:<25}"
                    f"{corr:<10.2f}{p_val:<10.8f}")
            

def descritize_by_bounds(df):
    num_stdv = 1
    df = df.copy()
    for col in df.columns:
        if col in cols:
            col_mean = df[col].mean()
            col_stdv = df[col].std()
            lower_bound = col_mean - col_stdv * num_stdv
            upper_bound = col_mean + col_stdv * num_stdv
            bins = [-float('inf'), lower_bound - 0.5 * col_stdv, lower_bound + 0.5 * col_stdv, 
                upper_bound - 0.5 * col_stdv, upper_bound + 0.5 * col_stdv, float('inf')]
            df[col] = pd.cut(df[col], bins=bins, labels=['very low', 'low', 'avg', 'high', 'very high'])

    return df

def remove_duplicates(df):
    dup=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z']
    dup2=['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z','label']
    dfr=df[dup2].copy()
    dfr.drop_duplicates(inplace=True)
    return dfr

def remove_outliersIQR(data,multiplier=1.5,quan1=0.25,quan3=0.75):
    df=data.copy()
    for column in cols:
        Q1 = df[column].quantile(quan1)
        Q3 = df[column].quantile(quan3)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        initial_count = df.shape[0]
        
        # Filter out the outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        final_count = df.shape[0]
        
        print(f"  Removed {initial_count - final_count} outliers")

    return df