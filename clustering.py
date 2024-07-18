from sklearn.cluster import MiniBatchKMeans,DBSCAN,Birch
from sklearn.metrics import silhouette_score
import pandas as pd



def minibatch_kmeans(data,labels,clusters):
        
    mbk = MiniBatchKMeans(init ='k-means++',max_iter=20, n_clusters = clusters,
                            batch_size = 512, n_init = 10,
                            max_no_improvement = 10, verbose = 0)
    mbk.fit(data)

    silhouette_score_val=silhouette_score(data, mbk.labels_)
    print(silhouette_score_val)
    

    df = pd.DataFrame({'real_labels': labels, 'clusters': mbk.labels_})

    return mbk.labels_,df,silhouette_score_val

def dbscan(data,labels,eps1):
   
    dbs = DBSCAN(n_jobs=-1,min_samples=50,eps=eps1)
    dbs.fit(data)

    silhouette_score_val=silhouette_score(data, dbs.labels_)
    print(f'DBSCAN silouette score --->{silhouette_score_val}')


    df = pd.DataFrame({'real_labels': labels, 'clusters': dbs.labels_})

    return dbs.labels_,df,silhouette_score_val

    
def birch(data,labels,clusters,th,bf):
   
    birch = Birch(n_clusters=clusters,threshold = th,branching_factor=bf)
    birch.fit(data)
    silhouette_score_val=silhouette_score(data, birch.labels_)
    print(f'BIRCH silouette score --->{silhouette_score_val}')

    df = pd.DataFrame({'real_labels': labels, 'clusters': birch.labels_})

    return birch.labels_,df,silhouette_score_val

def purity_metric(df):
    dominant_class_counts = df.groupby(level='clusters').max().sum()

    total_instances = df.sum()

    purity = dominant_class_counts / total_instances

    print(f"Purity: {purity}")

