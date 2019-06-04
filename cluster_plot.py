import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def plot_label_dist(cluster,palette='inferno',figsize=(10,5),flip_axes=False):
    labels = sorted(cluster.labels_.astype(float))
    Y = [labels.count(x) for x in set(labels)]
    X = list(set(labels))
    
    plt.figure(figsize=figsize)
    if flip_axes: sns.barplot(Y,X,palette=palette)
    else: sns.barplot(X,Y,palette=palette)
    plt.xlabel('Cluster label')
    plt.ylabel('# of member restaurants')

    plt.title('distribution across {} clusters'.format(len(set(labels))))
    

def plot_clusters(cluster
                  ,pc
                  ,text=False
                  ,n_names=10
                  ,centroids=False
                  ,figsize=(6,6)
                  ,multiple_plots=False):
    sns.set()
    pca0 = pc[:,0]
    pca1 = pc[:,1]
    pca2 = pc[:,2]
    
    if centroids: centroid_plot = cluster.cluster_centers_
    n = 2 if multiple_plots else 1
    for i,other_pc in enumerate([pca1,pca2][:n]):
        count=0
        
        plt.figure(figsize=figsize)
        plt.scatter(x=pca0,y=other_pc
                    ,c=kmeans.labels_.astype(float)
                    , s=50
                    , alpha=0.5)
        if centroids: plt.scatter(centroid_plot[:,0], centroid_plot[:,1], c='blue', s=50)
        
        #Add merchant name annontations
        if text:
            for n, (j, x, y) in enumerate(zip(names, pca0, other_pc)):
                if count>n_names: break
                if np.random.rand(1)[0]>0.5:
                    count+=1
                    
                    xytexts = [+3,-3,+5,-5,+7,-7]
                    xco = np.random.choice(xytexts); yco = np.random.choice(xytexts)

                    plt.annotate(j, xy=(x, y),xytext=(x+xco, y+yco),fontsize=10,
                        arrowprops=dict(facecolor=np.random.rand(3), shrink=0.05),)

        plt.title('component0 and component{}'.format(i))
        plt.show()

def plot_3d_clusters(clusters
                     ,pc
                     ,text=False
                     ,n_names=8):
    np.random.seed(5)

    X = pc[:,0]
    Y = pc[:,1]
    Z = pc[:,2]

    fignum = 1

    for name,est in clusters:
        count=0
        fig = plt.figure(fignum, figsize=(8, 6))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        ax.scatter(X, Y, Z,
                   c=est.labels_.astype(float), edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('component0')
        ax.set_ylabel('component1')
        ax.set_zlabel('component2')
        ax.set_title(name)
        ax.dist = 12
        fignum = fignum + 1
        
    
    
    if text:
        for n, (j, x, y, z) in enumerate(zip(names, X,Y,Z)):
            if count>n_names: break
            if np.random.rand(1)[0]>0.5:
                count+=1

                xytexts = [+3,-3,+5,-5,+7,-7]
                xco = np.random.choice(xytexts); yco = np.random.choice(xytexts)

                plt.annotate(j, xy=(x, y),xytext=(x+xco, y+yco),fontsize=10,
                    arrowprops=dict(facecolor=np.random.rand(3), shrink=0.05),)

    fig.show()    
