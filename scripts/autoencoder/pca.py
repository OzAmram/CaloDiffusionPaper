from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import gaussian_kde

import sys
import numpy as np
import os 


if __name__ == "__main__" :

    # Pull Raw and Encoded Data From Files
    pull_path = './data_for_latent/raw_dataset.npz'
    raw_data_full = np.load(pull_path)

    pull_path2 = './data_for_latent/encoded_dataset.npz'
    encoded_data = np.load(pull_path2)

    encoded_data = encoded_data['arr_0']

    raw_data = raw_data_full['data'] #(100000, 1, 45, 16, 9)
    raw_E = raw_data_full['E']



    # Reshape Batches Raw Data
    batch_reshape = raw_data.reshape(100000, -1) #(100000, 6480)

    batch_reshape_stack = raw_data.reshape(4500000, -1) #4500000, 
  
    batch_reshape_radial = np.transpose(raw_data, (0, 1, 3, 2, 4))  #(100000, 1, 16, 45, 9)
    batch_reshape_radial = batch_reshape_radial.reshape(1600000, -1)
  
    batch_reshape_angular = np.transpose(raw_data, (0, 1, 4, 2, 3)) #(100000, 1, 9, 45, 16)
    batch_reshape_angular = batch_reshape_angular.reshape(900000, -1) #(900000, 45, 16)
 
    # Reshape Batches Encoded Data
    #encoded_data.shape = (3200000, 12, 4, 2)
    encoded_data = encoded_data.reshape(3200000, -1) 

    # Sample of Raw and Encoded Data
    sample_raw_data = batch_reshape[np.random.choice(batch_reshape.shape[0], 5000, replace=False)]
    sample_encoded_data = encoded_data[np.random.choice(encoded_data.shape[0], 5000, replace=False)]

    sample_raw_data_radial = batch_reshape_radial[np.random.choice(batch_reshape_radial.shape[0], 5000, replace=False)]
    sample_raw_data_stack = batch_reshape_stack[np.random.choice(batch_reshape_stack.shape[0], 5000, replace=False)]

############################
    #PCA Raw Data
    X = sample_raw_data 
    pca = PCA(n_components = 0.95)
    Xt = pca.fit_transform(X)

        #Plot 1st and 2nd PC againt each other
    x = Xt[:,0]
    y = Xt[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    plot = plt.scatter(x, y, c=z)
    plt.savefig('./pca_plots/raw_pca_plot.png')
    plt.close() 

        #Plot Variance Explained
    features = range(len(pca.explained_variance_))  
    print("Number of principal components to explain 95 percent of data (RAW)=", len(features))
    plt.figure()
    total_variance = np.sum(pca.explained_variance_)
    explained_variance = (pca.explained_variance_ / total_variance) * 100
    plot2 = plt.bar(features[:10], explained_variance[:10])
    plt.savefig('./pca_plots/raw_pca_variance.png')
    plt.close() 


        #Same as above but Scaled
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)
    x1 = Xt[:,0]
    y1 = Xt[:,1]
    z1 = Xt[:,2]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(xy1)
    plt.figure()
    plot = plt.scatter(x1, y1, c=z1)
    plt.savefig('./pca_plots/raw_scaled_pca_plot.png')

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
   
    ax.scatter(x1, y1, z1,  s=40)
    ax.set_xlabel("First PC", fontsize=14)
    ax.set_ylabel("Second PC", fontsize=14)
    ax.set_zlabel("Third PC", fontsize=14)
    plt.title("3D Raw PCA Plot", fontsize=16)
    plt.savefig('./pca_plots/raw_3d_pca_plot.png')

############################
    
    #PCA Raw Data (Stack)
    X = sample_raw_data_stack
    pca = PCA(n_components = 0.95)
    Xt = pca.fit_transform(X)

        #Plot 1st and 2nd PC againt each other
    x = Xt[:,0]
    y = Xt[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    plot = plt.scatter(x, y, c=z)
    plt.savefig('./pca_plots/raw_stack_pca_plot.png')
    plt.close() 

        #Plot Variance Explained
    features = range(len(pca.explained_variance_))  
    print("Number of principal components to explain 95 percent of data (Radial Raw) =", len(features))
    plt.figure()
    total_variance = np.sum(pca.explained_variance_)
    explained_variance = (pca.explained_variance_ / total_variance) * 100
    plot2 = plt.bar(features[:10], explained_variance[:10])
    plt.savefig('./pca_plots/raw_stack_pca_variance.png')
    plt.close() 


        #Same as above but Scaled
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)
    x1 = Xt[:,0]
    y1 = Xt[:,1]
    z1 = Xt[:,2]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(xy1)
    plt.figure()
    plot = plt.scatter(x1, y1, c=z1)
    plt.savefig('./pca_plots/raw_stack_scaled_pca_plot.png')


    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
   
    ax.scatter(x1, y1, z1,  s=40)
    ax.set_xlabel("First PC", fontsize=14)
    ax.set_ylabel("Second PC", fontsize=14)
    ax.set_zlabel("Third PC", fontsize=14)
    plt.title("3D Raw PCA Plot (Radial)", fontsize=16)
    plt.savefig('./pca_plots/raw_stack_3d_pca_plot.png')

############################

   #PCA encoded Data
    X = sample_encoded_data 
    pca = PCA(n_components = 0.95)
    Xt = pca.fit_transform(X)

        #Plot 1st and 2nd PC againt each other
    x = Xt[:,0]
    y = Xt[:,1]
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    plot = plt.scatter(x, y, c=z)
    plt.savefig('./pca_plots/encoded_pca_plot.png')
    plt.close() 

        #Plot Variance Explained
    features = range(len(pca.explained_variance_))  
    print("Number of principal components to explain 95 percent of data (Encoded)=", len(features))
    plt.figure()
    total_variance = np.sum(pca.explained_variance_)
    explained_variance = (pca.explained_variance_ / total_variance) * 100
    plot2 = plt.bar(features[:10], explained_variance[:10])
    plt.savefig('./pca_plots/encoded_pca_variance.png')
    plt.close() 


        #Same as above but Scaled
    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    Xt = pipe.fit_transform(X)
    x1 = Xt[:,0]
    y1 = Xt[:,1]
    z1 = Xt[:,1]
    xy1 = np.vstack([x1,y1])
    z1 = gaussian_kde(xy1)(xy1)
    plt.figure()
    plot = plt.scatter(x1, y1, c=z1)
    plt.savefig('./pca_plots/encoded_scaled_pca_plot.png')
    plt.close() 



    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
   
    ax.scatter(x1, y1, z1,  s=40)
    ax.set_xlabel("First PC", fontsize=14)
    ax.set_ylabel("Second PC", fontsize=14)
    ax.set_zlabel("Third PC", fontsize=14)
    plt.title("3D PCA Encoded Data Plot", fontsize=16)
    plt.savefig('./pca_plots/encoded_3d_pca_plot.png')





   
