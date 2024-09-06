import random
import numpy as np
import torch as tc
import Spotipy_code
import matplotlib.pyplot as plt

# define a function to normalize the audio features
def normalize(tensor):

    """
    This function normalizes the tensor for further computation
    """

    # use z-score normalization technique
    # compute the mean of the tensor along each column
    mean = tc.mean(tensor, dim=0)

    # compute the standard deviation of the tensor along each column
    st_dev = tc.std(tensor, dim=0)

    # avoid division by 0
    tc.where(st_dev == 0, tc.tensor(1.0), st_dev)

    # compute the z-score
    z_score = (tensor - mean)/st_dev

    return z_score

# define the k-means clustering
def k_means(k, tensor_of_features): 

    """
    This function implements the k-means clustering algorithm to cluster songs with similar features together.

    The number of clusters k must be specified, along with a Pytorch tensor containing the numerical data of song features. 

    The function then uses k-means++ to initialize initial centroids, and the optimizes the location of each centroid to form clusters.
    """
    
    # normalize the tensor_of_features
    tensor_of_features = normalize(tensor_of_features)

    # initialize tensor of centroids
    centroids = tc.zeros([k, tensor_of_features.size(1)])

    # implement k-means++
    # choose a random data point as initial centroid
    #random.seed(142)
    initial_centroid = tensor_of_features[random.randint(0, tensor_of_features.size(0)-1)]
    centroids[0] = initial_centroid

    # compute the distance of datapoints from the nearest centroid
    for i in range(1,k):    

        distances = tc.min(tc.cdist(tensor_of_features, centroids[:i]), dim=1)[0]
        probabilities = (distances ** 2 / tc.sum(distances ** 2)).numpy()
        new_centroid_index = np.random.choice(np.array(range(tensor_of_features.size(0))), p=probabilities)
        new_centroid = tensor_of_features[new_centroid_index]
        centroids[i] = new_centroid     

    # start the algorithm
    while True:

        # compute distances and label the data into clusters nearest to them
        distances = tc.cdist(tensor_of_features, centroids)
        labels = tc.argmin(distances, dim=1)
        
        # compute the mean of the data in a cluster, and 
        # set the mean as the new centroid for the cluster
        new_centroids = tc.zeros(k, tensor_of_features.size(1))
        for i in range(k):
            points_in_cluster = tensor_of_features[labels == i]
            if points_in_cluster.size(0) > 0: # if there is atleast 1 point in the cluster
                new_centroids[i] = points_in_cluster.mean(dim=0)
        
        # check if the centroids need any more adjusting
        if tc.equal(centroids, new_centroids):
            return new_centroids, tensor_of_features, labels
        else:
            centroids = new_centroids
        
# plot clusters for visualization purposes in the 2D plane. Only for 2 features
def plot_clusters(x_tens, y_tens, labels, k): 

    """
    This function can be used to visualize clusters in 2-dimensional space.
    Note: Can only input a Pytorch tensor of features with 2-dimensions.
    """

    # Define a list of colors manually
    color_list = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'cyan', 'magenta']

    # Ensure that the number of colors matches or exceeds the number of clusters
    if k > len(color_list):
        raise ValueError(f"Not enough colors defined for {k} clusters. Add more colors to the list.")
    
    # Plot each point with the corresponding color
    for i in range(len(x_tens)):
        cluster_label = labels[i].item()  # Convert tensor to a scalar
        plt.scatter(x_tens[i], y_tens[i], color=color_list[cluster_label], marker='o', label=f'Cluster {cluster_label}')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Clusters')

# use elbow method to determine the number of clusters needed
def elbow_method(tensor_of_features,k_max):

    """
    This function uses the Elbow method to plot a graph of WCSS against the number of clusters.
    This allows for determining the optimal number of clusters visually by looking for a 'bend' in the graph.
    The bend indicates that there is negligible change inertia after k exceed k_optimum.
    """

    # initialize WCSS values (within-cluster sum of squares)
    WCSS = tc.zeros(k_max)

    for k in range(1, k_max+1):

        # run the k-means algorithm
        centroids, tensor_of_features, labels = k_means(k, tensor_of_features)
        total_WCSS = 0.0

        for i,centroid in enumerate(centroids):
            
            # compute the WCSS
            cluster_points = tensor_of_features[labels == i]
            distances = tc.sum((cluster_points - centroid) ** 2, dim=1) # sum along rows of the tensor

            # sum the distances and add to the WCSS tensor
            total_WCSS += tc.sum(distances).item()

        # add the value to the WCSS tensor    
        WCSS[k-1] = total_WCSS
    
    # plot the WCSS against k
    plt.plot(np.array(range(1,k_max + 1)), WCSS.numpy(), 'bx-')
    plt.title('WCSS against k-clusters')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.show()

# use the silhouette method to determine optimal number of clusters
def silhouette_method(tensor_of_features, k_max):

    """
    Another function to determine the optimal number of clusters for k-means. 
    Computes the average silhouette scores of a tensor of features for varying k, and chooses the cluster k that has the largest average silhouette score.
    """

    average_silhouette_scores = []
    
    for k in range(3, k_max+1):

        # initialize average intra-cluster distance, and 
        # average inter-cluster distance
        a = tc.zeros(tensor_of_features.size(0))
        b = tc.zeros_like(a)

        # implement the k-means algorithm
        centroids, tensor_of_features, labels = k_means(k, tensor_of_features)
        
        # loop over the number of data points
        for i in range(tensor_of_features.size(0)):

            # find the average intra-cluster distance
            cluster_label = labels[i].item()
            cluster_points = tensor_of_features[labels == cluster_label]
            distance_a = tc.norm(tensor_of_features[i] - cluster_points, dim=1)
            a[i] = distance_a.mean()

            # find the clusters that the data point is not in
            other_labels = []
            average_distance = []
            for not_in in range(centroids.size(0)):
                if not_in != cluster_label:
                    other_labels.append(not_in)
            
            # find the distance to the nearest cluster 
            for label in other_labels:
                distance_b = tc.norm(tensor_of_features[i] - tensor_of_features[labels == label], dim=1)
                average_distance.append(distance_b.mean())
            b[i] = min(average_distance)
        
        s = (b - a) / tc.maximum(a, b)
        average_silhouette_scores.append(s.mean().item())

    # find the optimum number of clusters
    optimal_k = average_silhouette_scores.index(max(average_silhouette_scores)) + 3 # account for starting at k=3
    return optimal_k

# plot a spider plot to visualize how the data is spread
def spider_plot(k, tensor_of_features):

    """
    Generates a spider plot based on the features obtained from the features tensor
    """

    # use the silhouette method and k_means to get clusters
    # by running it 10 times and taking the mode for 7000 songs (as a start), 
    # k = 4 seems to be the optimal number of clusters
    _, tensor_of_features, labels = k_means(k, tensor_of_features)
    clusters = []

    # find the points in a cluster
    for i in range(k):
        cluster_points = tensor_of_features[labels == i]
        clusters.append(cluster_points)

    # list the features we are analyzing
    features = ['energy', 'instrumentalness', 'loudness', 'valence', 'acousticness']

    # number of features we are analyzing
    num_vars = len(features)

    # compute the spacing of the spider plot for each feature
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles.append(0.0) # close the circle of the spider plot
    
    # initialize radar plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))

    # list the colors to use
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'cyan', 'magenta']
    #labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

    # find the mean  values of each feature in each cluster
    for i, cluster in enumerate(clusters):

        # compute the mean of each feature
        mean_values = np.mean(cluster.numpy(), axis=0)
        mean_values = np.concatenate((mean_values, [mean_values[0]]))  # complete the loop

        # generate the plot
        ax.fill(angles, mean_values, color=colors[i], alpha=0.25)
        ax.plot(angles, mean_values, color=colors[i], linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)

    plt.title('Prominent Features Across Clusters', size=20)
    #plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.show()


if __name__ == "__main__":  
    example_features = Spotipy_code.get_audio_features(features_file="no_bad_songs.json")
    
    spider_plot(5, example_features)

    
   
    
    
    