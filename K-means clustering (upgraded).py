import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

time_1 = time.time()

# Generate data set to work with

@nb.njit
def generate_data(length, 
                  k):

    array = np.random.uniform(low = -1000, high = 1000, size = (length, k))
     
    return array

# Compute distance 

@nb.njit
def distance(coord1, 
             coord2):
    
    dst = np.sqrt(np.sum((coord1 - coord2) ** 2, axis = 1))
    
    return dst

# Helper function for K-means++

def random_choice_with_probabilities(n_samples, 
                                     probabilities):
    
    return np.random.choice(n_samples, p = probabilities)

# K-means++ centroids initialization

def kmeans_plus_plus_initialization(data_set, 
                                    k):
    
    n_samples, n_features = data_set.shape
    
    centroids = np.zeros((k, n_features))
    
    centroids[0] = data_set[np.random.randint(n_samples)]
    
    closest_distances = np.full(n_samples, np.inf)
    
    for i in range(1, k):
        
        distances = distance(data_set, centroids[i - 1])
        closest_distances = np.minimum(closest_distances, distances)
        
        probabilities = closest_distances ** 2
        probabilities /= np.sum(probabilities)
        
        next_centroid_idx = random_choice_with_probabilities(n_samples, probabilities)
        centroids[i] = data_set[next_centroid_idx]
        
    return centroids

# Assign the data points to chosen centroids

def assign_clusters(data_set, 
                    centroid_list):
    
    assignment = dict([tuple(centroid_list[idx]), list()] for idx in range(centroid_list.shape[0]))
    distances = np.zeros((centroid_list.shape[0], data_set.shape[0]))
    
    for index in range(centroid_list.shape[0]):
        
        distances[index] = distance(data_set, centroid_list[index])
        
    closest_distances = np.minimum.reduce(distances)
    
    for i in range (data_set.shape[0]):
        for j in range (distances.shape[0]):
            if closest_distances[i] == distances[j][i]:
                
                assignment[tuple(centroid_list[j])].append(data_set[i])
    
    return assignment

# Reassign data points with respect to weighted centroids selection 

def update_clusters(data_set, 
                    assignment):
    
    old_centroids = list(assignment.keys())
    new_centroids = np.zeros((len(old_centroids), len(old_centroids[0])))
    
    for index in range(len(old_centroids)):
        new_centroids[index] = np.mean(assignment[old_centroids[index]], axis = 0)
        
    new_assignment = assign_clusters(data_set, new_centroids)
    return new_assignment
    
# Compute variance in a particular group
    
def variance_within_cluster(key, 
                            item):
    
    var = np.sum(np.sum(np.sum((np.array(item) - np.array(key)) ** 2, axis = 1)))
    var /= var / np.array(item).shape[0]
    return var
    
# Sum of variances inside clusters
    
def k_means_variance(assignment):
    
    total = 0.0
    
    for key, item in assignment.items():
        total += variance_within_cluster(key, item)
        
    return total

# Compute variance between groups 

def variance_between_groups(data_set, 
                            assignment):
    
    data_center = np.mean(data_set, axis = 0)
    key_list = np.array(list(assignment.keys()))
    auxiliary_array = np.zeros((key_list.shape[0], 1))
    
    for idx in range(key_list.shape[0]):
        
        auxiliary_array[idx] = (np.sum((data_center - key_list[idx]) ** 2, axis = 0)) * (len(assignment[tuple(key_list[idx])]))
    
    variance = np.sum(auxiliary_array)
    return variance

# Define variance coefficient

def within_between_ratio(data_set, 
                         assignment):
    
    total_within = 0.0
    
    total_within += k_means_variance(assignment)
        
    sub_var = variance_between_groups(data_set, assignment)
    total_within = total_within / sub_var
    
    return total_within

# Implement all functions

data_set = generate_data(*list(map(int, input().split())))

k_range = np.arange(1, data_set.shape[0] // 25)
inertia_values = np.zeros(len(k_range))

# Elbow method

for element in k_range:
    
    centrs = kmeans_plus_plus_initialization(data_set, element)
    assgnmt = assign_clusters(data_set, centrs)
    
    for itr in range(25):
        
        res = update_clusters(data_set, assgnmt)
        assgnmt = res
    
    inertia_values[element - 1] = within_between_ratio(data_set, assgnmt)

time_2 = time.time()

plt.plot(k_range, inertia_values, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Close the current figure
plt.close()

time_3 = time.time()

number_of_clusters = int(input())

tracking_vessel = dict()

for iteration in range(25):
        
    centroid_list = kmeans_plus_plus_initialization(data_set, number_of_clusters)
    assignment = assign_clusters(data_set, centroid_list)
        
    for _ in range(50):
            
        result = update_clusters(data_set, assignment)
        assignment = result 
            
    tracking_vessel.setdefault(within_between_ratio(data_set, assignment), assignment)

min_variance = min(tracking_vessel.keys())
final_choice = tracking_vessel[min_variance]

time_4 = time.time()

# Plot the whole distribution

def random_color_generator():
        
    color = np.random.choice(list(mcolors.CSS4_COLORS.keys()))
    return color

if data_set.shape[1] == 2:
    
    fig = plt.figure(figsize = (10, 10))

    keys_list = list(final_choice.keys())
    
    for k in range(number_of_clusters):
            
        color_choice = random_color_generator()
        
        plt.scatter([final_choice[tuple(keys_list[k])][i][0] for i in range(len(final_choice[keys_list[k]]))], [final_choice[tuple(keys_list[k])][i][1] for i in range(len(final_choice[keys_list[k]]))], marker = "*", color = color_choice, s = 35)
        plt.scatter(np.array(keys_list[k][0]), np.array(keys_list[k][1]), marker = "*", color = color_choice, s = 75)
    
    # Label the plot
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Scatter Plot")   
    
    # Display the plot
    plt.show()
        
elif data_set.shape[1] == 3:
    
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection = '3d')

    keys_list = list(final_choice.keys())
    
    for k in range(number_of_clusters):
            
        color_choice = random_color_generator()
        
        ax.scatter([final_choice[tuple(keys_list[k])][i][0] for i in range(len(final_choice[keys_list[k]]))], [final_choice[tuple(keys_list[k])][i][1] for i in range(len(final_choice[keys_list[k]]))] , [final_choice[tuple(keys_list[k])][i][2] for i in range(len(final_choice[keys_list[k]]))] ,marker = "*", color = color_choice, s = 35)
        ax.scatter(np.array(keys_list[k][0]), np.array(keys_list[k][1]), np.array(keys_list[k][2]), marker = "*", color = color_choice, s = 75)
    
    # Label the plot    
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Scatter Plot")

    # Display the plot
    plt.show()

else:
    
    print(final_choice)
    
print(f"Time to execute code: {(time_2 - time_1) + (time_4 - time_3)}")
