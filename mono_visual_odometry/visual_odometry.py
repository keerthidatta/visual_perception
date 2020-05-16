
import os
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from m2bk2 import *
np.set_printoptions(threshold=sys.maxsize)


#np.random.seed(1)
#np.set_printoptions(threshold=np.nan)

#Define Feature descriptor:

#SURF, SIFT, ORB
feature_descriptor = "ORB"


dataset_handler = DatasetHandler()



def extract_features(image):  
    if feature_descriptor == "ORB":
        orb = cv2.ORB_create(nfeatures=1000)
        kp = orb.detect(image)
        kp, des = orb.compute(image, kp)    
    return kp, des


i = 0
image = dataset_handler.images[i]
kp, des = extract_features(image)
print("Number of features detected in frame {0}: {1}\n".format(i, len(kp)))
print("Coordinates of the first keypoint in frame {0}: {1}".format(i, str(kp[0].pt)))



def visualize_features(image, kp):
    display = cv2.drawKeypoints(image, kp, None)
    plt.imshow(display)


i = 0
image = dataset_handler.images_rgb[i]
visualize_features(image, kp)


def extract_features_dataset(images, extract_features_function):
    """
    Find keypoints and descriptors for each image in the dataset    
    """
    kp_list = []
    des_list = []
    
    for i in range(0, dataset_handler.num_frames):
        kp, des = extract_features_function(images[i])
        kp_list.append(kp)
        des_list.append(des)
    
    return kp_list, des_list


images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)

i = 0
print("Number of features detected in frame {0}: {1}".format(i, len(kp_list[i])))
print("Coordinates of the first keypoint in frame {0}: {1}\n".format(i, str(kp_list[i][0].pt)))


def match_features(des1, des2):
    """
    Match features from two images
    Returns:
    match -- list of matched features from two images. Each match[i] is k or less matches for the same query descriptor

    #Todo: flann matcher with orb
    #Todo: flann matcher with surf
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)
    match = sorted(matches, key = lambda x:x.distance)
    return match

i = 0 
des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(match)))



# Optional
def filter_matches_distance(match, dist_threshold):
    """
    Filter matched features from two images by distance between the best matches
    """
    filtered_match = []

    return filtered_match


i = 0 
des1 = des_list[i]
des2 = des_list[i+1]
match = match_features(des1, des2)

dist_threshold = 0.6
filtered_match = filter_matches_distance(match, dist_threshold)

print("Number of features matched in frames {0} and {1} after filtering by distance: {2}".format(i, i+1, len(filtered_match)))


def visualize_matches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)

n = 20
filtering = False

i = 0 
image1 = dataset_handler.images[i]
image2 = dataset_handler.images[i+1]

kp1 = kp_list[i]
kp2 = kp_list[i+1]

des1 = des_list[i]
des2 = des_list[i+1]

match = match_features(des1, des2)
if filtering:
    dist_threshold = 0.6
    match = filter_matches_distance(match, dist_threshold)

image_matches = visualize_matches(image1, kp1, image2, kp2, match[:n])    


def match_features_dataset(des_list, match_features):
    """
    Match features for each subsequent image pair in the dataset
    """
    matches = []
    for i in range(0, dataset_handler.num_frames-1):
        match = match_features(des_list[i], des_list[i+1])
        matches.append(match)
    return matches

matches = match_features_dataset(des_list, match_features)

i = 0
print("Number of features matched in frames {0} and {1}: {2}".format(i, i+1, len(matches[i])))


def filter_matches_dataset(filter_matches_distance, matches, dist_threshold):
    """
    Filter matched features by distance for each subsequent image pair in the dataset    
    """
    filtered_matches = []
    
    return filtered_matches

dist_threshold = 0.6

filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)

if len(filtered_matches) > 0:
    
    is_main_filtered_m = False
    if is_main_filtered_m: 
        matches = filtered_matches

    i = 0
    print("Number of filtered matches in frames {0} and {1}: {2}".format(i, i+1, len(filtered_matches[i])))



def estimate_motion(match, kp1, kp2, k, depth1=None):
    """
    Estimate camera motion from a pair of subsequent image frames     
    """
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    image1_points = []
    image2_points = []
    objectpoints = np.zeros((3, len(match)))
    i=0
    for m in match:
        query_idx = m.queryIdx
        train_idx = m.trainIdx
        x1, y1 = kp1[query_idx].pt
        image1_points.append([x1, y1])
        
        x2, y2 = kp2[train_idx].pt
        image2_points.append([x2, y2])
        
    E, mask = cv2.findEssentialMat(np.array(image1_points), np.array(image2_points), k)
    _, R, t, mask = cv2.recoverPose(E, np.array(image1_points), np.array(image2_points), k)

    rmat = R
    tvec = t    
    return rmat, tvec, image1_points, image2_points



i = 0
match = matches[i]
kp1 = kp_list[i]
kp2 = kp_list[i+1]
k = dataset_handler.k
#depth = dataset_handler.depth_maps[i]

rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k, depth1=None)

print("Estimated rotation:\n {0}".format(rmat))
print("Estimated translation:\n {0}".format(tvec))


i=30
image1  = dataset_handler.images_rgb[i]
image2 = dataset_handler.images_rgb[i + 1]

image_move = visualize_camera_movement(image1, image1_points, image2, image2_points)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)


image_move = visualize_camera_movement(image1, image1_points, image2, image2_points, is_show_img_after_move=True)
plt.figure(figsize=(16, 12), dpi=100)
plt.imshow(image_move)
# These visualizations might be helpful for understanding the quality of image points selected for the camera motion estimation

def estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=[]):
    """
    Estimate complete camera trajectory from subsequent image pairs
    """
    trajectory = [np.array([0, 0, 0])]
    
    ### START CODE HERE ###
    R = np.diag([1,1,1])
    T = np.zeros([3, 1])
    RT = np.hstack([R, T])
    RT = np.vstack([RT, np.zeros([1, 4])])
    RT[-1, -1] = 1
    
    file1 = open('traj.txt', 'w+')

    for i in range(len(matches)):     
        match = matches[i]
        kp1 = kp_list[i]
        kp2 = kp_list[i+1]
        #depth = depth_maps[i]
        
        rmat, tvec, image1_points, image2_points = estimate_motion(match, kp1, kp2, k)
        rt_mtx = np.hstack([rmat, tvec])
        rt_mtx = np.vstack([rt_mtx, np.zeros([1, 4])])
        rt_mtx[-1, -1] = 1
        print("rtmax", rt_mtx)
        
#         https://docs.opencv.org/3.4.3/d9/dab/tutorial_homography.html
        rt_mtx_inv = np.linalg.inv(rt_mtx)

        RT = np.dot(RT, rt_mtx_inv)
        new_trajectory = RT[:3, 3]

        print("new traj", new_trajectory)


        #print("trajectory {0}", i, new_trajectory,"\n")
        file1.write(str(new_trajectory)+"\n")
        trajectory.append(new_trajectory)
         
    
    trajectory = np.array(trajectory).T
    file1.close()
    
    return trajectory

#depth_maps = dataset_handler.depth_maps
#trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=None)

#i = 1
#print("Camera location in point {0} is: \n {1}\n".format(i, trajectory[:, [i]]))
#print("Length of trajectory: {0}".format(trajectory.shape[1]))

#dataset_handler = DatasetHandler()


# Part 1. Features Extraction
images = dataset_handler.images
kp_list, des_list = extract_features_dataset(images, extract_features)


# Part II. Feature Matching
matches = match_features_dataset(des_list, match_features)

is_main_filtered_m = False
if is_main_filtered_m:
    dist_threshold = 0.75
    filtered_matches = filter_matches_dataset(filter_matches_distance, matches, dist_threshold)
    matches = filtered_matches

    
# Part III. Trajectory Estimation
#depth_maps = dataset_handler.depth_maps
trajectory = estimate_trajectory(estimate_motion, matches, kp_list, k, depth_maps=None)


# Print Submission Info
#print("Trajectory X:\n {0}".format(trajectory[0,:].reshape((1,-1))))
#print("Trajectory Y:\n {0}".format(trajectory[1,:].reshape((1,-1))))
#print("Trajectory Z:\n {0}".format(trajectory[2,:].reshape((1,-1))))


#visualize_trajectory(trajectory)
