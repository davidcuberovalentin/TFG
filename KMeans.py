import misc
import numpy as np
import pathlib
import dask.array as da
from dask_ml.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.image as im
import sys
import evaluation as e



def clustered_pixels(x_fit, pixels):
    labels = x_fit.predict(pixels).compute()
    ## Creating RGB colors for each cluster (see colors.txt to replace depending on the experiment you want to do)
    cluster_centers_rgb = np.zeros([8,3,3], dtype=np.uint8)  #[num_clusters, num_eigen, num_channels]
    cluster_centers_rgb[0, :] = [255, 145, 0]  #Orange
    cluster_centers_rgb[1, :] = [0, 0, 0]  #Black
    cluster_centers_rgb[2, :] = [255, 255, 255]  #White
    cluster_centers_rgb[3, :] = [108, 59, 42]  #Dark brown
    cluster_centers_rgb[4, :] = [155, 155, 155]  #Grey
    cluster_centers_rgb[5, :] = [255, 0, 0]  #Red (blood pixels)
    cluster_centers_rgb[6, :] = [0, 255, 0]  #Green
    cluster_centers_rgb[7, :] = [128, 64, 0]  #Brown
    res = x_fit.cluster_centers_[labels]
    res_rgb = cluster_centers_rgb[labels]
    return res, res_rgb

def k_means(cube, num_eigen):
    C = da.cov(cube, rowvar=False, bias=True)
    print(np.shape(C))
    w, v = np.linalg.eig(C)
    ## Uncomment to see eigenvalues energy
    #plt.scatter(list(range(0, len(w))), w)
    #plt.show()
    ## Reducing dimensionality of the cube skipping 1st principal component
    w = w[1:4] #Uncomment if K-Means + OSP
    v = v[:, 1:4] #Uncomment if K-Means + OSP
    #w = w[1:num_eigen] #Uncomment if AMF, SAM, OSP, K-Means + AMF, K-Means + SAM
    #v = v[:, 1:num_eigen] #Uncomment if AMF, SAM, OSP, K-Means + AMF, K-Means + SAM
    lowdim_cube = np.matmul(cube, v)
    #lowdim_cube = np.matmul(cube - np.mean(cube), v) #Try this if with the previous does not work
    print("RDY TO K-MEANS")
    km = KMeans(n_clusters=8, random_state=1)
    x_fit = km.fit(lowdim_cube)  #We fit the centroids and labels.
    clust_pixels, clust_pixels_rgb = clustered_pixels(x_fit, lowdim_cube)
    return clust_pixels, clust_pixels_rgb, lowdim_cube

if __name__ == "__main__":
    ## Decide first between blood data or spruces data
    ## Loading blood data hypercube
    cube, wavelength = misc.load_cube(str(pathlib.Path(__file__).parent.resolve()) + "/HSImage", "1", 1, 25, False)
    ## Loading spruces data hypercube
    #cube, wavelength = misc.load_cube(str(pathlib.Path(__file__).parent.resolve()) + "/david", "2", 1, 25, False)
    height = len(cube[:, 1, 1])
    width = len(cube[1, :, 1])
    bands = len(cube[1, 1, :])
    cube = np.reshape(cube, (-1, bands)) #Reshaping hypercube to get 2D matrix (pixels number, bands)
    clust_pixels, clust_pixels_rgb, lowdim_cube = k_means(cube, 4)

    ## Code to visualize K-Means output:
    ## -Plotting scatterplot with the data samples, blood data samples and centroids
    ## -Plotting K-Means labelled image
    ideal_mask = im.imread('Ideal_Mask.png')
    blood_pixels_idx = np.array(np.where(ideal_mask == 1))
    blood_pixels_idx = blood_pixels_idx[0, :] * blood_pixels_idx[1, :]
    plt.scatter(lowdim_cube[:, 0], lowdim_cube[:, 1], marker='x', label='samples')
    plt.scatter(lowdim_cube[[blood_pixels_idx], 0], lowdim_cube[[blood_pixels_idx], 1], marker='.', color='red', label='blood samples')
    plt.scatter(clust_pixels[:, 0], clust_pixels[:, 1], marker='+', color='black', label='cluster centers')
    plt.title("K-Means in 2D")
    plt.legend()
    plt.show()
    final_im = np.zeros([height, width, 3], dtype=np.uint8)
    print(np.shape(final_im))
    pixel = 0
    for i in list(range(0, height)):
        for j in list(range(0, width)):
            final_im[i, j] = clust_pixels_rgb[pixel, 0]
            pixel += 1
    plt.imshow(final_im)
    plt.axis('off')
    #plt.savefig("K_Means_final_im_spruces_2eigen_v2.png", bbox_inches='tight', pad_inches=0)
    plt.show()

    ## Code to create mask from K-Means in blood data case without using detectors
    # clustering_mask = np.zeros((height, width))
    # F_score_arr = []
    # for i in list(range(0, height)):
    #     for j in list(range(0, width)):
    #         if np.array_equal(final_im[i, j], [255, 0, 0] ) or np.array_equal(final_im[i, j], [255, 145, 0]):
    #             clustering_mask[i, j] = 1
    # plt.imshow(clustering_mask, cmap='gray')
    # plt.axis('off')
    # #plt.savefig("Clustering_Mask_3kdim.png", bbox_inches='tight',pad_inches = 0)
    # plt.show() #Plotting mask created from taking only blood clusters
    # Prec, Rec, F_score, FPR = e.getPRF(clustering_mask) #Getting quantitative results of this mask
    # F_score_arr.append(F_score)
    # #np.savetxt("results_F1_KMeans.csv", F_score_arr, fmt='%s', delimiter=",")