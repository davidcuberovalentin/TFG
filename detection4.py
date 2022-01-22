import misc
import numpy as np
import pathlib
import dask.array as da
import matplotlib.pyplot as plt
import sys
import cv2
import KMeans as k
import evaluation as e


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def getBinIm(Y, th, height, width):
    Y_bin = np.zeros((height, width))
    Y_bin[Y<=th] = 0
    Y_bin[Y>th] = 1
    return Y_bin

######################################################### SAM DETECTOR
def extractMuandgetNormalized(spectral_vector):
    spectral_vector = spectral_vector - np.mean(spectral_vector)
    spectral_vector = spectral_vector / np.linalg.norm(spectral_vector)
    return spectral_vector

def whitening(cube_vec, num_eigen):
    C = da.cov(cube_vec,rowvar=True,bias=True)
    w, v = np.linalg.eig(C)
    ## Uncomment to see eigenvalues energy
    #plt.scatter(list(range(0, len(w))), w)
    #plt.title("Hyper-cube eigenvalues")
    #plt.xlabel("Eigenvalue number")
    #plt.ylabel("Value")
    #plt.show()
    ## Reducing dimensionality of the cube to remove noise
    w = w[0:num_eigen]
    v = v[:,0:num_eigen]
    print(np.shape(cube_vec))
    print(np.shape(v))
    diagw = np.diag(1 / (w ** 0.5))
    diagw = diagw.real.round(4)
    ## Whitening transform using PCA (Principal Component Analysis)
    wpca = np.dot(np.dot(diagw, v.T), cube_vec)
    ## Whitening transform using ZCA (Zero Component Analysis)
    wzca = np.dot(np.dot(np.dot(v, diagw), v.T), cube_vec)
    ## Uncomment to see data structure in first 2 bands before-after whitening
    # plt.scatter(cube_vec[0, :], cube_vec[1, :], marker='x', label="raw data")
    # plt.scatter(wpca[0, :], wpca[1, :], marker='+', label="whitened data") #PCA
    # #plt.scatter(wzca[0, :], wzca[1, :], marker='*') #ZCA
    # plt.title("Data structure in first 2 bands")
    # plt.xlabel("1st band")
    # plt.ylabel("2nd band")
    # plt.legend()
    # plt.show()
    return wpca


def preprocess_cube_SAM(cube, num_eigen):
    cube = extractMuandgetNormalized(cube)
    cube = whitening(cube, num_eigen)
    return cube


def detection_SAM(cube_w, t_m, t_n, im_height, im_width):
    Y = np.zeros((im_height, im_width))
    t = cube_w[:,(t_m*im_width) + t_n]  #Getting target pixel
    y = np.matmul(t.T, cube_w)  #Detector output values (normalized-cross correlation)
    pixel = 0
    for i in list(range(0, im_height)):
        for j in list(range(0, im_width)):
            Y[i, j] = y[pixel]  #Getting detected image
            pixel += 1
    maxvalue = np.amax(Y)
    maxidx = np.where(Y == np.amax(Y))
    print(maxvalue, maxidx) #Printing max value of detection and its index
    return Y

def detection_SAM_clust(cube_w, target_pixels_idx, im_height, im_width):
    Y = np.zeros((im_height, im_width))
    t = cube_w[:, len(cube_w[1, :]) - 1]    #Getting target pixel
    cube_w = cube_w[:, :-1] #Removing target pixel since the purpose was only to get it easily in the previous line having been preprocessed
    y = np.matmul(t.T, cube_w)  #Detector output values (normalized-cross correlation)
    pixel = 0
    pos = 0
    for i in list(range(0, im_height)):
        for j in list(range(0, im_width)):
            if pos <= len(target_pixels_idx)-1:
                if pixel == target_pixels_idx[pos]:
                    Y[i, j] = y[pos]    #Getting detected image
                    pos += 1
                pixel += 1
    maxvalue = np.amax(Y)
    maxidx = np.where(Y == np.amax(Y))
    print(maxvalue, maxidx) #Printing max value of detection and its index
    return Y
#################################################### AMF DETECTOR

def extractMu(spectral_vector):
    spectral_vector = spectral_vector - np.mean(spectral_vector)
    return spectral_vector

def getNormSquared(spectral_vector):
    return np.linalg.norm(spectral_vector)**2

def preprocess_cube_AMF(cube, num_eigen):
    cube = extractMu(cube)
    cube = whitening(cube, num_eigen)
    return cube

def detection_AMF(cube_w, t_m, t_n, im_height, im_width):
    Y = np.zeros((im_height, im_width))
    t = cube_w[:,(t_m*im_width) + t_n]  #Getting target pixel
    y = np.matmul(t.T, cube_w) #Detector output values (correlation with the matched filter 't')
    y = y / getNormSquared(t)
    pixel = 0
    for i in list(range(0, im_height)):
        for j in list(range(0, im_width)):
            Y[i, j] = y[pixel]  #Getting detected image
            pixel += 1
    maxvalue = np.amax(Y)
    maxidx = np.where(Y == np.amax(Y))
    print(maxvalue, maxidx) #Printing max value of detection and its index
    return Y

def detection_AMF_clust(cube_w, target_pixels_idx, im_height, im_width):
    Y = np.zeros((im_height, im_width))
    t = cube_w[:,len(cube_w[1,:])-1]    #Getting target pixel
    cube_w = cube_w[:, :-1] #Removing target pixel since the purpose was only to get it easily in the previous line having been preprocessed
    y = np.matmul(t.T, cube_w)  #Detector output values (correlation with the matched filter 't')
    y = y / getNormSquared(t)
    pixel = 0
    pos = 0
    for i in list(range(0, im_height)):
        for j in list(range(0, im_width)):
            if pos <= len(target_pixels_idx) - 1:
                if pixel == target_pixels_idx[pos]:
                    Y[i, j] = y[pos]    #Getting detected image
                    pos += 1
                pixel += 1
    maxvalue = np.amax(Y)
    maxidx = np.where(Y == np.amax(Y))
    print(maxvalue, maxidx) #Printing max value of detection and its index
    return Y
############################################### OSP Detector

def getPhi(cube_vec, num_eigen):
    C = da.cov(cube_vec, rowvar=True, bias=True)
    w, v = np.linalg.eig(C)
    ## Uncomment to see eigenvalues energy
    #plt.scatter(list(range(0,len(w))), w)
    #plt.show()
    ## Reducing dimensionality to predict better background subspace and save computations
    w = w[0:num_eigen]
    v = v[:, 0:num_eigen]
    return v

def getComplBackSub(v):
    dim = np.shape(v)
    I = np.identity(dim[0])
    P_orth = I - np.matmul(v, np.matmul(np.linalg.inv(np.matmul(v.T, v)), v.T))
    return P_orth

def preprocess_cube_OSP(cube, num_eigen):
    Phi = getPhi(cube, num_eigen)   #Reducing dimensionality using PCA
    P_orth = getComplBackSub(Phi)   #Getting complementary background subspace
    return cube, P_orth

def detection_OSP(cube, P_orth, t_m, t_n, im_height, im_width):
    Y = np.zeros((im_height, im_width))
    t = cube[:,(t_m*im_width) + t_n]    #Getting target pixel
    y = np.matmul(t.T, np.matmul(P_orth,cube))  #Detector output values (matching process in compl background subspace)
    pixel = 0
    for i in list(range(0, im_height)):
        for j in list(range(0, im_width)):
            Y[i, j] = y[pixel]  #Getting detected image
            pixel += 1
    maxvalue = np.amax(Y)
    maxidx = np.where(Y == np.amax(Y))
    print(maxvalue, maxidx) #Printing max value of detection and its index
    return Y

def detection_OSP_clust(cube, target_pixels_idx, P_orth, im_height, im_width):
    Y = np.zeros((im_height, im_width))
    t = cube[:, len(cube[1, :]) - 1]    #Getting target pixel
    cube = cube[:, :-1] #Removing target pixel since the purpose was only to get it easily in the previous line having been preprocessed
    y = np.matmul(t.T, np.matmul(P_orth,cube))  #Detector output values (matching process in compl background subspace)
    pixel = 0
    pos = 0
    for i in list(range(0, im_height)):
        for j in list(range(0, im_width)):
            if pos <= len(target_pixels_idx) - 1:
                if pixel == target_pixels_idx[pos]:
                    Y[i, j] = y[pos]
                    pos += 1
                pixel += 1
    maxvalue = np.amax(Y)
    maxidx = np.where(Y == np.amax(Y))
    print(maxvalue, maxidx)  #Printing max value of detection and its index
    return Y


if __name__ == "__main__":
    ## Decide first between blood data or spruces data
    ## Loading blood data hypercube
    #cube, wavelength = misc.load_cube(str(pathlib.Path(__file__).parent.resolve()) + "/HSImage", "1", 1, 25, False)
    ## Loading spruces data hypercube
    cube, wavelength = misc.load_cube(str(pathlib.Path(__file__).parent.resolve()) + "/david", "2", 1, 25, False)
    height = len(cube[:, 1, 1])
    width = len(cube[1, :, 1])
    bands = len(cube[1, 1, :])
    cube = np.reshape(cube, (-1,bands)) #Reshaping hypercube to get 2D matrix (pixels number, bands)

    ## Deciding target pixel and auxiliar if want to keep with more than a cluster
    #t_m = round(604 / 4)  #Mushroom pixel
    #t_n = round(1293 / 4)  #Mushroom pixel
    #t_m = round(1112/4) #Target pixel (blood)
    #t_n = round(2400/4) #Target pixel (blood)
    #t_m_k = round(1155/4) #Auxiliar target pixel (more blood)
    #t_n_k = round(2562/4) #Auxiliar target pixel (more blood)
    t_m = round(1622 / 4) #Spruce target
    t_n = round(104 / 4) #Spruce target
    t_m_k = round(1622 / 4)  #Spruce target
    t_n_k = round(104 / 4)  #Spruce target

    ## Deciding method (if decide using K-Means, take a look at KMeans.py)
    clustering = 1  #1:K-Means, other: only detection
    detection = 3   #1:SAM, 2:AMF, 3:OSP

    ## Deciding number of principal components to make experiments
    #num_eigen_arr = [2, 3, 4, 5, 6, 7]  #SAM & AMF detector experiments
    #num_eigen_arr = [1, 2, 3] #OSP Detector experiment
    #num_eigen_arr = [3, 4, 5, 6, 7, 8]  #Clustering SAM & AMF & OSP detector experiment
    num_eigen_arr = [2] #For spruces data (cannot be an array)

    ## Performance parameters
    ths = np.arange(0, 1, 0.02) #Array of thresholds to get binarized images and thus, performance curves
    #ths = [0.7]
    P = np.zeros((len(num_eigen_arr), len(ths)))    #Precision
    R = np.zeros((len(num_eigen_arr), len(ths)))    #Recall
    F1 = np.zeros((len(num_eigen_arr), len(ths)))   #F1-Score
    FPR_arr = np.zeros((len(num_eigen_arr), len(ths)))  #False Positive Rate

    for num_eigen_idx, num_eigen_val in enumerate(num_eigen_arr):
        ## Clustering step
        if clustering == 1:
            cube_all_clust, cube_rgb, lowdim_cube = k.k_means(cube, num_eigen_val)  #Getting pixels with centroid values and low-dimensional cube after reducing dimensionality
            target_clusters_idx = [] #Creating array with only pixel indices of clusters selected
            cube_clust = [] #Creating cube with only pixels of clusters selected
            pxl = 0
            ## Keeping with only the target clusters
            for i in range(0, len(cube_all_clust[:,1])):
                for j in range(0, len(cube_all_clust[1,:])):
                    if np.array_equal(cube_all_clust[i,j], cube_all_clust[t_m*width + t_n,j]) or np.array_equal(cube_all_clust[i,j], cube_all_clust[t_m_k*width + t_n_k,j]):
                        pxl = 1
                if pxl == 1:    #If the pixel in observation is in the same cluster as the target pixel
                    target_clusters_idx.append(i)  #We keep with its pixel index for the detection step
                    if detection == 3:
                        cube_clust.append(cube[i, :])  #Creating cube with only clusters selected, using the raw cube
                    else:
                        cube_clust.append(lowdim_cube[i,:]) #Creating cube with only clusters selected, using the low-dimensional cube
                    pxl = 0
            if detection == 3:
                cube_clust.append(cube[(t_m * width) + t_n, :])  #Adding target pixel in order to get it easily and preprocessed in the detectors
            else:
                cube_clust.append(lowdim_cube[(t_m * width) + t_n, :])  #Adding target pixel index in order to get it easily and preprocessed in the detectors
            cube_clust = np.transpose(cube_clust)
        cube = np.transpose(cube)

        ## Detection Step
        if detection == 1:
            if clustering == 1:
                cube_pre = preprocess_cube_SAM(cube_clust, num_eigen_val)   #Whitening and Reducing Dimensionality
                Y = detection_SAM_clust(cube_pre, target_clusters_idx, height, width) #Getting detected image
                cube = np.transpose(cube)
            else:
                cube_pre = preprocess_cube_SAM(cube, num_eigen_val)     #Whitening and Reducing Dimensionality
                Y = detection_SAM(cube_pre, t_m, t_n, height, width)    #Getting detected image
                cube = np.transpose(cube)
        elif detection == 2:
            if clustering == 1:
                cube_pre = preprocess_cube_AMF(cube_clust, num_eigen_val)   #Whitening and Reducing Dimensionality
                Y = detection_AMF_clust(cube_pre, target_clusters_idx, height, width)   #Getting detected image
                cube = np.transpose(cube)
            else:
                cube_pre = preprocess_cube_AMF(cube, num_eigen_val)     #Whitening and Reducing Dimensionality
                Y = detection_AMF(cube_pre, t_m, t_n, height, width)    #Getting detected image
                cube = np.transpose(cube)
        elif detection == 3:
            if clustering == 1:
                cube_pre, P_orth = preprocess_cube_OSP(cube_clust, num_eigen_val)   #Reducing Dimensionality and finding compl background subspace
                Y = detection_OSP_clust(cube_pre, target_clusters_idx, P_orth, height, width)   #Getting detected image
                cube = np.transpose(cube)
            else:
                cube_pre, P_orth = preprocess_cube_OSP(cube, num_eigen_val) #Reducing Dimensionality and finding compl background subspace
                Y = detection_OSP(cube_pre, P_orth, t_m, t_n, height, width)    #Getting detected image
                cube = np.transpose(cube)
        Y_spruces_idx = np.asarray(np.where(Y != 0))    #Getting selected target clusters indices for spruces data
        Y = NormalizeData(Y)    #Normalizing data to simplify evaluation method
        ## Showing detected image
        plt.imshow(Y, cmap='gray')
        plt.axis('off')
        #plt.savefig("Detected_spruces_OSP_clust_k3dim_2dim.png", bbox_inches='tight',pad_inches = 0)
        plt.show()

        ## Code to represent qualitative results of new data collection (spruces), uncomment if working with spruces data
        Y_spruces = np.zeros([height, width, 3], dtype=np.uint8)
        for i in range(0, len(Y_spruces_idx[0])):
                if Y[Y_spruces_idx[0,i],Y_spruces_idx[1,i]] >= 0.76:   #OSP->0.76, SAM/AMF->0.47
                    Y_spruces[Y_spruces_idx[0,i],Y_spruces_idx[1,i]] = [0, 255, 0]
                else:  #OSP->0.649, SAM/AMF->0
                    Y_spruces[Y_spruces_idx[0,i],Y_spruces_idx[1,i]] = [255, 0, 0]
        plt.imshow(Y_spruces)
        plt.axis('off')
        #plt.savefig("Detected_spruces_OSP_clust_k3dim_2dim.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        sys.exit()

        ## Applying evaluation method to get performance measures
        for th_idx, th_val in enumerate(ths):
            det_mask = getBinIm(Y, th_val, height, width)    #Binarizing image using th values
            ## Uncomment if you want to see a concrete binarized image
            # plt.imshow(det_mask, cmap='gray')
            # plt.axis('off')
            # plt.show()
            # sys.exit()
            print(num_eigen_idx, th_idx)
            Prec, Rec, F_score, FPR = e.getPRF(det_mask)    #Getting performance measures
            P[num_eigen_idx,th_idx] = Prec
            R[num_eigen_idx, th_idx] = Rec
            FPR_arr[num_eigen_idx, th_idx] = FPR
            F1[num_eigen_idx, th_idx] = F_score
    ## Save results into a csv file
    print(F1)
    #np.savetxt("results_OSP_8clust_v3.csv", F1, fmt='%s', delimiter=",")

    ## Plot PR-Curve (modify depending on the experiment)
    plt.plot(R[0,:], P[0,:], '.-', label="3 eigenvalues")
    plt.plot(R[1, :], P[1, :], '.-', label="4 eigenvalues")
    plt.plot(R[2, :], P[2, :], '.-', label="5 eigenvalues")
    #plt.plot(R[3, :], P[3, :], '.-', label="6 eigenvalues")
    #plt.plot(R[4, :], P[4, :], '.-', label="7 eigenvalues")
    #plt.plot(R[5, :], P[5, :], '.-', label="8 eigenvalues")
    #plt.title("SAM Detector: PR-Curves for different eigenvalues number")
    plt.title("K-Means (8 clusters) with OSP Detector: PR-Curves for different background subspace dimensions")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()

    ## Plot ROC (modify depending on the experiment)
    plt.plot(FPR_arr[0, :], R[0, :], '.-', label="3 eigenvalues")
    plt.plot(FPR_arr[1, :], R[1, :], '.-', label="4 eigenvalues")
    plt.plot(FPR_arr[2, :], R[2, :], '.-', label="5 eigenvalues")
    #plt.plot(FPR_arr[3, :], R[3, :], '.-', label="6 eigenvalues")
    #plt.plot(FPR_arr[4, :], R[4, :], '.-', label="7 eigenvalues")
    #plt.plot(FPR_arr[5, :], R[5, :], '.-', label="8 eigenvalues")
    #plt.title("SAM Detector: ROC for different eigenvalues number")
    plt.title("K-Means (8 clusters) with OSP Detector: ROC for different background subspace dimensions")
    plt.xlabel("FPR")
    plt.ylabel("Recall/TPR")
    plt.legend()
    plt.show()

    ## Plot F-Score (modify depending on the experiment)
    plt.plot(ths, F1[0, :], '.-', label="3 eigenvalues")
    plt.plot(ths, F1[1, :], '.-', label="4 eigenvalues")
    #plt.plot(ths, F1[2, :], '.-', label="5 eigenvalues")
    #plt.plot(ths, F1[3, :], '.-', label="6 eigenvalues")
    #plt.plot(ths, F1[4, :], '.-', label="7 eigenvalues")
    #plt.plot(ths, F1[5, :], '.-', label="8 eigenvalues")
    #plt.title("SAM Detector: F1-Score curve for different eigenvalues number")
    plt.title("K-Means (8 clusters) with OSP Detector: F1-Score for different background subspace dimensions")
    plt.xlabel("Threshold")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.show()


    ## Code to help to create ideal mask of the blood data
    #kernel = np.ones((2,2),np.uint8)
    #Y_bin = cv2.morphologyEx(Y_bin, cv2.MORPH_OPEN, kernel)
    #plt.imshow(Y_bin, cmap='gray')
    #plt.axis('off')
    #plt.figure(figsize=(width,height))
    #my_dpi = 149
    #plt.savefig('Binary_Mask.png', bbox_inches='tight', pad_inches=0, dpi=my_dpi)
    #plt.show()