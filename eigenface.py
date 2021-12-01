"""========================================================================
Purpose:
    The purpose of this script is to perform the eigen face problem.
========================================================================"""
#=========================================================================#
# Preamble                                                                #
#=========================================================================#
#-------------------------------------------------------------------------#
# python packages                                                         #
#-------------------------------------------------------------------------#
import os
import sys
from subprocess import call
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from matplotlib import image
from PIL import Image
import matplotlib.image as mpimg
#-------------------------------------------------------------------------#
# Plot settings                                                           #
#-------------------------------------------------------------------------#
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams.update({'font.size': 18})
#=========================================================================#
# Main                                                                    #
#=========================================================================#
if __name__ == '__main__':
    #---------------------------------------------------------------------#
    # Main preamble                                                       #
    #---------------------------------------------------------------------#
    call(['clear'])
    sep         = os.sep
    pwd         = os.getcwd()
    data_path   = pwd + '%c..%cdata%c'              %(sep, sep, sep)
    media_path  = pwd +'%c..%cmedia%c'              %(sep, sep, sep)
    #---------------------------------------------------------------------#
    # Making media path                                                   #
    #---------------------------------------------------------------------#
    if os.path.exists(media_path) is False:
        os.mkdir(media_path)
    #---------------------------------------------------------------------#
    # Loading face data                                                   #
    #---------------------------------------------------------------------#
    face_data   = scipy.io.loadmat('/content/drive/MyDrive/Colab Notebooks/Faces.mat')
    faces       = face_data['faces']
    m           = int(face_data['m'])
    n           = int(face_data['n'])
    nfaces      = np.ndarray.flatten(face_data['nfaces'])
    #print('nfaces --->')
    #print(nfaces)
    #---------------------------------------------------------------------#
    # Plotting all the faces                                              #
    #---------------------------------------------------------------------#
    allPersons  = np.zeros((n*6,m*6))
    count       = 0
    for j in range(6):
        for k in range(6):
            allPersons[j*n:(j+1)*n, k*m:(k+1)*m] =\
                        np.reshape(faces[:,sum(nfaces[:count])],(m,n)).T
            count += 1
            #print(count)
    #---------------------------------------------------------------------#
    # Plotting all the faces                                              #
    #---------------------------------------------------------------------#
    img = plt.imshow(allPersons)
    #print("All Persons")
    #plt.show()
    img.set_cmap('gray')
    plt.axis('off')
    plt.savefig(media_path + 'all-faces.png')
    plt.close()
    #---------------------------------------------------------------------#
    # Plotting all the faces                                              #
    #---------------------------------------------------------------------#
    for person in range(len(nfaces)):
        subset      = faces[:,sum(nfaces[:person]):sum(nfaces[:(person+1)])]
        allFaces    = np.zeros((n*8,m*8))
        count       = 0
        for j in range(8):
            for k in range(8):
                if count < nfaces[person]:
                    allFaces[j*n:(j+1)*n,k*m:(k+1)*m] = \
                            np.reshape(subset[:,count],(m,n)).T
                    count += 1
        #-----------------------------------------------------------------#
        # Plotting all the faces individually                             #
        #-----------------------------------------------------------------#
        img = plt.imshow(allFaces)
        #if person in [7,23]:
        #  print('face --> %i'                         %(person+1))
        #  plt.show()
        img.set_cmap('gray')
        plt.axis('off')
        plt.savefig(media_path + 'face-%i.png'                   %(person))
        plt.close()
        
    #---------------------------------------------------------------------#
    # Training points using the first 36 faces                            #
    #---------------------------------------------------------------------#
    training_faces = faces[:, :np.sum(nfaces[:36])]

    #---------------------------------------------------------------------#
    # Plotting the average face                                           #
    #---------------------------------------------------------------------#
    avg_face = np.mean(training_faces, axis=1) # size n*m by 1

    #---------------------------------------------------------------------#
    # Computing the SVD                                                   #
    #---------------------------------------------------------------------#
    A = training_faces - np.tile(avg_face,(training_faces.shape[1], 1)).T
    U, S, VTranspose = np.linalg.svd(A, full_matrices=0)
    
    #---------------------------------------------------------------------#
    # Plotting different modes  here                                      #
    #---------------------------------------------------------------------#
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    img_avg = ax1.imshow(np.reshape(avg_face,(m,n)).T)
    img_avg.set_cmap('gray')
    plt.axis('off')

    ax2 = fig1.add_subplot(122)
    img_u1 = ax2.imshow(np.reshape(U[:,0],(m,n)).T)
    img_u1.set_cmap('gray')
    plt.axis('off')

    #print("comparing Average Face with first column of U")
    #plt.show()
    plt.close ()

    #---------------------------------------------------------------------#
    # Plotting U24 modes  here                                            #
    #---------------------------------------------------------------------# 
    img_u24 = plt.imshow(np.reshape(U[:,23],(m,n)).T)
    img_u24.set_cmap('gray')
    plt.axis('off')
    #print("24th mode of U")
    #plt.show()
    plt.close ()
    
    #---------------------------------------------------------------------#
    # Test face - 37                                                      #
    #---------------------------------------------------------------------#
    test_face = faces[:,np.sum(nfaces[:36])] # First face of person 37
    plt.imshow(np.reshape(test_face,(m,n)).T)
    plt.set_cmap('gray')
    plt.title('face-37')
    plt.axis('off')
    plt.show()
    plt.close ()
    #---------------------------------------------------------------------#
    # Different modes                                                     #
    #---------------------------------------------------------------------#
    test_faceMS = test_face - avg_face
    re_list = [25, 50, 100, 200, 400, 800, 1600]

    for mode in re_list:
        reconFace = avg_face + U[:,:mode]  @ U[:,:mode].T @ test_faceMS
        img = plt.imshow(np.reshape(reconFace,(m,n)).T)
        img.set_cmap('gray')
        plt.title('r = ' + str(mode))
        plt.axis('off')
        plt.show()
        plt.close ()
    
    #---------------------------------------------------------------------#
    # Test face - 38                                                      #
    #---------------------------------------------------------------------#
    test_face = faces[:,np.sum(nfaces[:37])] # First face of person 38
    plt.imshow(np.reshape(test_face,(m,n)).T)
    plt.set_cmap('gray')
    plt.title('face-38')
    plt.axis('off')
    plt.show()
    plt.close ()
    #---------------------------------------------------------------------#
    # Different modes                                                     #
    #---------------------------------------------------------------------#
    test_faceMS = test_face - avg_face
    re_list = [25, 50, 100, 200, 400, 800, 1600]

    for mode in re_list:
        reconFace = avg_face + U[:,:mode]  @ U[:,:mode].T @ test_faceMS
        img = plt.imshow(np.reshape(reconFace,(m,n)).T)
        img.set_cmap('gray')
        plt.title('r = ' + str(mode))
        plt.axis('off')
        plt.show()
        plt.close ()