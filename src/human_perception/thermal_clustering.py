#! /usr/bin/python3

#required packages
import rospy 
#import message_filters #to sync the messages
from sensor_msgs.msg import Image
from math import * #to avoid prefix math.
import numpy as np #to use matrix
from cv_bridge import CvBridge, CvBridgeError
import cv2

from sklearn.cluster import MeanShift, estimate_bandwidth
##########################################################################################
#Initializating cv_bridge
bridge = CvBridge()
visualization=True
#########################################################################################################################

class human_class:
    def __init__(self): #It is done only the first iteration
        #Variables to store temporally the info of human detectection at each time instant
        self.n_human=1 # considering up to 1 human to track initially
        self.centroid=np.zeros([self.n_human,2]) #x,y (pixels) of the human centroid, from camera
        self.image_size=[120,160] #initial condition of each camera
        self.camera_id=np.zeros([self.n_human,1]) #to know which camera detected the human
        self.image=np.zeros((120,160), np.uint8) #initial condition
        
    def image_callback(self,image_front):
        #print("DATA FROM CAMERA")
        try:
            #Front camera info extraction
            image_front = bridge.imgmsg_to_cv2(image_front, "mono8") #assuming grey scale thermal output
            ##################################################################################
            #######################################################################################
            [self.image,self.centroid,self.n_human,self.camera_id]=self.clustering(image_front,0) #front camera clustering
            print("NUM CLUSTER",self.n_human)
            self.image_size = self.image.shape
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

    #def param_callback(self,param_front):
    #    self.t_range[0]=param_front.min_range_val
    #    self.t_range[1]=param_front.max_range_val
    #    print("MIN",param_front.min_range_val)
    #    print("MAX",param_front.max_range_val)
    
    
################################################################################################################            
    def clustering(self,image,id):
        #print("clustering")
        min_samples_clust=100
        max_samples_clust=2000#
        size_ok=False
        occlusion=False
        no_signal=False
        filt_max=220 
        filt_min=30
        filt=100 #init
        n_iteration=4
        n=0
        while size_ok==False and occlusion==False and no_signal==False and n<n_iteration:
            n=n+1
            ret, thresh1 = cv2.threshold(image, filt, 255, cv2.THRESH_BINARY)
            X=np.argwhere(thresh1==255)
            print("SIZE",X.shape)
            print("FILT",filt)
            if X.shape[0]>min_samples_clust and X.shape[0]<max_samples_clust:
                size_ok=True
            elif X.shape[0]<=min_samples_clust:
                factor=(min_samples_clust-X.shape[0])/min_samples_clust
                filt=int(filt/(1+factor))
                if filt<filt_min:
                    no_signal=True #There is not thermal signal 
            elif X.shape[0]>=max_samples_clust:
                factor=(X.shape[0]-max_samples_clust)/X.shape[0]
                filt=int(filt*(1+factor))
                if filt>filt_max:
                    occlusion=True #There is a person just in front of the camera                   
             
        if X.shape[0]>min_samples_clust and X.shape[0]<max_samples_clust:
            bandwidth = estimate_bandwidth(X, quantile=0.05, n_samples=min_samples_clust)
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            
            labels_unique = np.unique(labels)
            n_clusters = len(labels_unique)

            if n_clusters>0:
                centroid=np.zeros([n_clusters,2])
                camera_id=np.zeros([n_clusters,1])
                for i in range(0,n_clusters):
                    centroid[i,0] = cluster_centers[i,1]
                    centroid[i,1] = cluster_centers[i,0]
                    camera_id[i,:]= id
                n_human=n_clusters
                #if n_clusters>1: #more than one cluster, second cluster round over the previous results
                #    min_samples_clust=10
                #    bandwidth = estimate_bandwidth(cluster_centers, quantile=0.3, n_samples=min_samples_clust)
                #    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                #    ms.fit(cluster_centers)
                #    labels = ms.labels_
                #    cluster_centers = ms.cluster_centers_
                #    labels_unique = np.unique(labels)
                #    n_clusters = len(labels_unique)
                #    if n_clusters>1:
                #        centroid=np.zeros([n_clusters,2])
                #        camera_id=np.zeros([n_clusters,1])
                #        for i in range(0,n_clusters):
                #            centroid[i,0] = cluster_centers[i,1]
                #            centroid[i,1] = cluster_centers[i,0]
                #            camera_id[i,:]= id
                #        n_human=n_clusters
                #    else: #one cluster
                #        centroid=np.zeros([1,2])
                #        camera_id=np.zeros([1,1])
                #        centroid[0,0] = cluster_centers[0,1]
                #        centroid[0,1] = cluster_centers[0,0]
                #        camera_id[0,:]= id
                #        n_human=1
                #else: #one cluster
                #    centroid=np.zeros([1,2])
                #    camera_id=np.zeros([1,1])
                #    centroid[0,0] = cluster_centers[0,1]
                #    centroid[0,1] = cluster_centers[0,0]
                #    camera_id[0,:]= id
                #    n_human=1
                
            else: #none cluster
                centroid=np.zeros([1,1])
                camera_id=id
                n_human=0
        elif X.shape[0]<=min_samples_clust:
            print("No human detected")
            centroid=np.zeros([1,1])
            camera_id=id
            n_human=0
        elif X.shape[0]>=max_samples_clust:
            print("Human occluding the camera")
            centroid=np.zeros([1,2])
            centroid[0,0]=self.image_size[1]/2 #assuming centroid in the center of the image
            centroid[0,1]=self.image_size[0]/2 #assuming centroid in the center of the image
            camera_id=id
            n_human=1
        return thresh1,centroid,n_human,camera_id
    
###############################################################################################
# Main Script

if __name__ == '__main__':
    # Initialize our node
    human=human_class()  
    rospy.init_node('human_detector_thermal',anonymous=True)
    # Setup and call subscription
    #Camara front
    rospy.Subscriber('/flir_module_driver/thermal/image_raw', Image,human.image_callback)
    #rospy.Subscriber('/flir_module_driver/thermal/temp_meas_range', TempMeasRange,human.param_callback)
    #rospy.spin()
    #Rate setup
    rate = rospy.Rate(1/0.01) # ROS publishing rate in Hz
    while not rospy.is_shutdown():	
        if visualization==True:
            n_human=human.n_human
            centroid=human.centroid
            image=np.zeros((120,160,3), np.uint8)
            image[:,:,0]=human.image
            image[:,:,1]=human.image
            image[:,:,2]=human.image
            for i in range(0,n_human):
                center_coordinates = (int(centroid[i,0]), int(centroid[i,1]))                        
                #print(center_coordinates)              
                image = cv2.circle(image, center_coordinates, 5, (0, 255,0), 10)
            cv2.imshow("Thermal clustering",image)
            cv2.waitKey(10)  
        