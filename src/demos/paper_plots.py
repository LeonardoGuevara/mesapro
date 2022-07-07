#! /usr/bin/python3

#required packages
import rospy
import message_filters #to sync the messages
import geometry_msgs.msg
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
import numpy as np #to use matrix
import cv2
import yaml
import threading # Needed for Timer
from mesapro.msg import human_msg, hri_msg, robot_msg
import ros_numpy
##########################################################################################
#Importing global parameters from .yaml file
config_direct=rospy.get_param("/hri_visualization/config_direct")
a_yaml_file = open(config_direct+"global_config.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
##############################################################################################################################
#CAMERAS INFO
image_rotation=parsed_yaml_file.get("camera_config").get("orient_param") #to know if the images have to be rotated
#################################################################################################################################
#IMPORTING LABELS NAMES
posture_labels=parsed_yaml_file.get("action_recog_config").get("posture_labels") # labels of the gestures used to train the gesture recognition model
motion_labels=parsed_yaml_file.get("action_recog_config").get("motion_labels") # labels of the possible motion actions
orientation_labels=parsed_yaml_file.get("action_recog_config").get("orientation_labels") # labels of the possible human orientations
hri_status_label=parsed_yaml_file.get("human_safety_config").get("hri_status") # labels of the possible HRI status
audio_message_label=parsed_yaml_file.get("human_safety_config").get("audio_message") # labels of the possible safety audio messages
safety_action_label=parsed_yaml_file.get("human_safety_config").get("safety_action") # labels of the possible safety actions
human_command_label=parsed_yaml_file.get("human_safety_config").get("human_command") # labels of the possible human commands based on gesture recognition
action_label=parsed_yaml_file.get("robot_config").get("action") # labels of the possible robot actions
#########################################################################################################################

topic_list=['/camera1/color/image_raw','/camera2/color/image_raw','/human_info','/human_safety_info','/robot_info','/nav_vel'] #name of topics (old and new) to be extracted from bag files
  
###############################################################################################
# Main Script

counter=0
new_data=[0,0]
for k in range(0,n_labels):
    if mode==1: #For testing
        folder_name=testing_folder
    else: # For training
        folder_name=training_folder+labels[k]+"/"   
    files = os.listdir(folder_name)
    for file in files:
        bag = rosbag.Bag(folder_name+file)
        for topic, msg, t in bag.read_messages(topics=topic_list):          
            #################################################################################################
            ## EXTRACTING MSG FROM BAG FILES
            #################################################################################################
            if topic==topic_list[0] or topic==topic_list[1]: #RGB          
                new_data[0]=1
                msg.__class__ = sensor_msgs.msg._Image.Image # to fix problems with msg classes when using rosbag and ros_numpy
                bgr_image = ros_numpy.numpify(msg) #replacing cv_bridge
                rgb_image = bgr_image[...,[2,1,0]].copy() #from bgr to rgb
                if topic==topic_list[0]: #this image don't need to be rotated
                    image_rotation=0
                else: #topic==topic_list[1] this image needs to be rotated
                    image_rotation=270
                if image_rotation==90:
                    img_rgb_rot=cv2.rotate(rgb_image,cv2.ROTATE_90_CLOCKWISE)
                elif image_rotation==270:
                    img_rgb_rot=cv2.rotate(rgb_image,cv2.ROTATE_90_COUNTERCLOCKWISE)
                else: #0 degrees
                    img_rgb_rot=rgb_image  
                color_image=img_rgb_rot
            if topic==topic_list[2] or topic==topic_list[3]: #Depth
                new_data[1]=1
                msg.__class__ = sensor_msgs.msg._Image.Image # to fix problems with msg classes when using rosbag and ros_numpy
                depth_image = ros_numpy.numpify(msg) #replacing cv_bridge
                depth_array = np.array(depth_image, dtype=np.float32)/1000
                if topic==topic_list[2]: #this image don't need to be rotated
                    image_rotation=0
                else: #topic==topic_list[3] this image needs to be rotated
                    image_rotation=270
                if image_rotation==90:
                    img_d_rot=cv2.rotate(depth_array,cv2.ROTATE_90_CLOCKWISE)
                elif image_rotation==270:
                    img_d_rot=cv2.rotate(depth_array,cv2.ROTATE_90_COUNTERCLOCKWISE)
                else: #0 degrees
                    img_d_rot=depth_array
                depth_array=img_d_rot
            #################################################################################################
            ## OPENPOSE DETECTION
            #################################################################################################
            if new_data==[1,1] and len(depth_array[0])==len(color_image[0]) and len(depth_array[1])==len(color_image[1]):
                new_data=[0,0]  
                mirror_counter=0 
                while mirror_counter<=1:           
                    if mirror_counter==1: #to mirror the image the second time
                        color_image=cv2.flip(color_image, 1)
                        depth_array=cv2.flip(depth_array, 1)
                    ####################################################################################################
                    #Keypoints extraction using OpenPose
                    datum.cvInputData = color_image
                    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                    #Feature extraction
                    [features,n_human,centroid]=feature_extraction_3D(datum.poseKeypoints,depth_array)
                    if n_human==1: #execute only if there is a valid detection                                      
                        if mode==1: #TESTING
                            #HUMAN POSTURE RECOGNITION
                            X=np.array(features).transpose()
                            Y=model_rf.predict([X])
                            # Model Accuracy, how often is the classifier correct?
                            print("RESULT:",counter,labels[int(Y)])
                        else: #TRAINING                  
                            X_new=np.array(features)
                            X=np.concatenate((X, [X_new]), axis=0)
                            if mirror_counter==0:
                                Y_new=k
                            else:
                                if k>=1 and k<=4: #if originally was left label, then renamed then to right
                                    Y_new=k+4
                                elif k>=5 and k<=8: #if originally was right label, then renamed then to left
                                    Y_new=k-4
                                else:
                                    Y_new=k #label is not changing after mirror
                            Y=np.concatenate((Y, [Y_new]), axis=0)
                            print("RESULT:",counter,labels[int(Y_new)])
                        counter=counter+1
                    else:
                        print("Not valid OpenPose detection")
                        ######################################################################
                        ######################################################################        
                    #Visualization
                    image=datum.cvOutputData
                    if centroid!=[]:
                        center_coordinates = (int(centroid[0]), int(centroid[1])) 
                        image = cv2.circle(image, center_coordinates, 5, (255, 0, 0), 20) #BLUE   
                    cv2.imshow("OpenPose Python API",image)
                    cv2.waitKey(10)
                    if mode==1:    
                        mirror_counter=mirror_counter+2
                    else:
                        mirror_counter=mirror_counter+1
            #if i==5 or i==10 or i==15 or i==20:
            #    break
        bag.close()


if __name__ == '__main__':
    # Setup and call subscription
    #Rate setup
    rate = rospy.Rate(1/pub_hz)  # main loop frecuency in Hz
    while not rospy.is_shutdown():	
        if visual_mode==2:  
            color_image = np.zeros((black_image_size[0],black_image_size[1],3), np.uint8) 
        else: #visual_mode=1 or 3
            color_image=human.image
        visual_outputs(color_image)
        print("MODE",visual_mode)
        rate.sleep() #to keep fixed the publishing loop rate
        
        
