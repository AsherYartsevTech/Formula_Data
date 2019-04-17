import setup_path 
import airsim
import numpy as np
import cv2
import time
from simulationClass import Simulation

######## FUNCTIONS ########
# commit check

def opencv_image(airsim_images):
    """ convert airsim image to opencv from standard image processing
    :param airsim_image uncompressed 8-bit color airsim image
    :return numpy image (ndarray)
    """
    processedImages=[]
    # get numpy array
    for airsim_image in airsim_images:
        
        img1d = np.fromstring(airsim_image.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        I = img1d.reshape(airsim_image.height, airsim_image.width, 4)

        # opencv bgr abomination
        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    
        processedImages.append(I)

    return processedImages


def grab(client):
    
    # get depth image
    # D = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)])[0]))
    
    # get infrared image
    I = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                                            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])))
    
    return I
    
    # get infrared image
    #S = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]))
    
    # stick them together
    #R = np.concatenate((D, I, S), axis=1)
    
    # return R


def saveSnapshotsInPath(path):
    pass






# connect to the AirSim simulator 
#client = airsim.CarClient()
#client.confirmConnection()
#car_controls = airsim.CarControls()

start = time.time()

## asher todo: extract to functionality
#success = client.simSetSegmentationObjectID('spline_cones_best_200', 20)
#conesId = client.simGetSegmentationObjectID('spline_cones_best')


print("Time,Speed,Gear,PX,PY,PZ,OW,OX,OY,OZ")

# monitor car state while you drive it manually.
while (cv2.waitKey(1) & 0xFF) == 0xFF:
    # get state of the car
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    orientation = car_state.kinematics_estimated.orientation
    milliseconds = (time.time() - start) * 1000
    print("%s,%d,%d,%f,%f,%f,%f,%f,%f,%f" % \
       (milliseconds, car_state.speed, car_state.gear, pos.x_val, pos.y_val, pos.z_val, 
        orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val))
    
    snapshots = grab(client)

    saveSnapshotsInPath('/path/to/folder')
    
    for idx,I in enumerate(snapshots):
     # show
        cv2.imshow('image_{}_{}'.format(idx,conesId),I)
    time.sleep(0.1)
