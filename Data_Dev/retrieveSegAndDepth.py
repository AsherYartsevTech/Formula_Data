import setup_path 
import airsim
import numpy as np
import cv2
import time
from simulationClass import Simulation
import matplotlib.pyplot as pyplot

######## FUNCTIONS ########
# commit check

def opencv_image(airsim_images):
    """ convert airsim image to opencv from standard image prppocessing
    :param airsim_image uncompressed 8-bit color airsim image
    :return numpy image (ndarray)
    """
    processedImages=[]
    # get numpy array
    for airsim_image in airsim_images:
        
        img1d = np.fromstring(airsim_image.image_data_uint8, dtype=np.uint8)
        # reshape array tpipo 4 channel image array H X W X 4
        I = img1d.reshape(airsim_image.height, airsim_image.width, 4)

        # opencv bgr abomination
        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
    
        processedImages.append(I)

    return processedImages


def grab(client):

    # get depth image
    # D = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)])[0]))

    # get depth and sementation image
    I = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                                            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)])))

    # stick them together
    #R = np.concatenate((D, I, S), axis=1)

    # return R
    return I


def saveSnapshotsInPath(path):

    pass






# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
#car_controls = airsim.CarControls()

start = time.time()

## asher todo: extract to functionality
success = client.simSetSegmentationObjectID('spline_cones_best_200', 20)
if not success:
    print('couldnt set color')
conesId = client.simGetSegmentationObjectID('spline_cones_best')
# simulation = Simulation(sim_config_filename = "/home/afst/Desktop/formula 4.18/Formula_Data/Data_Dev/simulationConfig.json")
print("Time,Speed,Gear,PX,PY,PZ,OW,OX,OY,OZ")

# monitor car state while you drive it manually.
j=0
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
    
    # snapshots = simulation.grab(["0"],[airsim.ImageType.Segmentation,airsim.ImageType.Scene])
    for idx,I in enumerate(snapshots):
      # show
        cv2.imshow('image_{}'.format(idx),I)
        cv2.imwrite('/home/afst/Desktop/recordingForTomYarden18.4.19/image_{}_{}.png'.format(idx,j), I)
        j=j+1
    time.sleep(0.1)
