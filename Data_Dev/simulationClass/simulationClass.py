import logging
import airsim
import json
import random
import numpy as np
import cv2
import time
import setup_path

# maximal ID from simulation
MAX_OBJECT_ID = 255


class Simulation(object):
    def __init__(self, ip='127.0.0.1', sim_config_filename=None, reset_simulation=True):
        """ initialize simulation
        :param config configuration file
        """

        logging.info('connecting to simulator at ip {}'.format(ip))
        self.client = airsim.CarClient()

        # confirm connection succeeded
        logging.info("confirming connection")
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        # reset simulation at beginning if so requested
        if reset_simulation:
            self.client.reset()

        if sim_config_filename:
            with open(sim_config_filename) as f:
                config = json.load(f)

            # randomly select ids for meshes
            random_ids = random.sample(range(MAX_OBJECT_ID), len(config['mesh_names'].values()))

            # mesh:mesh_id dictionary
            self.ids = {mesh: mesh_id for (mesh, mesh_id) in zip(config['mesh_names'].values(), random_ids)}

            # set ID for each requested object in segmentation
            for mesh, mesh_id in self.ids.items():
                logging.info("setting mesh {} object id to {}".format(mesh, mesh_id))
                success = self.client.simSetSegmentationObjectID("{}[\w]*".format(mesh), mesh_id, True)
                if not success:
                    error_message = 'simulation: error assigning object id for object {}'.format(mesh)
                    logging.warning(error_message)

            # read segmentation color LUT
            self.colors = self._read_segmentation_colors(config['segmentation_LUT_filename'])

    @staticmethod
    def _read_segmentation_colors(filename):
        # open files
        with open(filename, 'r') as f:
            # read lines
            lines = list(f)

        # parse text to dictionary of lists. inefficient but who cares
        colors = {
            int(line.split()[0]): (int(line.split()[1][1:-1]), int(line.split()[2][:-1]),
                                   int(line.split()[3][:-1])) for line in lines}

        return colors

    @staticmethod
    def _opencv_image(airsim_image):
        """ convert airsim image to opencv from standard image processing
        :param airsim_image uncompressed 8-bit color airsim image
        :return numpy image (ndarray)
        """
        # get numpy array
        img1d = np.frombuffer(airsim_image.image_data_uint8, dtype=np.uint8)

        # reshape array to 4 channel image array H X W X 4
        image = img1d.reshape(airsim_image.height, airsim_image.width, 4)

        # get rid of alpha
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image

    def grab(self, camera_names, image_types):
        """ convert airsim image to opencv from standard image processing
        :param camera_names list of camera names to grab from
        :param image_types list of image types to grab (e.g. airsim.ImageType.Scene)
        :return list of (image, segmenation_image) tuples per camera
        """

        # get camera and segmentation images for each of the camera names given
        images = self.client.simGetImages([
            airsim.ImageRequest(c, t, False, False) for c in camera_names for t in image_types])

        # convert to numpy and rearrange as dictionary
        return {(c, t): self._opencv_image(images[ind]) for ind, (c, t) in
                enumerate([(c, t) for c in camera_names for t in image_types])}

    def segment_objects(self, segmentation_image, object_name):
        """ get mask of objects
        :param segmentation_image to look for objects in
        :param object_name to look for
        :return mask containing object pixels
        """

        # get object mask from simulator
        mask = np.logical_and(np.logical_and(segmentation_image[:, :, 0] == self.colors[self.ids[object_name]][0],
                                             segmentation_image[:, :, 1] == self.colors[self.ids[object_name]][1]),
                              segmentation_image[:, :, 2] == self.colors[self.ids[object_name]][2])

        # opencv whines about object type without this
        mask = 255 * np.uint8(mask)

        return mask

    def set_pose(self, x, y, z, roll, pitch, yaw):
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z),
                                                  airsim.to_quaternion(roll, pitch, yaw)), True)
    def get_pose(self):
        return self.client.simGetVehiclePose(self.client)

    def getCarState(self):
        controls = self.client.getCarState()
        return controls

    def setCarControls(self,controls):
        self.client.setCarControls(self.client,controls)

    def setThrottle(self,throttle):
        controls = airsim.CarControls()
        controls.throttle = 1
        #airsim.CarControls.set_throttle(controls,1,1)
        client.setCarControls(controls)

def opencv_image(airsim_images):
    """ convert airsim image to opencv from standard image prppocessing
    :param airsim_image uncompressed 8-bit color airsim image
    :return numpy image (ndarray)
    """
    processedImages = []
    # get numpy array
    for airsim_image in airsim_images:
        img1d = np.fromstring(airsim_image.image_data_uint8, dtype=np.uint9)
        # reshape array tpipo 4 channel image array H X W X 4
        I = img1d.reshape(airsim_image.height, airsim_image.width, 4)

        # opencv bgr abomination
        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)

        processedImages.append(I)

    return processedImages



def getImages(client):
    # get depth image
    # D = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)])[0]))

    # get depth and sementation image
    I = opencv_image((client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])))

    # stick them together
    # R = np.concatenate((D, I, S), axis=1)

    # return R
    return I


# # connect to the AirSim simulator
# client = airsim.CarClient()
# client.confirmConnection()
# # car_controls = airsim.CarControls()

# start = time.time()

# ## asher todo: extract to functionality
# success = client.simSetSegmentationObjectID('spline_cones_best_200', 20)
# if not success:
#     print('couldnt set color')
# conesId = client.simGetSegmentationObjectID('spline_cones_best')
# # simulation = Simulation(sim_config_filename = "/home/afst/Desktop/formula 4.18/Formula_Data/Data_Dev/simulationConfig.json")
# print("Time,Speed,Gear,PX,PY,PZ,OW,OX,OY,OZ")

# # monitor car state while you drive it manually.
# '''while (cv2.waitKey(1) & 0xFF) == 0xFF:
#     # get state of the car
#     car_state = client.getCarState()
#     pos = car_state.kinematics_estimated.position
#     orientation = car_state.kinematics_estimated.orientation
#     milliseconds = (time.time() - start) * 1000
#     print("%s,%d,%d,%f,%f,%f,%f,%f,%f,%f" % \
#           (milliseconds, car_state.speed, car_state.gear, pos.x_val, pos.y_val, pos.z_val,
#            orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val))

#     snapshots = getImages(client)

#     #saveSnapshotsInPath('/path/to/folder')

#     # snapshots = simulation.grab(["0"],[airsim.ImageType.Segmentation,airsim.ImageType.Scene])
#     for idx, I in enumerate(snapshots):
#         # show
#         cv2.imshow('image_{}'.format(idx), I)
#     time.sleep(0.1)'''
# #sim = Simulation(client)
# #client.enableApiControl(True)
# #sim.setThrottle(1)
# #while (cv2.waitKey(1) & 0xFF) == 0xFF:



#  #   snapshots = getImages(client)
#   #  for idx, I in enumerate(snapshots):
#         # show
#    #     cv2.imshow('image_{}'.format(idx), I)
#    # time.sleep(0.1)






'''
 # get state of the car
        pos = car_state.kinematics_estimated.position
        orientation = car_state.kinematics_estimated.orientation
        #        milliseconds = (time.time() - start) * 1000

        # populate PoseStamped ros message
        simPose = PoseStamped()
        simPose.pose.position.x = pos.x_val
        simPose.pose.position.y = pos.y_val
        simPose.pose.position.z = pos.z_val
        simPose.pose.orientation.w = orientation.w_val
        simPose.pose.orientation.x = orientation.x_val
        simPose.pose.orientation.y = orientation.y_val
        simPose.pose.orientation.z = orientation.z_val
        simPose.header.stamp = rospy.Time.now()
        simPose.header.seq = 1
        simPose.header.frame_id = "simFrame"

        # log PoseStamped message
        #rospy.loginfo(simPose)
        # publish PoseStamped message
        car_state.brake = 0
        client.setCarControls(car_state)
        print(client.simGetVehiclePose('PhysXCar'))
        pub.publish(simPose)
        
        # sleeps until next cycle
'''