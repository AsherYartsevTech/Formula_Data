import logging
import airsim
import json
import random
import numpy as np
import cv2

# maximal ID from simulation
MAX_OBJECT_ID = 255


class Simulation(object):
    def __init__(self, ip='127.0.0.1', sim_config_filename=None, reset_simulation=True):
        """ initialize simulation
        :param config configuration file
        """

        logging.info('connecting to simulator at ip {}'.format(ip))
        self.client = airsim.CarClient(ip=ip)

        # confirm connection succeeded
        logging.info("confirming connection")
        self.client.confirmConnection()

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
