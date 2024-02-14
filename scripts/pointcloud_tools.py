import json
import logging
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


class PointCloudCreator:
    """object to create a point cloud from a json file"""

    def __init__(
        self,
        conf_file: Path,
        logger_level: int = logging.DEBUG,
        logger_name: str = "point_cloud_creator",
    ) -> None:
        """Initialize the PointCloudCreator for a single camera, with constant configuration,
         to batch process depth images.

        Parameters
        ----------
        conf_file: Path
            location to the conf json with the camera intrinsics.
        logger_level: int, optional
            set log_level to 100 to supress all logging.
        logger_name: str, optional
            create a logger for writing outputs and debugging
        """
        # create logger
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(name=logger_name)
        self.logger.setLevel(logger_level)

        self.logger.info(f"Initialing point cloud creator from config file {conf_file}")

        # get camera settings from conf file
        self.settings = {}
        self.load_settings(conf_file=conf_file)

        # init a dummy placeholder to speed up the generation of images
        self.dummy_x = self.settings["x"]
        self.dummy_y = self.settings["y"]
        self.dummy_img = np.array(list(product(np.arange(0, self.dummy_y), np.arange(0, self.dummy_x))))
        self.dummy_index = np.zeros(self.dummy_x * self.dummy_y, np.float64)

    def load_settings(self, conf_file: Path) -> None:
        """Read and interpret the json config file of the camera.

        Parameters
        ----------
        conf_file: Path
            Json file with the camera intrinsics
        """
        with open(conf_file) as f:
            settings = json.loads(f.read())

        intrinsics, depth_scale, depth_trunc = self.set_intrinsics(settings)
        self.settings = {
            "intrinsics": intrinsics,
            "depth_scale": depth_scale,
            "depth_trunc": depth_trunc,
            "x": settings["color_int"]["width"],
            "y": settings["color_int"]["height"],
        }

    def set_intrinsics(self, settings: dict) -> (o3d.camera.PinholeCameraIntrinsic, float, float):
        """Set camera intrinsics from a dictionary

        Parameters
        ----------
        settings: dict
            dictionary of settings

        Returns
        -------
        o3d.camera.PinholeCameraIntrinsic:
            Intrinsics
        depth_scale: float
            See description in input args for o3d.geometry.RGBDImage().create_from_color_and_depth
        depth_trunc: float
            See description in input args for o3d.geometry.RGBDImage().create_from_color_and_depth
        """
        """converts json object to o3d pinhone camera model"""
        if "DepthScale" not in settings.keys():
            depth_scale = 1 / 1000
            self.logger.debug(f"DepthScale not found in conf, using default value ({depth_scale=}).")
        else:
            depth_scale = settings["DepthScale"]
        if "DepthTrunc" not in settings.keys():
            depth_trunc = 20000
            self.logger.debug(f"DepthTrunc not found in conf, using default value ({depth_trunc=}).")
        else:
            depth_trunc = settings["DepthTrunc"]

        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            settings["color_int"]["width"],
            settings["color_int"]["height"],
            settings["color_int"]["fx"],
            settings["color_int"]["fy"],
            settings["color_int"]["ppx"],
            settings["color_int"]["ppy"],
        )

        return intrinsics, depth_scale, depth_trunc

    def convert_depth_to_pcd(
        self,
        rgb_file: Path,
        depth_file: Path,
    ) -> o3d.geometry.PointCloud:
        """Directly convert depth image to point cloud (pcd)
        This is faster than the custom approach via a points array, but open3d removes some points.

        Parameters
        ----------
        rgb_file: Path
            file to the color image
        depth_file: Path
            file to the depth image

        Returns
        -------
        point_cloud_object: o3d.geometry.PointCloud
            The point cloud as open3d object
        """
        t0 = time.time()
        self.logger.info(f"Converting depth image file {depth_file} to pcd.")

        # read images
        rgb_img = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)
        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)

        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_img).astype(np.uint8))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_img).astype(np.uint16))

        rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
            color=rgb_o3d,
            depth=depth_o3d,
            convert_rgb_to_intensity=(len(rgb_img.shape) != 3),
            depth_trunc=self.settings["depth_trunc"],
            depth_scale=1 / self.settings["depth_scale"],
        )

        # extrinsic is the default argument. Providing it explicitly suppresses a warning.
        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=self.settings["intrinsics"],
            extrinsic=np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            ),
        )

        self.logger.info(f"Generated pcd for {depth_file} in {time.time() - t0:.2f} seconds.")
        return pcd

    def convert_depth_to_point_array(self, depth_file: Path) -> np.array:
        """This is a custom version to reconstruct the point cloud. It is a bit slower but could useful
        if you want the point cloud to be exactly the same size as the image

        Parameters
        ----------
        depth_file: Path
            file to the depth image

        Returns
        -------
        points_array: np.array
            An array with x,y,z coordinates of each pixel.

        """
        t0 = time.time()
        self.logger.info(f"Converting depth image file {depth_file} to points array.")

        intrinsics = self.settings["intrinsics"]
        depth_scale = self.settings["depth_scale"]

        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)

        # copy indexes
        y_index = self.dummy_img[:, 0].copy()
        x_index = self.dummy_img[:, 1].copy()

        # convert complete array no forloop needed. X en Y are vectors!
        Z = depth_scale * depth_img[y_index, x_index]
        X = self.dummy_index.copy()
        Y = self.dummy_index.copy()

        z_bool = Z != 0

        X[z_bool] = (
            Z[z_bool] * (x_index[z_bool] - intrinsics.get_principal_point()[0]) / intrinsics.get_focal_length()[0]
        )
        Y[z_bool] = (
            Z[z_bool] * (y_index[z_bool] - intrinsics.get_principal_point()[1]) / intrinsics.get_focal_length()[1]
        )

        points_array = np.array([X, Y, Z], dtype=np.float64).transpose()
        self.logger.info(
            f"Generated points array of {len(points_array)=} for {depth_file} in {time.time() - t0:.2f} seconds."
        )

        return points_array


def write_pcd(
    pcd: o3d.geometry.PointCloud,
    pcd_file: Path or str,
    down_sample: bool = False,
    down_factor: int = 9,
) -> None:
    """Writes the open3d point cloud to a file

    Parameters
    ----------
    pcd: o3d.geometry.PointCloud
        The point cloud as open3d object
    pcd_file: Path
    down_sample: bool
        Naively Discard a fraction of the points to reduce the point cloud size.
    down_factor: int
        number of points to keep. e.g., it keeps 1 out of down_factor points.
    """
    if down_sample:
        pcd = pcd.select_by_index(range(0, len(pcd.points), down_factor))
    o3d.io.write_point_cloud(str(pcd_file), pcd, write_ascii=False, compressed=True)


def create_pcd_from_array(
    rgb_file: Path,
    points_array: np.array,
) -> o3d.geometry.PointCloud:
    """Convert point array (output of convert_depth_to_point_array) to an open3d pcd object.
    Note: this step is relatively slow.

    Parameters
    ----------
    rgb_file: Path
        file to the color image
    points_array: np.array
        the point array (output of convert_depth_to_point_array)

    Returns
    -------
    point_cloud_object: o3d.geometry.PointCloud
        The point cloud as open3d object
    """
    rgb_img = cv2.cvtColor(cv2.imread(str(rgb_file)), cv2.COLOR_BGR2RGB)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(np.reshape(rgb_img, (rgb_img.shape[0] * rgb_img.shape[1], 3)) / 255.0)
    return pcd


def read_pcd(file_name: Path) -> o3d.geometry.PointCloud:
    """Reads a point cloud file (.pcd)

    Parameters
    ----------
    file_name: Path
        The path of the pcd file

    Returns
    -------
    pcd: o3d.geometry.PointCloud
        The point cloud as open3d object
    """
    if not file_name.is_file():
        raise Exception(f"{file_name} is not a file")
    return o3d.io.read_point_cloud(str(file_name))
