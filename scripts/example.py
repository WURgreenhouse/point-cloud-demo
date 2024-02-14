from pathlib import Path

from pointcloud_tools import PointCloudCreator, create_pcd_from_array, read_pcd, write_pcd


def main():
    """Demo script that read the image in data/depth + data/rgb and outputs a pcd file in data/pointclouds
    with the same name, in two different ways."""

    # point to the data dir, relative from this script folder
    current_folder = Path(__file__).resolve().parent
    base_dir = current_folder.parent / "data"

    # init a point cloud creator with the camera configuration
    point_cloud_creator = PointCloudCreator(conf_file=base_dir / "oak-d-s2-poe_conf.json")

    # point to the image files
    image = "A_1a111b40"
    rgb_file = base_dir / "rgb" / f"{image}.png"
    depth_file = base_dir / "depth" / f"{image}_depth.png"

    #
    # First approach, Using open3d fully. Is faster but will remove some points
    #
    out_file = base_dir / "pointclouds" / f"{image}_open3d.pcd"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.unlink(missing_ok=True)

    # create pcd object
    pcd_object2 = point_cloud_creator.convert_depth_to_pcd(rgb_file=rgb_file, depth_file=depth_file)

    # write the (down-sampled) pcd object to file
    write_pcd(pcd=pcd_object2, pcd_file=out_file, down_sample=False, down_factor=9)

    # now check that we can read it again
    pcd_object = read_pcd(out_file)
    assert pcd_object is not None

    #
    # Second approach, a bit faster and more configurable.
    #
    out_file2 = base_dir / "pointclouds" / f"{image}.pcd"
    out_file2.unlink(missing_ok=True)
    out_file2.parent.mkdir(parents=True, exist_ok=True)

    # Start by converting depth image to point array
    points_array = point_cloud_creator.convert_depth_to_point_array(depth_file=depth_file)

    # Now create open3d point cloud file (.pcd). This is quite slow.
    pcd_object2 = create_pcd_from_array(rgb_file=rgb_file, points_array=points_array)

    # write the (down-sampled) pcd object to file
    write_pcd(pcd=pcd_object2, pcd_file=out_file2, down_sample=False, down_factor=9)

    # now check that we can read it again
    pcd_object2 = read_pcd(out_file2)
    assert pcd_object2 is not None


if __name__ == "__main__":
    main()
