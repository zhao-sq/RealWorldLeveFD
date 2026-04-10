import pyrealsense2 as rs
import numpy as np
import os
# import open3d as o3d
# import open3d.pipelines.registration as registration
import time
import copy

# from robot_controller import robot_controller
# from scipy.spatial.transform import Rotation as R

def findDevices():
    ctx = rs.context() # Create librealsense context for managing devices
    serials = []
    if (len(ctx.devices) > 0):
        for dev in ctx.devices:
            print ('Found device: ', \
                    dev.get_info(rs.camera_info.name), ' ', \
                    dev.get_info(rs.camera_info.serial_number))
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
        
    return serials, ctx

def initialize_camera(serial, ctx):
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)
    # assuming we are using 435i, no need to align the depth and color
    # Create a pipeline
    # check whether we are using 515
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
    else:
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 30)

    # config.enable_stream(rs.stream.depth, 640, 480)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    print("cfg", config)
    profile = pipeline.start(config)
    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 500.000)
    sensor.set_option(rs.option.brightness,1)

    if device_product_line == 'L500':
        align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 5)
    else:
        align = None
    return pipeline, align

def get_rgbd_image(pipeline, align):
    frames = pipeline.wait_for_frames()
    if align is None:
        aligned_frames = frames
    else:
        aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    return color_image, depth_image

def get_rgbd_image_res(pipeline, align, results):
    frames = pipeline.wait_for_frames()
    if align is None:
        aligned_frames = frames
    else:
        aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    results.append(color_image)
    results.append(depth_image)

    return None

if __name__ == "__main__":
    # Find devices
    serials, ctx = findDevices()
    # Initialize camera
    # pipeline, align = initialize_camera("246322301968", ctx)
    pipeline, align = initialize_camera("241122306995", ctx)
    # Get RGBD image
    color_image, depth_image = get_rgbd_image(pipeline, align)
    # Show the image
    import cv2
    cv2.imshow("color", color_image)
    cv2.imshow("depth", depth_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Stop the pipeline
    pipeline.stop()

