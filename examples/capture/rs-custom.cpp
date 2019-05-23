#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp"          // Include short list of convenience functions for rendering
#include <opencv2/opencv.hpp> 
#include <librealsense2/rs_advanced_mode.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
//#include <opencv2/imgcodecs/imgcodecs.hpp>
#include<iostream>
#include "stb_image_write.h"
#include<pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>

using namespace std;
using namespace cv;
using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
using rs2_ptr = rs2::points;
static rs2::device get_a_realsense_device();

//--------------------------------------------------------------------------------------------------------------------------------------------------
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointXYZ PointXYZ;

//--------------------------------------------------------------------------------------------------------------------------------------------------
std::tuple<rs2_ptr, rs2_ptr>   apply_filters(rs2::frame& depth, rs2::decimation_filter& dec, rs2::pointcloud& pc)
{
        rs2::disparity_transform depth2disparity;
        rs2::disparity_transform disparity2depth(false);
        
        // Define spatial filter (edge-preserving)
        rs2::spatial_filter spat;

        spat.set_option(RS2_OPTION_HOLES_FILL, 5);
        rs2::temporal_filter temp;
        rs2::align align_to(RS2_STREAM_DEPTH);
        
        auto depth2 = dec.process(depth);
        depth2 = depth2disparity.process(depth);
        // Apply spatial filtering


        depth2 = spat.process(depth);

        // Apply temporal filtering
        auto depth_both_f = temp.process(depth2);
        // If we are in disparity domain, switch back to depth
        //depth = disparity2depth.process(depth);
        //end of applying filters
        auto points_spat = pc.calculate(depth2);
        // Generate the pointcloud and texture mappings
        auto points_both = pc.calculate(depth_both_f);

        return std::make_tuple(points_spat,points_both);
}
//-----------------------------------------------------------------------------------------------------------------------------------
//function for depth clamp controls
static void set_depth_clamp(rs2::device& dev)
{
    //advanced mode API
    rs400::advanced_mode advanced_device(dev);

    auto depth_table = advanced_device.get_depth_table();   //initial value : 65536
    std::cout << "  Max Value     : " << depth_table.depthClampMax << std::endl;
    //set new depth clamp max
    depth_table.depthClampMax = 4000; 
    advanced_device.set_depth_table(depth_table);
    //print changes
    auto depth_table2 = advanced_device.get_depth_table();
    std::cout << "  Max Value     : " << depth_table2.depthClampMax << std::endl;

}
//-----------------------------------------------------------------------------------------------------------------------------------
//Convert point cloud rs2 -> PCL 
pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}
//-----------------------------------------------------------------------------------------------------------------------------------
//Class with method for voxel grid filter implementation
class FilterAndPublish
{
    public:
        FilterAndPublish()
        {
            printf("Made object for voxel grid filter\n");
            this->thresh = 15; // This is the minimum number of points that have to occupy a voxel in order for it to survive the downsample.
        }

        pcl_ptr callback(const PointCloud::ConstPtr& msg)
        {
            PointCloud::Ptr cloud (new PointCloud);
            PointCloud::Ptr cloud_filtered (new PointCloud);
            *cloud = *msg;
            // 1. Take cloud and put it in a voxel grid while restricting the bounding box
            // 2. Go through the voxels and remove all points in a voxel that has less than this.thresh points
            // 3. Publish resulting cloud
            pcl::VoxelGrid<PointXYZ> vox;
            vox.setInputCloud(cloud);
            // The leaf size is the size of voxels pretty much. Note that this value affects what a good threshold value would be.
            vox.setLeafSize(0.05f, 0.05f, 0.05f);
            // I limit the overall volume being considered so lots of "far away" data that is just terrible doesn't even have to be considered.
            vox.setFilterLimits(-1.0, 1.0);
            // The line below is perhaps the most important as it reduces ghost points.
            vox.setMinimumPointsNumberPerVoxel(this->thresh);
            vox.filter(*cloud_filtered);
            
            return cloud_filtered;
            //pub.publish (cloud_filtered);
        }

    private:
        int thresh;

};
//-----------------------------------------------------------------------------------------------------------------------------------
//Main program
// Get depth and color video streams & export to ply and png formats respectively
int main(int argc, char * argv[]) try
{

    rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
    // Create a simple OpenGL window for rendering:
    window app(1280, 720, "RealSense Capture Example");
    // Construct an object to manage view state
    // Construct an object to manage view state


    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;
    //Filters
    // Decimation filter reduces the amount of data (while preserving best samples)
    rs2::decimation_filter dec;
    dec.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2);

    // Declare rates printer for showing streaming rates of the enabled streams.
    rs2::rates_printer printer;
    
    rs2::config cfg;

    // RGB-Depth channel format
    cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Start streaming with default recommended configuration
    // The default video configuration contains Depth and Color streams
    // If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
    //pipe.start();
    rs2::pipeline_profile selection = pipe.start();

    //Get first RealSense device(line 65  set_option(MAX_DIST) not working) + advance mode for changing depth_max value 
    rs2::context ctx;
    auto list = ctx.query_devices();
    rs2::device dev = list.front();
    //calling advance mode
    set_depth_clamp(dev);
    //-----------------------------------------------------------------------------------------------------------------------------------

    //variable to iterate through rgb_frame and depth_frame sequences
    int i=0;
    const string s = "/home/user/Documents/frame_";
    const string s1 = ".png";

    const string cl =  "/home/user/Documents/p_cloud__";
    const string clf =  "/home/user/Documents/p_cloud_filtered_";
    const string clf2 =  "/home/user/Documents/p_cloud_filtered_both_";
    const string clf_no_f = "/home/user/Documents/p_cloud_no_filter_";
    string num;
    const string pcd =  "/home/user/Documents/cloud_";
    const string cl1 = ".pcd";
    const string cl2 = ".ply";
    //-----------------------------------------------------------------------------------------------------------------------------------
    //Get intrinsics
    //auto profile = pipe.start(cfg);
    //auto stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    //auto intrinsics = stream.get_intrinsics(); // Calibration data
    
    while (app) // Application still alive?
    {
        // Capture 20 frames to give autoexposure, etc. a chance to settle
        for (auto i = 0; i < 20; ++i) pipe.wait_for_frames();

        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        auto rgb_frame = frames.get_color_frame();
        //depth is type: rs2::frame
        auto depth = frames.get_depth_frame();

        rs2::points points_spat, points_both;
        //points_spat - filtered with spatial filter, points_both - filtered with spatial&temporal filters
        
        //apply filters to depth
        // Decimation will reduce the resultion of the depth image,
        // closing small holes and speeding-up the algorithm
        tie(points_spat, points_both) = apply_filters(depth, dec, pc);

        points = pc.calculate(depth);

        auto color = frames.get_color_frame();
        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frames.get_infrared_frame();
                // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        // Draw the pointcloud
        //draw_pointcloud(app.width(), app.height(), app_state, points);

        i+=1;
        if (i<10){
            num="000";
        }
        else{
            num="00";
        }
        std::string i_s = to_string(i);
        const string s2 = s+num+i_s+s1;
        const string cl2_ply = pcd+num+i_s+cl1;
        const string cl_ply = cl+i_s+cl2;
        const string clf_ply = clf+i_s+cl2;
        const string clf2_ply = clf2+i_s+cl2;
        const string clf_no_f_ply = clf_no_f+i_s+cl2;

         //export RGB frame
         stbi_write_png(s2.c_str(), rgb_frame.get_width(), rgb_frame.get_height(),
               rgb_frame.get_bytes_per_pixel(), rgb_frame.get_data(), rgb_frame.get_stride_in_bytes());
         

         //Convert points to pcl
         auto pcl_points = points_to_pcl(points_spat);
         auto raw_points = points_to_pcl(points);
         //Voxel filter
         //FilterAndPublish obj = FilterAndPublish();
         //auto filtered_cloud = obj.callback(pcl_points);
         //pcl::io::savePLYFile (filename, *filtered_cloud);
          
         //Once in pcl -> export to organized PC (pcd)
         //pcl::io::savePCDFileASCII(cl2_ply.c_str(), * pcl_points); 
         pcl::io::savePCDFileASCII(cl2_ply.c_str(), * raw_points); 
         //stbi_write_png("/home/aldoum/Documents/a.png", rgb_frame.get_width(), rgb_frame.get_height(),
           //    rgb_frame.get_bytes_per_pixel(), rgb_frame.get_data(), rgb_frame.get_stride_in_bytes());
         //Processing via OpenCV
         //cv::Mat dMat_colored = cv::Mat(cv::Size(1920, 1080), CV_8UC3, (void*)rgb_frame.get_data());
         //cv::imwrite( "/home/aldoum/Documents/coloredC3.png", dMat_colored);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
