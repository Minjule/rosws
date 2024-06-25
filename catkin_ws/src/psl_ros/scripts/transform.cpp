#include <ros/ros.h>

// Include pcl
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <thread>
#include <chrono>

#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/common/distances.h> 

// Topics
static const std::string IMAGE_TOPIC = "/camera/depth/points";
static const std::string PUBLISH_TOPIC = "/pcl/transformed/points";

ros::Publisher pub;

typedef pcl::PointXYZ PointType;
//pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
pcl::visualization::PCLVisualizer color_viewer("Color Viewer");
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr voxeled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
pcl::PassThrough<pcl::PointXYZ> filter;
pcl::VoxelGrid<pcl::PointXYZ> sor;

pcl::SACSegmentation<pcl::PointXYZ> seg;
pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
pcl::ExtractIndices<pcl::PointXYZ> extract;
pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_inliers(new pcl::PointCloud<pcl::PointXYZ>);

pcl::SACSegmentation<pcl::PointXYZ> seg2;
pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);
pcl::ExtractIndices<pcl::PointXYZ> extract2;
pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_inliers2(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_extracted_inliers(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);

pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> clustering;
std::vector<pcl::PointIndices> euclidean_cluster_indices;
pcl::EuclideanClusterExtraction<PointType> ec;

std::vector<pcl::PointIndices> clusters;

pcl::PointXYZ target_point;

sensor_msgs::PointCloud2 cloud_msg_transformed;

using namespace std::chrono_literals;
std::vector<int> mapping;

bool cloud_added = false;
void func(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
	auto start = std::chrono::high_resolution_clock::now();


    /*pcl::PCLPointCloud2 pcl_pc2;
    
    pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);*/
    
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    
    float theta = M_PI; // The angle of rotation in radians
    transform_1(0, 0) = std::cos(theta);
    transform_1(0, 1) = -sin(theta);
    transform_1(1, 0) = sin(theta);
    transform_1(1, 1) = std::cos(theta);
    
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform_1);
    
    filter.setInputCloud(transformed_cloud);
    filter.setFilterFieldName("z");
    filter.setFilterLimits(0.0, 4.0);
    filter.filter(*filtered_cloud);
    
    sor.setInputCloud(filtered_cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*voxeled_cloud);
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(40);
    seg.setDistanceThreshold(0.02);
    seg.setEpsAngle(90.0f * (M_PI/180.0f));

    seg.setInputCloud(voxeled_cloud);
    seg.segment(*inliers, *coefficients);
    
    int i = 0, nr_points = (int)voxeled_cloud->points.size();
    
    
    while (voxeled_cloud->points.size() > 0.2 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud(voxeled_cloud);
        seg.segment(*inliers, *coefficients);

        // Extract the inliers
        extract.setInputCloud(voxeled_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*extracted_inliers);
        voxeled_cloud.swap(extracted_inliers);
        i++;
    }
    
    kdtree->setInputCloud(extracted_inliers);

        // Run further processing on the cluster_cloud
        // For example, you can apply RANSAC for sphere detection here
    
    normalEstimation.setInputCloud(extracted_inliers);
    normalEstimation.setRadiusSearch(0.03);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);
	
    clustering.setMinClusterSize(300);
    clustering.setMaxClusterSize(3000);
    clustering.setSearchMethod(kdtree);
    clustering.setNumberOfNeighbours(30);
    clustering.setInputCloud(extracted_inliers);
    clustering.setInputNormals(normals);
    clustering.setSmoothnessThreshold(25.0 / 180.0 * M_PI); // 7 degrees.
    clustering.setCurvatureThreshold(30.0);
    
    clustering.extract(clusters);
	
    seg2.setOptimizeCoefficients(true);
    seg2.setModelType(pcl::SACMODEL_SPHERE);
    seg2.setMethodType(pcl::SAC_RANSAC);
    seg2.setDistanceThreshold(0.01); // Set your desired distance threshold here
    seg2.setMaxIterations(40);
    seg2.setRadiusLimits(0.05, 0.1);

    seg2.setInputCloud(extracted_inliers);
    seg2.segment(*inliers2, *coefficients2);
    
    //color_viewer.removeAllPointClouds();
    int a=1;
    for (const auto &cluster : clusters) {
        // Extract cluster point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &index : cluster.indices) {
            cluster_cloud->points.push_back(extracted_inliers->points[index]);
        }
        cluster_cloud->width = cluster_cloud->points.size();
        cluster_cloud->height = 1;
        cluster_cloud->is_dense = true;
	    
	seg2.setInputCloud(cluster_cloud);
        seg2.segment(*inliers2, *coefficients2);
        
        extract2.setInputCloud(cluster_cloud);
        extract2.setIndices(inliers2);
        extract2.setNegative(true);
        extract2.filter(*extracted_inliers2);
        
        for (const auto& point : cluster_cloud->points) {
		pcl::PointXYZRGB color_point;
		color_point.x = point.x;
		color_point.y = point.y;
		color_point.z = point.z;
		color_point.r = 255;
		color_point.g = 255;
		color_point.b = 255;
		color_extracted_inliers->points.push_back(color_point);
    	}

    	for (size_t i = 0; i < inliers2->indices.size(); ++i) {
		pcl::PointXYZRGB color_point;
		color_point.x = cluster_cloud->points[inliers2->indices[i]].x;
		color_point.y = cluster_cloud->points[inliers2->indices[i]].y;
		color_point.z = cluster_cloud->points[inliers2->indices[i]].z;
		color_point.r = 255;
		color_point.g = 0;
		color_point.b = 0;
		color_extracted_inliers->points.push_back(color_point);
	    }

        // Check if a sphere model was found
        if (inliers2->indices.size() == 0) {
            std::cerr << "No sphere model found for cluster." << std::endl;
            continue;
        }
        double min_distance = std::numeric_limits<double>::max();
        for (const auto& point : extracted_inliers2->points) {
            double distance = pcl::euclideanDistance(target_point, point);
            if (distance < min_distance) {
                min_distance = distance;
            }
        }

    // Output the minimum distance to the detected object
    std::cout << "Minimum distance to detected object: " << min_distance << " meters" << std::endl;

    std::cout << "Sphere model coefficients: " << *coefficients2 << std::endl;
        
        //color_viewer.updatePointCloud(color_extracted_inliers, "Color Viewer");
        //viewer.addPointCloud<pcl::PointXYZ>(extracted_inliers2, "largest cluster");
    }
    
    pcl::toROSMsg(*color_extracted_inliers, cloud_msg_transformed);
    
    cloud_msg_transformed.header.frame_id = "map"; // Set your frame ID here
    cloud_msg_transformed.header.stamp = ros::Time::now();
    
    color_viewer.removeAllPointClouds();
    color_viewer.addPointCloud<pcl::PointXYZRGB>(color_extracted_inliers, "Color Viewer");
   //color_viewer.updatePointCloud(color_extracted_inliers, "Color Viewer");
    
    //viewer.removeAllPointClouds();
    //viewer.addPointCloud<pcl::PointXYZ>(extracted_inliers, "largest cluster");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], "largest cluster");
    auto end = std::chrono::high_resolution_clock::now();
    pub.publish(cloud_msg_transformed);
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << " seconds" << std::endl;
}
int main (int argc, char** argv){
      ros::init (argc, argv, "pcl_ros");
      ros::NodeHandle nh;

      target_point.x = 0.0; // Replace with the x-coordinate of the detected object
      target_point.y = 0.0; // Replace with the y-coordinate of the detected object
      target_point.z = 0.0;
    
      ROS_INFO_STREAM("Hello from ROS Node: " << ros::this_node::getName());
      ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 1, func);
      
      //viewer.setBackgroundColor(0.0, 0.0, 0.0);
      //viewer.addCoordinateSystem(1.0);
      //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");
      
      color_viewer.setBackgroundColor(0.0, 0.0, 0.0);
      color_viewer.addCoordinateSystem(1.0);
      color_viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
    
      pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);
      
      while (ros::ok() && !color_viewer.wasStopped())
    {
        color_viewer.spinOnce(100);
        ros::spinOnce(); // Handle ROS callbacks
    }
      return 0;
}
