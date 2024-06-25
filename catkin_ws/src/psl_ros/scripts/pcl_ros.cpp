#include <ros/ros.h>

// Include pcl
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <thread>

#include <pcl/console/parse.h>
#include <pcl/point_cloud.h> // for PointCloud
#include <pcl/common/io.h> // for copyPointCloud
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/impl/angles.hpp>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>

// Include PointCloud2 message
#include <sensor_msgs/PointCloud2.h>

using namespace std::chrono_literals;

// Topics
static const std::string IMAGE_TOPIC = "/camera/depth/points";
static const std::string PUBLISH_TOPIC = "/pcl/points";

ros::Publisher pub;

std::vector<std::vector<float>> colors = {
        {1.0f, 0.0f, 0.0f},  // Red
        {0.0f, 1.0f, 0.0f},  // Green
        {0.0f, 0.0f, 1.0f},  // Blue
        {1.0f, 1.0f, 0.0f},  // Yellow
        {1.0f, 0.0f, 1.0f},  // Magenta
        {0.0f, 1.0f, 1.0f},  // Cyan
        {1.0f, 0.5f, 0.0f},  // Orange
        {0.0f, 0.5f, 1.0f},  // Sky Blue
        {0.5f, 0.0f, 0.5f},  // Purple
        {0.5f, 0.5f, 0.5f}   // Gray
};
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud2(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);
    
pcl::SACSegmentation<pcl::PointXYZ> seg;
pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

pcl::ExtractIndices<pcl::PointXYZ> extract;
pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_inliers(new pcl::PointCloud<pcl::PointXYZ>);
pcl::ExtractIndices<pcl::PointXYZ> extract2;
pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_inliers2(new pcl::PointCloud<pcl::PointXYZ>);
pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cluster Viewer"));

pcl::visualization::PCLVisualizer::Ptr simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud) {
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer4(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer4->setBackgroundColor(0, 0, 0);
    viewer4->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
    viewer4->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    //viewer->addCoordinateSystem (1.0, "global");
    viewer4->initCameraParameters();
    return (viewer4);
}

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // populate our PointCloud with points
    cloud->width = 500;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width * cloud->height);

    std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    pcl::PassThrough<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud);
	// Filter out all points with Z values not in the [0-2] range.
	filter.setFilterFieldName("z");
	filter.setFilterLimits(0.0, 4.0);

	filter.filter(*filtered_cloud);

    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*filtered_cloud);
    filtered_cloud2 = filtered_cloud;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.02);
    seg.setEpsAngle(90.0f * (M_PI/180.0f));

    seg.setInputCloud(filtered_cloud2);
    seg.segment(*inliers, *coefficients);

    /*extract2.setInputCloud(filtered_cloud2);
    extract2.setIndices(inliers2);
    extract2.setNegative(true);
    extract2.filter(*extracted_inliers2);*/

    int i = 0, nr_points = (int)filtered_cloud2->points.size();
    while (filtered_cloud2->points.size() > 0.2 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud(filtered_cloud2);
        seg.segment(*inliers, *coefficients);

        // Extract the inliers
        extract.setInputCloud(filtered_cloud2);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*extracted_inliers);
        filtered_cloud2.swap(extracted_inliers);
        i++;
    }

    pcl::SACSegmentation<pcl::PointXYZ> seg2;
    pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients);
    seg2.setOptimizeCoefficients(true);
    seg2.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
    seg2.setMethodType(pcl::SAC_RANSAC);
    seg2.setMaxIterations(1000);
    seg2.setDistanceThreshold(0.02);
    seg2.setEpsAngle(90.0f * (M_PI/180.0f));

    seg2.setInputCloud(filtered_cloud2);
    seg2.segment(*inliers2, *coefficients2);

    i = 0, nr_points = (int)filtered_cloud2->points.size();
    while (filtered_cloud2->points.size() > 0.2 * nr_points) {
        // Segment the largest planar component from the remaining cloud
        seg2.setInputCloud(filtered_cloud2);
        seg2.segment(*inliers2, *coefficients2);

        // Extract the inliers
        extract2.setInputCloud(filtered_cloud2);
        extract2.setIndices(inliers2);
        extract2.setNegative(true);
        extract2.filter(*extracted_inliers2);
        filtered_cloud2.swap(extracted_inliers2);
        i++;
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	kdtree->setInputCloud(extracted_inliers2);

    // Estimate the normals.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(extracted_inliers2);
	normalEstimation.setRadiusSearch(0.03);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

    // Region growing clustering object.
	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> clustering;
	clustering.setMinClusterSize(100);
	clustering.setMaxClusterSize(6000);
	clustering.setSearchMethod(kdtree);
	clustering.setNumberOfNeighbours(30);
	clustering.setInputCloud(extracted_inliers2);
	clustering.setInputNormals(normals);
    // Set the angle in radians that will be the smoothness threshold
	// (the maximum allowable deviation of the normals).
	clustering.setSmoothnessThreshold(25.0 / 180.0 * M_PI); // 7 degrees.
	// Set the curvature threshold. The disparity between curvatures will be
	// tested after the normal deviation check has passed.
	clustering.setCurvatureThreshold(30.0);

	std::vector<pcl::PointIndices> clusters;
	clustering.extract(clusters);

    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(extracted_inliers2, "cloud");
    //viewer2->addPointCloud<pcl::PointXYZ>(extracted_inliers2, "cloud");
    // Define colors for visualization


    // For every cluster...
	int currentClusterNum = 1;
    int largest_cluster_index = 0;
    size_t largest_cluster_size = 0;
	for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
	{
		// ...add all its points to a new cloud...
		pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
			cluster->points.push_back(extracted_inliers2->points[*point]);
		cluster->width = cluster->points.size();
		cluster->height = 1;
		cluster->is_dense = true;

		// ...and save it to disk.
		if (cluster->points.size() <= 0)
			break;
        if (cluster->points.size() > largest_cluster_size) {
            largest_cluster_size = cluster->points.size();
            largest_cluster_index = currentClusterNum;
        }
		std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;

        std::stringstream ss;
        ss << "cluster_" << currentClusterNum;
        std::string cluster_name = ss.str();

        
        std::vector<float> color = colors[currentClusterNum % colors.size()];
        viewer->addPointCloud<pcl::PointXYZ>(cluster, cluster_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], cluster_name);


		currentClusterNum++;
	}
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);

	for (std::vector<int>::const_iterator point = clusters[largest_cluster_index].indices.begin(); point != clusters[largest_cluster_index].indices.end(); point++)
		cluster->points.push_back(extracted_inliers2->points[*point]);

	cluster->width = cluster->points.size();
	cluster->height = 1;
	cluster->is_dense = true;

    std::vector<float> color = colors[largest_cluster_index % colors.size()];
    
    //viewer2->addPointCloud<pcl::PointXYZ>(cluster, "largest cluster");
    //viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], "largest cluster");
    //viewer2->spin();
    pub.publish(cloud_msg);
    viewer->spin();
}

int main (int argc, char** argv){
      // Initialize the ROS Node "roscpp_pcl_example"
      ros::init (argc, argv, "pcl_ros");
      ros::NodeHandle nh;

      // Print "Hello" message with node name to the terminal and ROS log file
      ROS_INFO_STREAM("Hello from ROS Node: " << ros::this_node::getName());

      // Create a ROS Subscriber to IMAGE_TOPIC with a queue_size of 1 and a callback function to cloud_cb
      ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 1, cloud_cb);

      // Create a ROS publisher to PUBLISH_TOPIC with a queue_size of 1
      pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);

      // Spin
      ros::spin();

      // Success
      return 0;
}
