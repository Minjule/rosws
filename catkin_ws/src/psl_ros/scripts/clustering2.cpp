#include <ros/ros.h>

// Include pcl
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <thread>
#include <chrono>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sstream>
#include <pcl/common/colors.h>
#include <pcl/common/common.h> // For pcl::isFinite
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/angles.h>

#include <pcl/segmentation/segment_differences.h>
#include <pcl/common/distances.h>

static const std::string IMAGE_TOPIC = "/camera/depth/points";
static const std::string PUBLISH_TOPIC = "/pcl/transformed/points";

ros::Publisher pub;

sensor_msgs::PointCloud2 cloud_msg_transformed;
pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cluster Viewer"));


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
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
    
pcl::ExtractIndices<pcl::PointXYZ> extract;
pcl::ExtractIndices<pcl::PointXYZ> extract2;
pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_inliers(new pcl::PointCloud<pcl::PointXYZ>);

pcl::PassThrough<pcl::PointXYZ> filter;
pcl::VoxelGrid<pcl::PointXYZ> sor;

pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> clustering;

pcl::PointXYZ target_point;

void func(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    auto start = std::chrono::high_resolution_clock::now();

    pcl::fromROSMsg(*cloud_msg, *cloud);

    std::vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    
	filter.setInputCloud(cloud);
	// Filter out all points with Z values not in the [0-2] range.
	filter.setFilterFieldName("z");
	filter.setFilterLimits(0.0, 3.0);

	filter.filter(*filteredCloud);

    sor.setInputCloud(filteredCloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*filteredCloud);
    cloud = filteredCloud;

	kdtree->setInputCloud(filteredCloud);

	normalEstimation.setInputCloud(filteredCloud);
	normalEstimation.setRadiusSearch(0.03);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

    // Region growing clustering object.
	clustering.setMinClusterSize(100);
	clustering.setMaxClusterSize(6000);
	clustering.setSearchMethod(kdtree);
	clustering.setNumberOfNeighbours(30);
	clustering.setInputCloud(filteredCloud);
	clustering.setInputNormals(normals);
    // Set the angle in radians that will be the smoothness threshold
	// (the maximum allowable deviation of the normals).
	clustering.setSmoothnessThreshold(7.0 / 180.0 * M_PI); // 7 degrees.
	// Set the curvature threshold. The disparity between curvatures will be
	// tested after the normal deviation check has passed.
	clustering.setCurvatureThreshold(30.0);

	std::vector<pcl::PointIndices> clusters;
	clustering.extract(clusters);

    //pcl::visualization::PCLVisualizer::Ptr viewer2(new pcl::visualization::PCLVisualizer("Largest Cluster Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem(1.0, "coordinate_system");
    //viewer2->addPointCloud<pcl::PointXYZ>(filteredCloud, "cloud");

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // Set the segmentation parameters
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);
    seg.setMaxIterations(100);

    // Perform plane segmentation
    seg.setInputCloud(filteredCloud);
    seg.segment(*inliers, *coefficients);

    // Create a new point cloud containing only the plane points
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::SegmentDifferences<pcl::PointXYZ> diff;

    // For every cluster...
	int currentClusterNum = 1;
    int largest_cluster_index = 0;
    size_t largest_cluster_size = 0;
    float horizontal_tolerance = pcl::deg2rad(20.0);
    pcl::PointIndices::Ptr indicestoremove(new pcl::PointIndices());
    Eigen::Vector4f centroid;
    viewer->removeAllPointClouds();

	for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
			cluster->points.push_back(cloud->points[*point]);

        pcl::compute3DCentroid(*cluster, centroid);

		cluster->width = cluster->points.size();
		cluster->height = 1;
		cluster->is_dense = true;
        

        seg.setInputCloud(cluster);
        seg.segment(*inliers, *coefficients);
        extract.setInputCloud(cluster);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*plane);

        Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        float angle_with_vertical = pcl::getAngle3D(plane_normal, Eigen::Vector3f::UnitX());

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
        if (angle_with_vertical < horizontal_tolerance) {
            for (const auto& index : clusters[currentClusterNum].indices) {
               indicestoremove->indices.push_back(index);
            }
            //viewer->addPointCloud<pcl::PointXYZ>(plane, cluster_name);
            //viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], cluster_name);
        }
		currentClusterNum++;
	}
    std::cout << "filtered cloud before " << filteredCloud->points.size() << std::endl;
    extract.setInputCloud(filteredCloud);
    extract.setIndices(indicestoremove);
    extract.setNegative(true);
    extract.filter(*filteredCloud);
    std::cout << "filtered cloud after " << filteredCloud->points.size() << std::endl;

    normalEstimation.setInputCloud(filteredCloud);
    normalEstimation.compute(*normals);
    clustering.setInputCloud(filteredCloud);
	clustering.setInputNormals(normals);
    clustering.setSmoothnessThreshold(60.0 / 180.0 * M_PI); 
	clustering.setCurvatureThreshold(30.0);

	std::vector<pcl::PointIndices> clusters2;
	clustering.extract(clusters2);

    std::vector<Eigen::Vector4f> centroids(clusters2.size());
    for (std::vector<pcl::PointIndices>::const_iterator i = clusters2.begin(); i != clusters2.end(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
			cluster->points.push_back(filteredCloud->points[*point]);

		cluster->width = cluster->points.size();
		cluster->height = 1;
		cluster->is_dense = true;

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
    //viewer->removeAllPointClouds();

    //viewer->addPointCloud<pcl::PointXYZ>(filteredCloud, "plane");
    //std::vector<float> color = colors[largest_cluster_index % colors.size()];
    
    //viewer2->addPointCloud<pcl::PointXYZ>(cluster, "largest cluster");
    //viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], "largest cluster");
    //viewer2->spin();

    //pcl::toROSMsg(*closest_cluster, cloud_msg_transformed);
    auto end = std::chrono::high_resolution_clock::now();
    //pub.publish(cloud_msg_transformed);
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << " seconds" << std::endl;
}
int main(int argc, char** argv) {

    ros::init (argc, argv, "pcl_ros");
    ros::NodeHandle nh;

    target_point.x = 0.0; // Replace with the x-coordinate of the detected object
      target_point.y = 0.0; // Replace with the y-coordinate of the detected object
      target_point.z = 0.0;

    ROS_INFO_STREAM("Hello from ROS Node: " << ros::this_node::getName());
    ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 1, func);
    
    //pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);
      
    while (ros::ok() && !viewer->wasStopped())
    {
        viewer->spinOnce(100);
        ros::spinOnce(); // Handle ROS callbacks
    }
    
    return 0;
}