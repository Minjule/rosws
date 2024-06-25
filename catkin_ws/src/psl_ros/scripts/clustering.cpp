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

pcl::SACSegmentation<pcl::PointXYZ> seg;
pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    
pcl::SACSegmentation<pcl::PointXYZ> seg2;
pcl::ModelCoefficients::Ptr coefficients2(new pcl::ModelCoefficients);
pcl::PointIndices::Ptr inliers2(new pcl::PointIndices);
pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_inliers2(new pcl::PointCloud<pcl::PointXYZ>);


pcl::PassThrough<pcl::PointXYZ> filter;
pcl::VoxelGrid<pcl::PointXYZ> sor;

pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> clustering;

pcl::PointXYZ target_point;

std::vector<int> mapping;
std::vector<pcl::PointIndices> clusters;

pcl::PointCloud<pcl::PointXYZ>::Ptr plane(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr ball(new pcl::PointCloud<pcl::PointXYZ>);

double min_distance = std::numeric_limits<double>::max();

void func(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    	auto start = std::chrono::high_resolution_clock::now();

    	pcl::fromROSMsg(*cloud_msg, *cloud);
	pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    
	filter.setInputCloud(cloud);
	filter.filter(*filteredCloud);

    	sor.setInputCloud(filteredCloud);
    	sor.filter(*filteredCloud);
    	
    	cloud = filteredCloud;

	kdtree->setInputCloud(filteredCloud);

	normalEstimation.setInputCloud(filteredCloud);
	normalEstimation.setRadiusSearch(0.03);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);

	clustering.setSearchMethod(kdtree);
	clustering.setInputCloud(filteredCloud);
	clustering.setInputNormals(normals);
	clustering.extract(clusters);

        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setMaxIterations(100);
    
       seg2.setOptimizeCoefficients(true);
       seg2.setModelType(pcl::SACMODEL_SPHERE);
       seg2.setMethodType(pcl::SAC_RANSAC);
       seg2.setDistanceThreshold(0.005); // Set your desired distance threshold here
       seg2.setMaxIterations(60);
       seg2.setRadiusLimits(0.07, 0.11);
    
       seg2.setInputCloud(filteredCloud);
       seg2.segment(*inliers2, *coefficients2);

       seg.setInputCloud(filteredCloud);
       seg.segment(*inliers, *coefficients);

    	int currentClusterNum = 1;
    	float horizontal_tolerance = pcl::deg2rad(20.0);
    	viewer->removeAllPointClouds();

	for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i){
		pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
			cluster->points.push_back(cloud->points[*point]);

		cluster->width = cluster->points.size();
		cluster->height = 1;
		cluster->is_dense = true;
        
        	seg2.setInputCloud(cluster);
        	seg2.segment(*inliers2, *coefficients2);
        	extract2.setInputCloud(cluster);
        	extract2.setIndices(inliers2);
        	extract2.setNegative(false);
        
        	std::cerr.setstate(std::ios_base::failbit);
        	extract2.filter(*ball);
        	std::cerr.clear();

        	if(ball->points.size() > 400){
           		std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;

           		std::stringstream ss;
           		ss << "cluster_" << currentClusterNum;
           		std::string cluster_name = ss.str();
        
            		std::vector<float> color = colors[currentClusterNum % colors.size()];
            		viewer->addPointCloud<pcl::PointXYZ>(ball, cluster_name);
            		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], cluster_name);
            		currentClusterNum++;
        	}
        
    	}
    for (const auto& point : ball->points) {
            double distance = pcl::euclideanDistance(target_point, point);
            if (distance < min_distance) {
                min_distance = distance;
            }
    }
    //std::cout << "Minimum distance to detected object: " << min_distance << " meters" << std::endl;
    pcl::toROSMsg(*ball, cloud_msg_transformed);
    auto end = std::chrono::high_resolution_clock::now();
    pub.publish(cloud_msg_transformed);
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << elapsed_seconds.count() << " seconds" << std::endl;
}
int main(int argc, char** argv) {

    	ros::init (argc, argv, "pcl_ros");
    	ros::NodeHandle nh;

    	target_point.x = 0.0;
    	target_point.y = 0.0;
    	target_point.z = 0.0;

    	ROS_INFO_STREAM("Hello from ROS Node: " << ros::this_node::getName());
    	ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 1, func);
    
    	filter.setFilterFieldName("z");
	filter.setFilterLimits(0.0, 3.0);
	
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	
	clustering.setMinClusterSize(100);
	clustering.setMaxClusterSize(6000);
	clustering.setNumberOfNeighbours(30);
	clustering.setSmoothnessThreshold(7.0 / 180.0 * M_PI);
	clustering.setCurvatureThreshold(30.0);
	
    	pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);
    	viewer->setBackgroundColor(0, 0, 0);
    	viewer->addCoordinateSystem(1.0, "coordinate_system");
      
    	while (ros::ok() && !viewer->wasStopped()){
        	viewer->spinOnce(100);
        	ros::spinOnce(); // Handle ROS callbacks
    	}
    
    	return 0;
}
