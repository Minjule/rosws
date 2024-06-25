#include <ros/ros.h>

// Include pcl
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <thread>

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

// Topics
static const std::string IMAGE_TOPIC = "/camera/depth/points";
static const std::string PUBLISH_TOPIC = "/pcl/transformed/points";

ros::Publisher pub;

typedef pcl::PointXYZ PointType;
pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr transformedz_cloud(new pcl::PointCloud<pcl::PointXYZ>());
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

pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> clustering;

using namespace std::chrono_literals;
std::vector<int> mapping;

bool cloud_added = false;
void func(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){

    /*pcl::PCLPointCloud2 pcl_pc2;
    
    pcl_conversions::toPCL(*cloud_msg, pcl_pc2);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);*/
    
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    
    Eigen::Affine3f camera_pose = Eigen::Affine3f::Identity();
    camera_pose.rotate(Eigen::AngleAxisf(M_PI/6, Eigen::Vector3f::UnitX()));
    
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    Eigen::Affine3f transformx = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
    transformx.rotate(Eigen::AngleAxisf(-M_PI/6, Eigen::Vector3f::UnitX()));
    
    pcl::transformPointCloud(*cloud, *transformedz_cloud, transform);
    pcl::transformPointCloud(*transformedz_cloud, *transformed_cloud, transformx);
    //pcl::transformPointCloud(*transformedz_cloud, *transformedz_cloud, camera_pose.inverse());
    
    filter.setInputCloud(transformed_cloud);
    filter.setFilterFieldName("z");
    filter.setFilterLimits(0.0, 4.0);
    filter.filter(*filtered_cloud);
    
    sor.setInputCloud(filtered_cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*voxeled_cloud);
    
    std::vector<int> indices_(voxeled_cloud->points.size());
    for (int i = 0; i < voxeled_cloud->points.size(); ++i){
        indices_[i] = i;
    }
    
    int i, x1, x2, x3, y1, y2, y3, z1, z2, z3;
    float a, b, c, d;
    for(i=0; i<1000; i++){
	    std::random_shuffle(indices_.begin(), indices_.end());
	    pcl::PointIndices::Ptr inliers_(new pcl::PointIndices);
	    
	    inliers_->indices.push_back(indices_[0]);
    	    inliers_->indices.push_back(indices_[1]);
            inliers_->indices.push_back(indices_[2]);
            
            x1 = indices_[0][0]; x2 = indices_[1][0]; x3 = indices_[2][0];
            y1 = indices_[0][1]; y2 = indices_[1][1]; y3 = indices_[2][1];
            z1 = indices_[0][2]; z2 = indices_[1][2]; z3 = indices_[2][2];
            
	    a = (y2 - y1)*(z3 - z1) - (z2 - z1)*(y3 - y1);
            b = (z2 - z1)*(x3 - x1) - (x2 - x1)*(z3 - z1);
            c = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1);
            d = -(a*x1 + b*y1 + c*z1);
            float plane_length = std::max(0.1, std::sqrt(a*a + b*b + c*c));
            
            for (int i = 0; i < voxeled_cloud.size(); ++i) {
            
                if (std::find(inliers_.begin(), inliers_.end(), i) != inliers_.end())
                    continue;

                float x = voxeled_cloud[i][0];
                float y = voxeled_cloud[i][1];
                float z = voxeled_cloud[i][2];

                float distance = std::fabs(a * x + b * y + c * z + d) / plane_length;

                if (distance <= 4)
                    inliers_.push_back(i);
            }
            if (inliers_.size() > inliers_result.size())
                inliers_result = inliers_;
    	} 
    	
     	std::vector<std::vector<float>> inlier_points;
        std::vector<std::vector<float>> outlier_points;
        
        for (int i = 0; i < voxeled_cloud.size(); ++i) {
            if (std::find(inliers_result.begin(), inliers_result.end(), i) != inliers_result.end())
                inlier_points.push_back(point_cloud[i]);
            else
                outlier_points.push_back(point_cloud[i]);
        }
    
    
    sensor_msgs::PointCloud2 cloud_msg_transformed;
    pcl::toROSMsg(*color_extracted_inliers, cloud_msg_transformed);
    
    viewer.removeAllPointClouds();
    viewer.addPointCloud<pcl::PointXYZ>(extracted_inliers, "largest cluster");
    //viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], "largest cluster");
    
    pub.publish(cloud_msg_transformed);
}
int main (int argc, char** argv){
      ros::init (argc, argv, "pcl_ros");
      ros::NodeHandle nh;

      ROS_INFO_STREAM("Hello from ROS Node: " << ros::this_node::getName());
      ros::Subscriber sub = nh.subscribe(IMAGE_TOPIC, 1, func);
      
      viewer.setBackgroundColor(0.0, 0.0, 0.0);
      viewer.addCoordinateSystem(1.0);
      viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");


      pub = nh.advertise<sensor_msgs::PointCloud2>(PUBLISH_TOPIC, 1);
      
      while (ros::ok() && !viewer.wasStopped())
    {
        viewer.spinOnce(100);
        ros::spinOnce(); // Handle ROS callbacks
    }
      return 0;
}
