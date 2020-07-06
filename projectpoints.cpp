#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <random>
#include <iostream>

#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

std::vector<cv::Point3d> Generate3DPoints(double xlidar);
std::vector<cv::Point3d> RandomTranslateGenerated3DPoints(std::vector<cv::Point3d> input);
float RandomNumber(float Min, float Max);

int main(int argc, char *argv[])
{
    srand (static_cast <unsigned> (time(0)));
    cv::RNG rng(cv::getCPUTickCount());

    //string path_camera_raw = "/media/ballardini/4tb/KITTI/unsync_2011_10_03/2011_10_03_drive_0027_extract/image_00/data/0000000113.png";

    for (double xlidar=15.0;xlidar>0.0;xlidar=xlidar-0.1)
    //for (double xlidar=15.0;xlidar>0.0;xlidar=xlidar-1.0)
    {

        //cv::Mat kitti;
        //kitti = cv::imread(path_camera_raw,cv::IMREAD_COLOR);   // Read the file
        //kitti.setTo(cv::Scalar(255,255,255));

        Mat kitti(512, 1392, CV_8UC3, Scalar(255,255,255));


        // Read 3D points
        std::vector<cv::Point3d> objectPoints = Generate3DPoints(xlidar);
        std::vector<cv::Point2d> imagePoints;
        std::vector<cv::Point2d> imagePointsUndistort;

        Matx33f K_00   (9.799200e+02, 0.000000e+00, 6.900000e+02,
                        0.000000e+00, 9.741183e+02, 2.486443e+02,
                        0.000000e+00, 0.000000e+00, 1.000000e+00);

        Matx33d R_00   (1.000000e+00, 0.000000e+00, 0.000000e+00,
                        0.000000e+00, 1.000000e+00, 0.000000e+00,
                        0.000000e+00, 0.000000e+00, 1.000000e+00);

        Matx44d velo_to_cam_RT   ( 7.967514e-03, -9.999679e-01, -8.462264e-04, -1.377769e-02,
                                   -2.771053e-03,  8.241710e-04, -9.999958e-01, -5.542117e-02,
                                   9.999644e-01,  7.969825e-03, -2.764397e-03, -2.918589e-01,
                                   0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00);


        for (auto & element : objectPoints) {
            cv::Matx41d P1;
            P1(0) = element.x;
            P1(1) = element.y;
            P1(2) = element.z;
            P1(3) = 1.0f;

            //cout << velo_to_cam_RT << endl;
            //cout << element << "\t";

            cv::Matx41d P2 = velo_to_cam_RT*element;
            element.x = P2(0);
            element.y = P2(1);
            element.z = P2(2);

            //cout << element << endl;;
        }

        cv::Mat distCoeffs(1, 5, cv::DataType<double>::type);
        distCoeffs.at<double>(0) = -3.745594e-01;
        distCoeffs.at<double>(1) =  2.049385e-01;
        distCoeffs.at<double>(2) =  1.110145e-03;
        distCoeffs.at<double>(3) =  1.379375e-03;
        distCoeffs.at<double>(4) = -7.084798e-02;

        cv::Mat T_00(3, 1, cv::DataType<double>::type);
        T_00.at<double>(0) = -9.251859e-17;
        T_00.at<double>(1) =  8.326673e-17;
        T_00.at<double>(2) = -7.401487e-17;

        cv::Mat distCoeffsZero;
        distCoeffsZero.zeros(1,5,cv::DataType<double>::type);

        cv::Mat rVec(3, 1, cv::DataType<double>::type);
        cv::Rodrigues(R_00,rVec);

        /// Evaluate the projection in the "distorted" image (imagePoints) and
        /// in the image without distortion
        cv::projectPoints(objectPoints, R_00, T_00, K_00, distCoeffs, imagePoints);
        cv::projectPoints(objectPoints, R_00, T_00, K_00, distCoeffsZero, imagePointsUndistort);

        /// First approach; do not use the points that are outside the image
        /// plane above/after some amount of "space"
        double x_offset = -150;
        double y_offset = -150;
        for (int i=0; i<imagePoints.size();i++)
        {
            //printf("%f %f %f %f ", xlidar, objectPoints[0].x, objectPoints[0].y, objectPoints[0].z);

            /// First approach
            if (
                ((imagePointsUndistort.at(i).x>.00+x_offset) && (imagePointsUndistort.at(i).y>0.0+y_offset)) &&
                ((imagePointsUndistort.at(i).x<kitti.size[1]-x_offset) && (imagePointsUndistort.at(i).y<kitti.size[0]-y_offset))
               )

            /// Second approach; the ratio is a way more difficult to "control".
            //if (fabs((imagePoints[i].x / imagePointsUndistort[i].x)) > 0.8)
            {
                // draw the "distorted" points
                cv::circle(kitti,cv::Point(imagePoints.at(i).x,imagePoints.at(i).y),1,cv::Scalar(0,0,255),-1);

                // draw the undistorted points, for visual debuggin
                //cv::circle(kitti,cv::Point(imagePointsUndistort.at(i).x,imagePointsUndistort.at(i).y),1,cv::Scalar(255,0,0),-1);
            }

            /// Write something; I used this is for the version of 1-projected-point only
            /// trying to identify when the point returns back to the imageplane
            //if (((imagePoints.at(i).x>.00) && (imagePoints.at(i).y>0.0)) && ((imagePoints.at(i).x<kitti.size[1]) && (imagePoints.at(i).y<kitti.size[0])))
            //{
            //    cv::putText(kitti, "BUG", cv::Point(25,60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,0,255), 2);
            //}
            //else
            //    cv::putText(kitti, "BUG", cv::Point(25,60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,255,255), 2);
        }

        /*for (auto & element : imagePoints)
        {
            cv::circle(kitti,cv::Point(element.x,element.y),1,cv::Scalar(0,0,255),-1);
            if (((element.x>.00) && (element.y>0.0)) && ((element.x<kitti.size[1]) && (element.y<kitti.size[0])))
            {
                cv::putText(kitti, "BUG", cv::Point(25,60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,0,255), 2);
            }
            else
                cv::putText(kitti, "BUG", cv::Point(25,60), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,255,255), 2);

        }*/

        cv::putText(kitti, std::to_string(xlidar), cv::Point(25,25), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,0,255), 2);

        cv::imshow( "raw with distorsion", kitti);
        int k = cv::waitKey(10);

        if (k == 113)
            break;

    }

    cv::waitKey(0);

    return 0;
}

/**
 * @brief Generate3DPoints
 * @param xlidar - the distance of the plane to be generated
 * @return the list of 3D points
 *
 * generate points in the "velodyne" reference frame; see the
 * http://www.cvlibs.net/datasets/kitti/setup.php to undestand the
 * reference system
 */
std::vector<cv::Point3d> Generate3DPoints(double xlidar)
{
    std::vector<cv::Point3d> points;

    //for(int y=-160;y<160;y++) // too extended..

    for(int y=-80;y<80;y++) // this is ok for me
        for(int z=-15;z<15;z++)
        {   //if (abs(z)<2)         //these two were use for debugging
            //   if (abs(y)==20)    //these two were use for debugging
                    points.push_back(cv::Point3d(xlidar/1.0f, y/10.0f, z/10.0f));
        }

    /// Just one point
    //points.push_back(cv::Point3d(xlidar/1.0f, 20/10.0f, 0/10.0f));

    // Python original part for generating points
    // data = []
    // for y in range(-160,160,1):
    //     for z in range(-15,15,1):
    //         data.append([xlidar/1., y/10., z/10., 1.])
    // data = np.array(data)

    return points;
}
