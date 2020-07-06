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

std::vector<cv::Point3d> Generate3DPoints();
std::vector<cv::Point3d> RandomTranslateGenerated3DPoints(std::vector<cv::Point3d> input);
float RandomNumber(float Min, float Max);

int main(int argc, char *argv[])
{
    srand (static_cast <unsigned> (time(0)));
    cv::RNG rng(cv::getCPUTickCount());

    // Read 3D points
    std::vector<cv::Point3d> objectPoints = Generate3DPoints();
    std::vector<cv::Point2d> imagePoints;

    // /media/RAIDONE/DATASETS/KITTI/RESIDENTIAL/2011_10_03_drive_0027_sync/calib_cam_to_cam_FUNZIONANONSOPERCHE.txt
    // calib_time: 09-Jan-2012 14:00:15
    // corner_dist: 9.950000e-02
    // S_00: 1.392000e+03 5.120000e+02
    // K_00: 9.799200e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.741183e+02 2.486443e+02 0.000000e+00 0.000000e+00 1.000000e+00
    // D_00: -3.745594e-01 2.049385e-01 1.110145e-03 1.379375e-03 -7.084798e-02
    // R_00: 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00
    // T_00: -9.251859e-17 8.326673e-17 -7.401487e-17

    // K
    // 9.799200e+02 0.000000e+00 6.900000e+02
    // 0.000000e+00 9.741183e+02 2.486443e+02
    // 0.000000e+00 0.000000e+00 1.000000e+00

    cv::Mat cameraMatrixOriginal(3, 3, cv::DataType<double>::type); // Intrisic matrix
    cameraMatrixOriginal.at<double>(0, 0) = 9.799200e+02; //1.6415318549788924e+003;
    cameraMatrixOriginal.at<double>(1, 0) = 0.000000e+00; //0;
    cameraMatrixOriginal.at<double>(2, 0) = 0.000000e+00; //0;

    cameraMatrixOriginal.at<double>(0, 1) = 0.000000e+00; // 0;
    cameraMatrixOriginal.at<double>(1, 1) = 9.741183e+02; // 1.7067753507885654e+003;
    cameraMatrixOriginal.at<double>(2, 1) = 0.000000e+00; // 0;

    cameraMatrixOriginal.at<double>(0, 2) = 6.900000e+02; // 5.3262822453148601e+002;
    cameraMatrixOriginal.at<double>(1, 2) = 2.486443e+02; // 3.8095355839052968e+002;
    cameraMatrixOriginal.at<double>(2, 2) = 1.000000e+00; // 1;

    cv::Mat rVec(3, 1, cv::DataType<double>::type); // Rotation vector
    rVec.at<double>(0) = 0.0f; //-3.9277902400761393e-002;
    rVec.at<double>(1) = 0.0f; //3.7803824407602084e-002;
    rVec.at<double>(2) = 0.0f; //2.6445674487856268e-002;

    cv::Mat tVec(3, 1, cv::DataType<double>::type); // Translation vector
    tVec.at<double>(0) = 0.0f; //2.1158489381208221e+000;
    tVec.at<double>(1) = 0.0f; //-7.6847683212704716e+000;
    tVec.at<double>(2) = 0.0f; //2.6169795190294256e+001;

    cv::Mat distCoeffsOriginal(5, 1, cv::DataType<double>::type);   // Distortion vector
    distCoeffsOriginal.at<double>(0) = -7.9134632415085826e-001;
    distCoeffsOriginal.at<double>(1) = 1.5623584435644169e+000;
    distCoeffsOriginal.at<double>(2) = -3.3916502741726508e-002;
    distCoeffsOriginal.at<double>(3) = -1.3921577146136694e-002;
    distCoeffsOriginal.at<double>(4) = 1.1430734623697941e+002;

    std::cout << "Original Intrisic matrix:   \n" << cameraMatrixOriginal << std::endl << std::endl;
    //std::cout << "Original Rotation vector:   \n" << rVec.t() << std::endl << std::endl;
    //std::cout << "Original Translation vector:\n" << tVec.t() << std::endl << std::endl;
    std::cout << "Original Distortion coef:   \n" << distCoeffsOriginal.t() << std::endl << std::endl;

    /// Generate the PROJECTED points with PARAMETERS
    std::vector<cv::Point2d> projectedPoints;
    cv::projectPoints(objectPoints, rVec, tVec, cameraMatrixOriginal, distCoeffsOriginal, projectedPoints);

    // https://github.com/opencv/opencv/blob/master/samples/cpp/3calibration.cpp
    vector<vector<Point3f>> _objpt;
    vector<vector<Point2f>> _imgpt;
    vector<Mat>             _rvecs, _tvecs;
    cv::Mat                 _cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat                 _distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    cv::Size                _imagesize(1392,512);

    /// Prepare the data for the CALIBRATION TEST, couples of 3D/2D correspondences
    vector<Point3f> panel;
    vector<Point2f> image;
    int j = 4; // number of "views" , ie, panels
    while(j--)
    {
        int i = 100; // how many points for each view
        while(i--)
        {
            //std::cout << "Image point: " << objectPoints[i] << " Projected to " << projectedPoints[i] << std::endl;

            panel.push_back(objectPoints[i]);
            image.push_back(projectedPoints[i]);
        }

        _objpt.push_back(panel);
        _imgpt.push_back(image);

        panel.clear();
        image.clear();
    }

    /// Create some noise to add to the ORIGINAL calibration matrix
    /// A rough calibration is needed
    _cameraMatrix = cameraMatrixOriginal.clone();
    _cameraMatrix.at<double>(0, 0) += rng.uniform(-10.0f,10.0f); //fx
    _cameraMatrix.at<double>(1, 1) += rng.uniform(-10.0f,10.0f); //fy
    _cameraMatrix.at<double>(0, 2) += rng.uniform(-20.0f,20.0f); //cx
    _cameraMatrix.at<double>(1, 2) += rng.uniform(-20.0f,20.0f); //cy

    // print the NOISY camera matrix + dist. coeff;
    // rvecs and tvecs are un-initialized so who cares
    cout << "Original Intrinsic Matrix + Noise\n " << _cameraMatrix << endl << endl;

    // CROSS FINGERS!
    cv::calibrateCamera(_objpt,
                        _imgpt,
                        _imagesize,
                        _cameraMatrix,
                        _distCoeffs,
                        _rvecs,
                        _tvecs,
                        CALIB_USE_INTRINSIC_GUESS);

    cout << "*ESTIMATED* CameraMatrix \n" << _cameraMatrix << endl << endl;
    cout << "*ESTIMATED* Dist. Coeff  \n" << _distCoeffs.t()   << endl << endl;
    //cout << "Rot-Vector   \n" << _rvecs[0]        << endl << endl;
    //cout << "Trans-Vector \n" << _tvecs[0]        << endl << endl;

    // DO IT AGAIN
    cout << "============================================================" << endl;
    cout << "     * * * *   DO IT AGAIN AND CROSS FINGERS    * * * *     " << endl;
    cout << "============================================================" << endl << endl;
    // DO IT AGAIN

    objectPoints = RandomTranslateGenerated3DPoints(objectPoints);
    cv::projectPoints(objectPoints, rVec, tVec, cameraMatrixOriginal, distCoeffsOriginal, projectedPoints);
    _objpt.clear();
    _imgpt.clear();
    _rvecs.clear();
    _tvecs.clear();
    _cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    _distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    /// Prepare the data for the CALIBRATION TEST, couples of 3D/2D correspondences
    panel.clear();
    image.clear();
    j = 4; // number of "views" , ie, panels
    while(j--)
    {
        int i = 100; // how many points for each view
        while(i--)
        {
            //std::cout << "Image point: " << objectPoints[i] << " Projected to " << projectedPoints[i] << std::endl;

            panel.push_back(objectPoints[i]);
            image.push_back(projectedPoints[i]);
        }

        _objpt.push_back(panel);
        _imgpt.push_back(image);

        panel.clear();
        image.clear();
    }

    /// Create some noise to add to the ORIGINAL calibration matrix
    /// A rough calibration is needed
    _cameraMatrix = cameraMatrixOriginal.clone();
    _cameraMatrix.at<double>(0, 0) += rng.uniform(-10.0f,10.0f); //fx
    _cameraMatrix.at<double>(1, 1) += rng.uniform(-10.0f,10.0f); //fy
    _cameraMatrix.at<double>(0, 2) += rng.uniform(-20.0f,20.0f); //cx
    _cameraMatrix.at<double>(1, 2) += rng.uniform(-20.0f,20.0f); //cy

    // print the NOISY camera matrix + dist. coeff;
    // rvecs and tvecs are un-initialized so who cares
    cout << "Intrinsic Matrix + Noise\n " << _cameraMatrix << endl << endl;

    // CROSS FINGERS!
    cv::calibrateCamera(_objpt,
                        _imgpt,
                        _imagesize,
                        _cameraMatrix,
                        _distCoeffs,
                        _rvecs,
                        _tvecs,
                        CALIB_USE_INTRINSIC_GUESS);

    cout << "*ESTIMATED* CameraMatrix \n" << _cameraMatrix << endl << endl;
    cout << "*ESTIMATED* Dist. Coeff  \n" << _distCoeffs.t()   << endl << endl;
    //cout << "Rot-Vector   \n" << _rvecs[0]        << endl << endl;
    //cout << "Trans-Vector \n" << _tvecs[0]        << endl << endl;

    //for(vector<Mat>::iterator it = _rvecs.begin(); it!=_rvecs.end(); it++)
    //    cout << "Rotation Vector\n" << *(it) << endl;
    //for(vector<Mat>::iterator it = _tvecs.begin(); it!=_tvecs.end(); it++)
    //    cout << "Translation Vector\n" << *(it) << endl;

    waitKey(0);
    return 0;
}


float RandomNumber(float Min, float Max)
{

    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

std::vector<cv::Point3d> Generate3DPoints()
{
    std::vector<cv::Point3d> points;

    double x, y, z;

    for(unsigned int i = 0; i < 1000; ++i)
    {
        cv::RNG rng(cv::getCPUTickCount());
        x = rng.uniform(-3.0f, 3.0f);  //RandomNumber(-1,1);
        y = rng.uniform(-3.0f, 3.0f);  //RandomNumber(-1,1);
        z = rng.uniform(10.0f, 20.0f); //RandomNumber(-1,1);
        points.push_back(cv::Point3d(x, y, z));
    }

    if(0)
    for(unsigned int i = 0; i < points.size(); ++i)
    {
        std::cout << points[i] << std::endl << std::endl;
    }

    return points;
}

std::vector<cv::Point3d> RandomTranslateGenerated3DPoints(std::vector<cv::Point3d> input)
{
    double x_, y_, z_;
    cv::RNG rng(cv::getCPUTickCount());
    x_ = rng.uniform(-1.0f, 1.0f); //RandomNumber(-1,1);
    y_ = rng.uniform(-1.0f, 1.0f); //RandomNumber(-1,1);
    z_ = rng.uniform(-1.0f, 1.0f); //RandomNumber(-1,1);
    cv::Point3d noise(x_,y_,z_);

    for(std::vector<cv::Point3d>::iterator it = input.begin(); it!=input.end(); it++)
    {
            //cout << *(it) << endl;
            *(it) = (*it) + noise;
            //cout << *(it) << endl;
    }

    return input;
}
