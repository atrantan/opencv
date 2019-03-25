//  Copyright (c) 2019 Antoine TRAN TAN

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <array>

// static void help()
// {
//     cout << "\nThis program demonstrates kmeans clustering.\n"
//             "It generates an image with random points, then assigns a random number of cluster\n"
//             "centers and uses kmeans to move those cluster centers to their representitive location\n"
//             "Call\n"
//             "./kmeans\n" << endl;
// }

int main( int /*argc*/, char** /*argv*/ )
{
    const int MAX_CLUSTERS = 5;
    std::array<cv::Scalar,5> colorTab =
    {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0,255,0),
        cv::Scalar(255,100,100),
        cv::Scalar(255,0,255),
        cv::Scalar(0,255,255)
    };

    cv::Mat img(500, 500, CV_8UC3);
    cv::RNG rng(12345);

    for(;;)
    {
        int k, clusterCount = rng.uniform(2, MAX_CLUSTERS+1);
        int i, sampleCount = rng.uniform(1, 1001);
        cv::Mat points(sampleCount, 1, CV_32FC2), labels;

        clusterCount = MIN(clusterCount, sampleCount);
        std::vector<cv::Point2f> centers;

        /* generate random sample from multigaussian distribution */
        for( k = 0; k < clusterCount; k++ )
        {
            cv::Point center;
            center.x = rng.uniform(0, img.cols);
            center.y = rng.uniform(0, img.rows);
            cv::Mat pointChunk =
                points.rowRange(
                    k*sampleCount/clusterCount,
                    k == clusterCount - 1 ? sampleCount : (k+1)*sampleCount/clusterCount);
            rng.fill(
                pointChunk,
                cv::RNG::NORMAL,
                cv::Scalar(center.x, center.y),
                cv::Scalar(img.cols*0.05, img.rows*0.05));
        }

        randShuffle(points, 1, &rng);

        double compactness =
            cv::kmeans(
                points,
                clusterCount,
                labels,
                cv::TermCriteria(
                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
                3,
                cv::KMEANS_PP_CENTERS,
                centers);

        img = cv::Scalar::all(0);

        for( i = 0; i < sampleCount; i++ )
        {
            int clusterIdx = labels.at<int>(i);
            cv::Point ipt = points.at<cv::Point2f>(i);
            cv::circle( img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA );
        }
        for (i = 0; i < (int)centers.size(); ++i)
        {
            cv::Point2f c = centers[i];
            cv::circle( img, c, 40, colorTab[i], 1, cv::LINE_AA );
        }
        std::cout << "Compactness: " << compactness << std::endl;

        cv::imshow("clusters", img);

        char key = (char)cv::waitKey();
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }

    return 0;
}
