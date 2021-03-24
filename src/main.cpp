#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <vector>

#include "my_header.h"

namespace fs = std::filesystem;

int main()
{
    // Generate .csv file
    {
        // Output file
        std::ofstream ostrm("my_csv_file.csv", std::ios_base::out);

        int label = 0;

        fs::path path = fs::current_path() / "examples";

        for (auto &&p : fs::directory_iterator(path))
        {
            for (auto &&q : fs::directory_iterator(p))
            {
                std::string entry = q.path().string() + ";" + std::to_string(label) + "\n";
                std::replace(entry.begin(), entry.end(), '\\', '/');
                ostrm << entry;
            }
            label++;
        }
    }

    {
        // Get the path to your CSV.
        std::string fn_csv = "my_csv_file.csv";
        // These vectors hold the images and corresponding labels.
        std::vector<cv::Mat> images;
        std::vector<int> labels;
        // Read in the data. This can fail if no valid
        // input filename is given.
        try
        {
            read_csv(fn_csv, images, labels);
        }
        catch (const cv::Exception &e)
        {
            std::cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << std::endl;
            // nothing more we can do
            exit(1);
        }
        // Quit if there are not enough images for this demo.
        if (images.size() <= 1)
        {
            std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
            ::CV_Error(cv::Error::Code::StsError, error_message);
        }
        // The following lines simply get the last images from
        // your dataset and remove it from the vector. This is
        // done, so that the training data (which we learn the
        // cv::face::LBPHFaceRecognizer on) and the test data we test
        // the model with, do not overlap.
        cv::Mat testSample = images[images.size() - 1];
        int testLabel = labels[labels.size() - 1];
        images.pop_back();
        labels.pop_back();
        // The following lines create an LBPH model for
        // face recognition and train it with the images and
        // labels read from the given CSV file.
        //
        // The LBPHFaceRecognizer uses Extended Local Binary Patterns
        // (it's probably configurable with other operators at a later
        // point), and has the following default values
        //
        //      radius = 1
        //      neighbors = 8
        //      grid_x = 8
        //      grid_y = 8
        //
        // So if you want a LBPH FaceRecognizer using a radius of
        // 2 and 16 neighbors, call the factory method with:
        //
        //      cv::face::LBPHFaceRecognizer::create(2, 16);
        //
        // And if you want a threshold (e.g. 123.0) call it with its default values:
        //
        //      cv::face::LBPHFaceRecognizer::create(1,8,8,8,123.0)
        //
        cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
        model->train(images, labels);
        // The following line predicts the label of a given
        // test image:
        int predictedLabel = model->predict(testSample);
        //
        // To get the confidence of a prediction call the model with:
        //
        //      int predictedLabel = -1;
        //      double confidence = 0.0;
        //      model->predict(testSample, predictedLabel, confidence);
        //
        std::string result_message = 
            cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);

        std::cout << result_message << std::endl;
        // First we'll use it to set the threshold of the LBPHFaceRecognizer
        // to 0.0 without retraining the model. This can be useful if
        // you are evaluating the model:
        //
        model->setThreshold(0.0);
        // Now the threshold of this model is set to 0.0. A prediction
        // now returns -1, as it's impossible to have a distance below
        // it
        predictedLabel = model->predict(testSample);
        std::cout << "Predicted class = " << predictedLabel << std::endl;
        // Show some informations about the model, as there's no cool
        // Model data to display as in Eigenfaces/Fisherfaces.
        // Due to efficiency reasons the LBP images are not stored
        // within the model:
        std::cout << "Model Information:" << std::endl;
        std::string model_info = 
            cv::format(
                "\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
                model->getRadius(),
                model->getNeighbors(),
                model->getGridX(),
                model->getGridY(),
                model->getThreshold());
        std::cout << model_info << std::endl;
        // We could get the histograms for example:
        std::vector<cv::Mat> histograms = model->getHistograms();
        // But should I really visualize it? Probably the length is interesting:
        std::cout << "Size of the histograms: " << histograms[0].total() << std::endl;
    }

    return 0;
}

static void read_csv(
    const std::string &filename,
    std::vector<cv::Mat> &images,
    std::vector<int> &labels,
    char separator)
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file)
    {
        std::string error_message = "No valid input file was given, please check the given filename.";
        ::CV_Error(cv::Error::StsBadArg, error_message);
    }
    std::string line, path, classlabel;
    while (std::getline(file, line))
    {
        std::stringstream liness(line);
        std::getline(liness, path, separator);
        std::getline(liness, classlabel);
        if (!path.empty() && !classlabel.empty())
        {
            images.push_back(cv::imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}
