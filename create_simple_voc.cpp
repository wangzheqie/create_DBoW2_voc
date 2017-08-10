/*************************************************************************
 > File Name: create_simple_voc.cpp
 > Author:wangzheqie
 > Mail:wangzheqie@gmail.com
 > Created Time: Fri 21 Jul 2017 10:13:52 AM UTC
 ************************************************************************/

#include <cstdlib>
#include <iostream>
#include <vector>
#include <bitset>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "Thirdparty/DBoW2/DBoW2/DBoW2.h"

using namespace DBoW2;
using namespace cv;
using namespace std;

void changeStructure(const vector<int> &plain, vector<vector<int>> &out,
		int size);
int main() {

	OrbFeatureDetector detector;
	vector<KeyPoint> kps;

	OrbDescriptorExtractor extractor;
	Mat dspts;

	typedef TemplatedVocabulary<FORB::TDescriptor, FORB> ORBVocabulary;

	//!!Note: 9 and 3 is the parameters you may want to change
	ORBVocabulary orb(9, 3, TF_IDF, L1_NORM);

	int NIMAGES;
	typedef cv::Mat mat;

	vector<vector<mat>> features;

	//load images
	//data.txt contain the images' name, left and right images have the same name
	//and each name lies on a single line in data.txt
	string imPathLeft = "images/left";
	string imPathRight = "images/right";
	string imagefile = "images/data.txt";
	vector<string> imLeft;
	vector<string> imRight;
	vector<double> timestamp;

	ifstream file;
	file.open(imagefile.c_str());
	double t;
	string name;
	stringstream ss;
	while (!file.eof()) {
		string s;
		getline(file, s);
		if (!s.empty()) {
			ss << s;
			imLeft.push_back(imPathLeft + "/" + ss.str() + ".png");
			imRight.push_back(imPathRight + "/" + ss.str() + "png");
			ss >> t;
			timestamp.push_back(t / 1e9);
		}
	}

	NIMAGES = imLeft.size();
	cout << "images now is :" << NIMAGES << endl;

	vector<float> des;

	features.clear();
	features.reserve(NIMAGES);

	for (int n = 0; n < NIMAGES; n++) {
		cv::Mat im = imread(imLeft.at(n));
		detector.detect(im, kps);
		extractor.compute(im, kps, dspts);

		for (int i = 0; i < kps.size(); i++) {
			cv::Mat single_des = dspts.row(i);
//			cout << single_des << endl;

			vector<vector<mat>>::const_iterator it1;
			vector<mat>::const_iterator it2;

			//must pre-initialize, to allocate memery
			features.push_back(vector<mat>());
			vector<mat> tmp;

			tmp.push_back(single_des);
			features.push_back(tmp);

		}
		cout << "features size now is: " << features.size() << endl;
	}
	cout << "creating the vocabulary..." << endl;
	orb.create(features);
//		cout << orb << endl;

	cout << "saving the vocabulary to text file..." << endl;
	orb.saveToTextFile("voc.txt");

	return 0;
}

