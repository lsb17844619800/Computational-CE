#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

#define ENABLE_LOG 1

vector<String> img_names;
bool preview = false;
bool try_gpu = true;
double work_megapix = 1;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.7f;
string features_type = "orb";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
string save_graph_to;
string warp_type = "cylindrical";
int expos_comp_type = ExposureCompensator::NO;
float match_conf = 0.3f;
string seam_find_type = "dp_color";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 5;
string result_name = "result.jpg";


int main(int argc, char* argv[])
{
	
	double ttt = getTickCount();
	
	img_names.push_back("E://edge//all2//1.jpg");
	img_names.push_back("E://edge//all2//2.jpg");
	img_names.push_back("E://edge//all2//3.jpg");
	img_names.push_back("E://edge//all2//4.jpg");




	int64 app_start_time = getTickCount();

	setBreakOnError(true);

	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		cout << "two small！！！" << endl;
		return -1;
	}

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	
	int64 t = getTickCount();

	Ptr<FeaturesFinder> finder;
	if (features_type == "surf")
	{
		finder = new SurfFeaturesFinder();
	}
	else if (features_type == "orb")
	{
		finder = new OrbFeaturesFinder();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;



	for (int i = 0; i < num_images; ++i)
	{
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty())
		{
			cout << "no！！！ " << img_names[i] << endl;
			return -1;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;
		//cout << "图 # " << i + 1 << " 特征点个数: " << features[i].keypoints.size() << endl;

		resize(full_img, img, Size(), seam_scale, seam_scale);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	

	
	t = getTickCount();
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(try_gpu, match_conf);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();
	

	// Check if we should save matches graph
	if (save_graph)
	{
		
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		cout << "！！！" << endl;
		return -1;
	}
	
	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
	else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	(*adjuster)(features, pairwise_matches, cameras);

	

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);
	vector<UMat> masks_warped(num_images);
	vector<UMat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<Mat> masks(num_images);

	// Preapre images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;

		if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
		else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();


	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);
	
	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type == "gc_color")
	{
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		
		return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();
	
	t = getTickCount();
	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;
	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		

		
		full_img = imread(img_names[img_idx]);
		if (!is_compose_scale_set)
		{
			if (compose_megapix > 0)
				compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
			is_compose_scale_set = true;

			
			//compose_seam_aspect = compose_scale / seam_scale;
			compose_work_aspect = compose_scale / work_scale;

			
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			
			for (int i = 0; i < num_images; ++i)
			{
				
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(full_img, img, Size(), compose_scale, compose_scale);
		else
			img = full_img;
		full_img.release();
		Size img_size = img.size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		
		warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = seam_mask & mask_warped;

		if (blender.empty())
		{
			blender = Blender::createDefault(blend_type, try_gpu);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_gpu);
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
				fb->setSharpness(1.f / blend_width);
				cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
			}
			blender->prepare(corners, sizes);
		}

		
		blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	}

	Mat result, result_mask;

	blender->blend(result, result_mask);

	

	imwrite(result_name, result);
	result.convertTo(result, CV_8UC1);
	imshow("stitch", result);
	ttt = ((double)getTickCount() - ttt) / getTickFrequency();
	
	vector<int>first_heng;
	for (int i = 0; i < result.rows; i++)
	{
		int m = 0;
		for (int j = 0; j < result.cols; j++)
		{
			if (result.at<Vec3b>(i, j)[0] != 0 && result.at<Vec3b>(i, j)[1] != 0 && result.at<Vec3b>(i, j)[2] != 0)
				m++;
		}
		if (m > result.cols*0.98)
			first_heng.push_back(i);
	}
	int min_first_row = *min_element(first_heng.begin(), first_heng.end());
	int max_first_row = *max_element(first_heng.begin(), first_heng.end());
	vector<int>first_cols;
	for (int j = result.cols - 1; j > 0; j--)
	{
		if (result.at<Vec3b>(min_first_row, j)[0] != 0 && result.at<Vec3b>(min_first_row, j)[1] != 0 && result.at<Vec3b>(min_first_row, j)[2] != 0)
			first_cols.push_back(j);
	}
	int first_col_first = *max_element(first_cols.begin(), first_cols.end());

	vector<int>final_first_cols;
	for (int j = 0; j < result.cols - 1; j++)
	{
		if (result.at<Vec3b>(max_first_row, j)[0] != 0 && result.at<Vec3b>(max_first_row, j)[1] != 0 && result.at<Vec3b>(max_first_row, j)[2] != 0)
			final_first_cols.push_back(j);
	}
	int first_col_final = *min_element(final_first_cols.begin(), final_first_cols.end());

	Mat first_roi;
	first_roi = result(Rect(first_col_final, min_first_row, first_col_first - first_col_final, max_first_row - min_first_row));
	imwrite("first_roi.jpg", first_roi);
	/*vector<int>second_heng;
	for (int i = 0; i < first_roi.rows; i++)
	{
		int m = 0;
		for (int j = 0; j < first_roi.cols; j++)
		{
			if (first_roi.at<Vec3b>(i, j)[0] != 0 && first_roi.at<Vec3b>(i, j)[1] != 0 && first_roi.at<Vec3b>(i, j)[2] != 0)
				m++;
		}
		if (m > first_roi.cols*0.9)
			second_heng.push_back(i);
	}
	int min_second_row= *min_element(second_heng.begin(), second_heng.end());
	int max_second_row= *max_element(second_heng.begin(), second_heng.end());
	vector<int>second_cols;
	for (int j = first_roi.cols - 1; j > 0; j--)
	{
		if (first_roi.at<Vec3b>(min_second_row, j)[0] != 0 && first_roi.at<Vec3b>(min_second_row, j)[1] != 0 && first_roi.at<Vec3b>(min_second_row, j)[2] != 0)
			second_cols.push_back(j);
	}
	int second_col_first = *max_element(second_cols.begin(), second_cols.end());

	vector<int>final_second_cols;
	for (int j = 0; j < first_roi.cols - 1; j++)
	{
		if (first_roi.at<Vec3b>(max_second_row, j)[0] != 0 && first_roi.at<Vec3b>(max_second_row, j)[1] != 0 && first_roi.at<Vec3b>(max_second_row, j)[2] != 0)
			final_second_cols.push_back(j);
	}
	int second_col_final = *min_element(final_second_cols.begin(), final_second_cols.end());

	Mat second_roi;
	second_roi = first_roi(Rect(second_col_final, min_second_row, second_col_first- second_col_final, max_second_row - min_second_row));
	imwrite("second_roi.jpg", second_roi);*/
	waitKey(0);

	
	return 0;
}