#pragma once

#include <opencv2/opencv.hpp>
#include <ScreenVision.h>
#include <thread>

#include "ViewCalibrator.hpp"

namespace vt
{
	
	class MaskGenerator
	{
	public:

		MaskGenerator();

		void start(const Webcam& webcam, const ViewCalibrator& calibration);

		void segment(const cv::UMat& view, cv::UMat& foreground_mask, cv::UMat& shadow_mask);

		void stop();
	
	private:

		void predictor_process(ViewProperties properties);

		void read_prediction(cv::UMat& dst);
	
	private:

		cv::UMat m_View, m_Background, m_Difference, m_Score;
		cv::UMat m_ForegroundView, m_BackgroundMask;
		cv::UMat m_SharpeningKernel, m_MorphKernel;
		cv::UMat m_NoiseMask, m_BorderMask;
		float m_AmbientIntensity = 0.0f;

		
		// Capture Thread Resources
		std::thread m_PredictionThread;
		std::mutex m_PredictionMutex;
		cv::Mat m_NextPrediction;
		cv::Mat m_RawFrame;
		bool m_Runflag;

		// Frame Queue
		std::vector<cv::Mat> m_FrameQueue;
		size_t m_WriteIndex = 0;
	};



}
