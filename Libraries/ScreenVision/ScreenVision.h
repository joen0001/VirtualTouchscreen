// Copyright (c) 2023 Sebastian Di Marco
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in 
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LIB_SCREEN_VISION_H
#define LIB_SCREEN_VISION_H

#include <vector>
#include <optional>
#include <opencv2/opencv.hpp>
#include <opencv2/core/directx.hpp>

#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Wincodec.h>
#include <atlbase.h>
#include <DXGITYPE.h>
#include <DXGI1_2.h>
#include <d3d11.h>

#pragma comment(lib, "dxgi")
#pragma comment(lib,"d3d11.lib")

namespace sv
{
	class ScreenCapture
	{
	public:
		// Captures the primary monitor. 
		static std::optional<ScreenCapture> OpenPrimary();

		// Captures monitor that contains the given window handle.
		static std::optional<ScreenCapture> Open(HWND window);

		// Captures monitor that corresponds to the given monitor handle. 
		static std::optional<ScreenCapture> Open(HMONITOR monitor);

		~ScreenCapture() = default;

		bool read(cv::UMat& dst, const uint32_t timeout_ms = 0);

		void operator>>(cv::UMat& dst);

		// TODO: add getters for size, format, etc.

	private:

		struct CaptureContext
		{
			ID3D11Device* m_D3D11Device = nullptr;
			ID3D11DeviceContext* m_D3D11Context = nullptr;
			ID3D11Texture2D* m_StagingTexture = nullptr;
		
			IDXGIOutput1* m_Output = nullptr;
			IDXGIOutputDuplication* m_OutputDuplicator = nullptr;

			~CaptureContext()
			{
				// TODO: fix this
			    //if(m_OutputDuplicator) m_OutputDuplicator->Release();
				//if(m_Output) m_Output->Release();
				//if(m_D3D11Device) m_D3D11Device->Release();
				//if(m_D3D11Context) m_D3D11Context->Release();
			}
		};

		ScreenCapture(CaptureContext&& context);

	
	private:

		CaptureContext m_Context;
	};


	
//---------------------------------------------------------------------------------------------------------------------

	inline std::optional<ScreenCapture> ScreenCapture::OpenPrimary()
	{
		// Point (0,0) is always on the primary monitor.
		return ScreenCapture::Open(MonitorFromPoint(POINT{0,0}, MONITOR_DEFAULTTOPRIMARY));
	}

//---------------------------------------------------------------------------------------------------------------------

	inline std::optional<ScreenCapture> ScreenCapture::Open(HWND window)
	{
		return ScreenCapture::Open(MonitorFromWindow(window, MONITOR_DEFAULTTONULL));
	}

//---------------------------------------------------------------------------------------------------------------------

	inline std::optional<ScreenCapture> ScreenCapture::Open(HMONITOR monitor)
	{
		if(monitor == NULL) return {};

		// Check that screen capture via interop is supported
		auto& opencl_device = cv::ocl::Device::getDefault();
		const bool interop = opencl_device.isExtensionSupported("cl_nv_d3d11_sharing")
			              || opencl_device.isExtensionSupported("cl_khr_d3d11_sharing");
		
		// TODO: add a backup method for when interop isn't supported. 
		if(!interop) return {};


		CV_Assert(SUCCEEDED(CoInitialize(NULL)));

		CComPtr<IDXGIFactory1> dxgi_factory = nullptr;
		if(auto hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)(&dxgi_factory)); FAILED(hr))
			return {};

		// Enumerate through all adapters and their outputs until we find the monitor.
		UINT adapter_idx = 0;
		CComPtr<IDXGIAdapter1> adapter = nullptr;
		while(dxgi_factory->EnumAdapters1(adapter_idx, &adapter) != DXGI_ERROR_NOT_FOUND)
		{
			UINT output_idx = 0;
			CComPtr<IDXGIOutput> output = nullptr;
			while(adapter->EnumOutputs(output_idx, &output) != DXGI_ERROR_NOT_FOUND)
			{
				// Check if the output matches the required monitor.
				DXGI_OUTPUT_DESC output_desc;
				output->GetDesc(&output_desc);

				if(output_desc.Monitor == monitor)
				{
					// We found the monitor, so initialize a screen capture context. 
					ScreenCapture::CaptureContext capture_context;
					D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_9_1;
					
					auto hr = D3D11CreateDevice(
						adapter,
						D3D_DRIVER_TYPE_UNKNOWN,
						NULL,
						0,
						NULL,
						0,
						D3D11_SDK_VERSION,
						&capture_context.m_D3D11Device,
						&feature_level,
						&capture_context.m_D3D11Context
					);
					if(FAILED(hr)) return {}; 

					// Initialize the OpenCL with the D3D11 device for DX-CL interop. 
					// NOTE: this can throw an OpenCV exception, which can be caught.
					cv::directx::ocl::initializeContextFromD3D11Device(capture_context.m_D3D11Device);

					// Add the output and its duplicator to the screen capture context. 
					hr = output->QueryInterface(&capture_context.m_Output);
					if(FAILED(hr)) return {};

					output = nullptr; // TODO: fix this

					hr = capture_context.m_Output->DuplicateOutput(
						capture_context.m_D3D11Device,
						&capture_context.m_OutputDuplicator
					);
					if(FAILED(hr)) return {}; 


					// Create staging texture for the capture
					D3D11_TEXTURE2D_DESC texture_desc = {0};
					texture_desc.Width = output_desc.DesktopCoordinates.right - output_desc.DesktopCoordinates.left;
					texture_desc.Height = output_desc.DesktopCoordinates.bottom - output_desc.DesktopCoordinates.top;
					texture_desc.MipLevels = 1;
					texture_desc.ArraySize = 1;
					texture_desc.SampleDesc.Count = 1;
					texture_desc.SampleDesc.Quality = 0;
					texture_desc.Usage = D3D11_USAGE_DEFAULT;
					texture_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
					texture_desc.BindFlags = 0;
					texture_desc.CPUAccessFlags = 0;
					texture_desc.MiscFlags = 0;

					hr = capture_context.m_D3D11Device->CreateTexture2D(
						&texture_desc,
						NULL,
						&capture_context.m_StagingTexture
					);
					if (FAILED(hr)) return {};

					return ScreenCapture(std::move(capture_context));
				}
				output_idx++;
				output.Release();
			}
		
			adapter_idx++;
			adapter.Release();
		}
		return {};
	}

//---------------------------------------------------------------------------------------------------------------------

	inline ScreenCapture::ScreenCapture(CaptureContext&& context)
		: m_Context(context)
	{
		CV_Assert(context.m_D3D11Device != nullptr);
		CV_Assert(context.m_D3D11Context != nullptr);

		CV_Assert(context.m_Output != nullptr);
		CV_Assert(context.m_OutputDuplicator != nullptr);
	}

//---------------------------------------------------------------------------------------------------------------------

	inline bool ScreenCapture::read(cv::UMat& dst, const uint32_t timeout_ms)
	{
		DXGI_OUTDUPL_FRAME_INFO frame_info = {0};
		CComPtr<IDXGIResource> frame_output = nullptr;
		
		if(!FAILED(m_Context.m_OutputDuplicator->AcquireNextFrame(timeout_ms == 0 ? INFINITE : timeout_ms,&frame_info,&frame_output)))
		{
			ID3D11Texture2D* frame_texture = nullptr;
			frame_output->QueryInterface(&frame_texture);
			m_Context.m_D3D11Context->CopyResource(m_Context.m_StagingTexture, frame_texture);
		
			cv::directx::convertFromD3D11Texture2D(m_Context.m_StagingTexture, dst);

			m_Context.m_OutputDuplicator->ReleaseFrame();
			
			return true;
		}
		return false;
	}


//---------------------------------------------------------------------------------------------------------------------

	inline void ScreenCapture::operator>>(cv::UMat& dst)
	{
		read(dst);
	}

//---------------------------------------------------------------------------------------------------------------------

}

#endif // LIB_SCREEN_VISION_H