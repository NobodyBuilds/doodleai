#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <random>
#include <cstdio>
#include <norender.h>
#include <algorithm>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <numeric>
#include <cmath>
int threads=256;
int blocks(int n){
return (n + threads - 1) / threads;
}
cudaError_t err;
struct Model
{
	float lr = 0.1f;
	int inputnode = 28 * 28;
	int outputnode = 9;
	int batchsize = 32;
	float outputbias = 0.0f;
	float outval = 0.0f;
};
Model model;

struct Layer
{
	int Nin;
	int Nout;
	int wIdx;
	int dIdx;
	int bIdx;
};
int width = 500;
int height = 500;
int pixelX = 28;
int pixelY = 28;
int row = 0;
int col = 0;
bool lmb = false;
bool Rmb = false;
std::vector<Layer> layer;
std::vector<float> weights;
std::vector<float> bias;
std::vector<float> guesses;
std::vector<const char*> names={"circle", "square", "triangle", "star", "house",
        "tree", "sun", "fish", "flower"};
int layers = 0;
int weightsBufferSize = 0;
int nodeDataSize = 0;
int biasSize = 0;
double mousex, mousey;
double prevmousex = 0, prevmousey = 0;
double dmx = 0, dmy = 0;
float radius = 1.0f;
constexpr float TRAINING_INPUT_MEAN = 39.8232f;
constexpr float TRAINING_INPUT_STDDEV = 80.4317f;
constexpr int PREPROCESS_TARGET_SIZE = 24;

void addlayer(int in, int out)
{
	Layer l;
	l.Nin = in;
	l.Nout = out;
	l.wIdx = 0;
	l.dIdx = 0;
	l.bIdx = 0;
	layer.push_back(l);
	layers++;
	weightsBufferSize += in * out;
	nodeDataSize += out;
	biasSize += out;
}

void setoffsets()
{

	for (int i = 0; i < layers; i++)
	{
		int k = i - 1;
		layer[i].wIdx = i == 0 ? 0 : layer[k].wIdx + layer[k].Nin * layer[k].Nout;
		layer[i].dIdx = i == 0 ? 0 : layer[k].dIdx + layer[k].Nout;
		layer[i].bIdx = i == 0 ? 0 : layer[k].bIdx + layer[k].Nout;
	}
}

void initlayers()
{

	addlayer(model.inputnode, 512);
	addlayer(512, 512);
	addlayer(512, 256);
	addlayer(256, 128);
	addlayer(128, 64);
	addlayer(64, model.outputnode);

	setoffsets();
}

float *dWeights = nullptr;
float *dNodeData = nullptr;
float *dBias = nullptr;
uint8_t *data = nullptr;
float* inputs=nullptr;

__constant__ Model dModel;

void initcontant()
{
	cudaMemcpyToSymbol(dModel, &model, sizeof(Model));
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("!ERROR! constant struct failed %s \n", cudaGetErrorString(err));
	}
	else
	{
		printf("struct on gpu copied \n");
	}
}

void initWB()
{

	weights.resize(weightsBufferSize);

	bias.resize(biasSize);

	// weights
	std::fstream weightsfile("D:/visual_studio/cudaNeuralNet/cudaNeuralNet/weights/weights.txt", std::ios::in);
	if (!weightsfile.is_open())
	{
		std::cerr << "Error opening weights file" << std::endl;
	}
	std::string w;
	int wi = 0;
	int bi = 0;
	while (weightsfile >> w)
	{

		weights[wi++] = std::stof(w);
	}

	weightsfile.close();

	// bias
	std::fstream biasfile("D:/visual_studio/cudaNeuralNet/cudaNeuralNet/weights/bias.txt", std::ios::in);
	if (!biasfile.is_open())
	{
		std::cerr << "Error opening bias file" << std::endl;
	}
	std::string b;
	while (biasfile >> b)
	{

		bias[bi++] = std::stof(b);
	}

	biasfile.close();

	if (wi != weightsBufferSize || bi != biasSize)
	{
		printf("outdated data delete old weights and bias\n");

		return;
	}
}
void initgpu()
{

	cudaMalloc(&dWeights, weightsBufferSize * sizeof(float));
	cudaMalloc(&dBias, biasSize * sizeof(float));
	cudaMalloc(&dNodeData, nodeDataSize * sizeof(float));
	cudaMalloc(&data, 28 * 28 * sizeof(uint8_t));
	cudaMalloc(&inputs, 28 * 28 * sizeof(float));

	cudaMemcpy(dWeights, weights.data(), weightsBufferSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dBias, bias.data(), biasSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(data, 0, 28 * 28 * sizeof(uint8_t));
	cudaMemset(inputs, 0, 28 * 28 * sizeof(float));
	cudaError_t err = cudaGetLastError();
	if (err)
	{
		printf("init gpu  failed :%s \n", cudaGetErrorString(err));
	}
}
void freegpu()
{
	cudaFree(dWeights);
	cudaFree(dBias);
	cudaFree(dNodeData);
	cudaFree(data);
	cudaFree(inputs);
}
void initmodel()
{
	initlayers();
	initWB();
	guesses.resize(model.outputnode);
}

static cudaGraphicsResource *screenres = nullptr;
void registertexture()
{

	unsigned int id = vbo_id.quad_texture_tex(pixelX, pixelY, 1);
	cudaError_t err = cudaGraphicsGLRegisterImage(&screenres, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	if (err != cudaSuccess)
	{
		std::cerr << "Failed to register OpenGL texture with CUDA: " << cudaGetErrorString(err) << std::endl;
	}
}

void unregisterbuffer()
{
	if (screenres)
	{
		cudaGraphicsUnregisterResource(screenres);
		screenres = nullptr;
	}
}
__global__ void texkernel(cudaSurfaceObject_t surf, int w, int h, uint8_t *data)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= w || y >= h)
		return;
	float d = data[y * w + i]/255.0f;
	float4 val = make_float4(d, d, d, 1.0f);

	surf2Dwrite(val, surf, i * sizeof(float4), y);
}
__global__ void datakernel(int w, int h, uint8_t *data, int row, int col, float radius, bool rmb, bool lmb)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h)
		return;
	uint8_t val;
	if (lmb)
	{
		float dx = x - col;
		float dy = y - row;
		float dist = sqrtf(dx * dx + dy * dy);
		if (dist < radius)
		{

			val =255;
			data[y * w + x] = val;
		}
	}
	if (rmb)
	{
		float dx = x - col;
		float dy = y - row;
		float dist = sqrtf(dx * dx + dy * dy);
		if (dist < radius*2.0f)
		{

			val = 0;
			data[y * w + x] = val;
		}
	}
}

void draw()
{
	dim3 blocks(16, 16);
	dim3 grid((pixelX + 15) / 16, (pixelY + 15) / 16);
	datakernel<<<grid, blocks>>>(pixelX, pixelY, data, row, col, radius, Rmb, lmb);
	cudaError_t err = cudaGetLastError();
	if (err)
	{
		printf("drawing kernel failed :%s \n", cudaGetErrorString(err));
	}
}
void render(){
	cudaGraphicsMapResources(1, &screenres);
	cudaArray_t arr;
	cudaGraphicsSubResourceGetMappedArray(&arr, screenres, 0, 0);

	cudaResourceDesc desc{};
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = arr;

	cudaSurfaceObject_t surf;
	cudaCreateSurfaceObject(&surf, &desc);
	dim3 blocks(16, 16);
	dim3 grid((pixelX + 15) / 16, (pixelY + 15) / 16);
	texkernel<<<grid, blocks>>>(surf, 28, 28, data);
	cudaError_t err = cudaGetLastError();
	if (err)
	{
		printf("texture kernel failed :%s \n", cudaGetErrorString(err));
	}
	cudaDestroySurfaceObject(surf);
	cudaGraphicsUnmapResources(1, &screenres);

	render2d.quadtexbyinterop(800, 450, width, height, pixelX, pixelY, 1);
}

void normalize(){
	const int inputSize = pixelX * pixelY;
	static std::vector<uint8_t> hostCanvas(inputSize);
	static std::vector<float> hostInputs(inputSize);
	const float background = (0.0f - TRAINING_INPUT_MEAN) / TRAINING_INPUT_STDDEV;

	cudaError_t copyErr = cudaMemcpy(hostCanvas.data(), data, inputSize * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if (copyErr != cudaSuccess)
	{
		printf("canvas copy failed :%s \n", cudaGetErrorString(copyErr));
		return;
	}

	int minX = pixelX;
	int minY = pixelY;
	int maxX = -1;
	int maxY = -1;
	for (int rawY = 0; rawY < pixelY; rawY++)
	{
		for (int x = 0; x < pixelX; x++)
		{
			if (hostCanvas[rawY * pixelX + x] == 0)
				continue;

			int topDownY = pixelY - 1 - rawY;
			minX = std::min(minX, x);
			maxX = std::max(maxX, x);
			minY = std::min(minY, topDownY);
			maxY = std::max(maxY, topDownY);
		}
	}

	std::fill(hostInputs.begin(), hostInputs.end(), background);

	if (maxX >= minX && maxY >= minY)
	{
		int boxW = maxX - minX + 1;
		int boxH = maxY - minY + 1;
		float scale = std::min(PREPROCESS_TARGET_SIZE / (float)boxW, PREPROCESS_TARGET_SIZE / (float)boxH);
		int dstW = std::max(1, (int)std::round(boxW * scale));
		int dstH = std::max(1, (int)std::round(boxH * scale));
		int dstX0 = (pixelX - dstW) / 2;
		int dstY0 = (pixelY - dstH) / 2;

		for (int dstY = 0; dstY < dstH; dstY++)
		{
			for (int dstX = 0; dstX < dstW; dstX++)
			{
				int srcX = minX + std::min(boxW - 1, (int)((dstX + 0.5f) * boxW / dstW));
				int srcY = minY + std::min(boxH - 1, (int)((dstY + 0.5f) * boxH / dstH));
				int rawY = pixelY - 1 - srcY;
				uint8_t pixel = hostCanvas[rawY * pixelX + srcX];
				hostInputs[(dstY0 + dstY) * pixelX + dstX0 + dstX] = (float(pixel) - TRAINING_INPUT_MEAN) / TRAINING_INPUT_STDDEV;
			}
		}
	}

	copyErr = cudaMemcpy(inputs, hostInputs.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);
	if (copyErr != cudaSuccess)
	{
		printf("input copy failed :%s \n", cudaGetErrorString(copyErr));
	}
}


__global__ void firstlayer(int n, int inputcount, const float *__restrict__ input, const float *__restrict__ dWeights, const float *__restrict__ dBias,  float *dNodeData)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i >= n)
		return;
float val=dBias[i];
	for (int k = 0; k < inputcount; k++)
	{
		
		val += dWeights[i*inputcount +k ] * input[k];
	}

	dNodeData[i] = (val > 0.f) ? val : 0.01f * val;
}
__device__ float relu(float x)
{
	return (x > 0.f) ? x : 0.01f * x;
}
__global__ void solvelayers(int n, int nin, int w, int d, int b, int p, bool isout, const float *__restrict__ dWeights, const float *__restrict__ dBias, float *dNodeData )
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= n)
		return;

	
	float val= dBias[b+i];
	for (int j = 0; j < nin; j++)
	{
		float v = dNodeData[p+j];
		val += v * dWeights[w + i * nin + j];
	}
	
	dNodeData[d + i] = isout ? val : relu(val);
}


void runnet(){

	normalize();
	
	firstlayer<<<blocks(layer[0].Nout),threads>>>(layer[0].Nout,pixelX*pixelY,inputs,dWeights,dBias,dNodeData);
cudaError_t err = cudaGetLastError();
	if (err)
	{
		printf("first layer  failed :%s \n", cudaGetErrorString(err));
	}
	for(int l=1;l<layers;l++){
            int nin = layer[l - 1].Nout;
			int wl = layer[l].wIdx;
			int d = layer[l].dIdx;
			int b = layer[l].bIdx;
			int p = layer[l - 1].dIdx;
			bool isout = (l == layers - 1);

		solvelayers<<<blocks(layer[l].Nout),threads>>>(layer[l].Nout,nin,wl,d,b,p,isout,dWeights,dBias,dNodeData);
		 err = cudaGetLastError();
	if (err)
	{
		printf("layer solver  failed :%s \n", cudaGetErrorString(err));
	}
	}
	int offset=layer[layers-1].dIdx;
	int Size=nodeDataSize-offset;
    cudaMemcpy(guesses.data() ,dNodeData + offset,Size*sizeof(float),cudaMemcpyDeviceToHost);
	 err = cudaGetLastError();
	if (err)
	{
		printf("output cpy   failed :%s \n", cudaGetErrorString(err));
	}
	
	

}

void showranked()
{
    int n = model.outputnode;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return guesses[a] > guesses[b]; });

    ImGui::Text("Best: %s", names[idx[0]]);

    for (int i : idx) {
        float score = std::clamp(guesses[i], 0.0f, 1.0f) * 100.f;
        ImGui::Text("%s: %.1f%% (raw %.3f)", names[i], score, guesses[i]);
    }
}

void processinput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	lmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	Rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
	const float quadPosX = 800.f, quadPosY = 450.f;
	const float quadLeft = quadPosX - width * 0.5f;
	const float quadBottom = quadPosY - height * 0.5f;

	col = (int)((mousex - quadLeft) / (width / (float)pixelX));
	row = (int)((((900.f - quadBottom) - mousey)) / (height / (float)pixelY));
	if (lmb || Rmb)
	{
		draw();
	}
}
int main()
{

	noRender.init();
	noRender.createWindow(1600, 900, "Neural Network Training", 0);
	noRender.setup2d();
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();

	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(noRender.getwindowid(), true);
	const char *glsl_version = "#version 330";
	ImGui_ImplOpenGL3_Init(glsl_version);
	initmodel();
	initgpu();
	registertexture();
	while (noRender.WindowOpen())
	{
		noRender.processinputs();
		noRender.clearscreen(0.1f, 0.1f, 0.1f);

		glfwGetCursorPos(noRender.getwindowid(), &mousex, &mousey);
		dmx = mousex - prevmousex;
		dmy = mousey - prevmousey;
		prevmousex = mousex;
		prevmousey = mousey;
		processinput(noRender.getwindowid());
		render();
		runnet();
		
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("output");
		showranked();

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		noRender.swapbuffers();
	}
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	unregisterbuffer();
	freegpu();
	noRender.closeWindow();
	return 0;
}
