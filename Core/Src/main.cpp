/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.edu
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#include "main.h"
#include "stdio.h"
#include "../testing_data/images.h"
#include "golden_data.h"
extern "C"{
#include "tinyengine_function.h"
#include "genNN.h"
}

#define IMAGE_H 80
#define IMAGE_W 80
uint16_t color;
void example_VWW(const int8_t* image) {
	signed char *input = getInput();
    int i;
    for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
            input[i] = *image++;
    }
    invoke();
    uint8_t* output = (uint8_t*)getOutput();
    uint8_t P = output[0], NP = output[1];

    if (P > NP){
    	printf("It's a person");
    }
    else{
    	printf("It's not a person");
    }
}

#define INPUT_CH 160
#define OUTPUT_CH 2
#define IMAGES 6

void SystemClock_Config(void);
void StartDefaultTask(void const * argument);
float feat_fp[INPUT_CH];
int8_t feat[INPUT_CH];
float w[INPUT_CH * OUTPUT_CH];
float b[OUTPUT_CH];
float out[OUTPUT_CH];
float dw[OUTPUT_CH*INPUT_CH];
float lr = 0.1;

const int label[6] = {0, 0, 0, 1, 1, 1};

void train_one_img(const int8_t* img, int cls)
{
  int i;
  signed char *input = getInput();
  const int8_t* image = img;
  for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
	  input[i] = *image++;
  }
  invoke();
  signed char *output = getOutput();
  for (i = 0; i < INPUT_CH; i++){
	  feat_fp[i] = (output[i] - zero_x)*scale_x;
  }

  // out = new_w @ feat + new_b
  fully_connected_fp(feat_fp, 1, 1, INPUT_CH, OUTPUT_CH, b, w, out);

  // softmax = _stable_softmax(out)
  statble_softmax_inplace(out, OUTPUT_CH);

  out[cls] -= 1;

  //dw = dy.reshape(-1, 1) @ feat.reshape(1, -1)
  mat_mul_fp(out, OUTPUT_CH, 1, feat_fp, INPUT_CH, dw);

  for (i = 0; i < OUTPUT_CH * INPUT_CH; i++){
	  w[i] = w[i] - lr * dw[i];
  }
  //new_w = new_w - lr * dw
  //new_b = new_b - lr *
  b[0] = b[0] - lr * out[0];
  b[1] = b[1] - lr * out[1];
}

void train_one_feat(const float* feat, int cls)
{
  int i;
  signed char *input = getInput();
  for (i = 0; i < IMAGE_H * IMAGE_W * 3; i++){
	  input[i] = feat[i];
  }

  // out = new_w @ feat + new_b
  fully_connected_fp(feat, 1, 1, INPUT_CH, OUTPUT_CH, b, w, out);

  // softmax = _stable_softmax(out)
  statble_softmax_inplace(out, OUTPUT_CH);

  out[cls] -= 1;

  //dw = dy.reshape(-1, 1) @ feat.reshape(1, -1)
  mat_mul_fp(out, OUTPUT_CH, 1, feat, INPUT_CH, dw);

  for (i = 0; i < OUTPUT_CH * INPUT_CH; i++){
	  w[i] = w[i] - lr * dw[i];
  }
  //new_w = new_w - lr * dw
  //new_b = new_b - lr *
  b[0] = b[0] - lr * out[0];
  b[1] = b[1] - lr * out[1];
}


int main(void)
{
  HAL_Init();

  SystemClock_Config();

  int i;
  for (i = 0; i < INPUT_CH*OUTPUT_CH; i++)
	  w[i] = new_w[i];
  for (i = 0; i < OUTPUT_CH; i++)
	  b[i] = new_b[i];
  uint32_t start, end;
  start = HAL_GetTick();
  train_one_feat(feat_mcunetv3_assets_vww_noperson1_jpg, 0);
  train_one_feat(feat_mcunetv3_assets_vww_noperson2_jpg, 0);
  train_one_feat(feat_mcunetv3_assets_vww_noperson3_jpg, 0);
  train_one_feat(feat_mcunetv3_assets_vww_person1_jpg, 1);
  train_one_feat(feat_mcunetv3_assets_vww_person2_jpg, 1);
  train_one_feat(feat_mcunetv3_assets_vww_person3_jpg, 1);
  end = HAL_GetTick();

  printf("Elapse time: %d ms\n", end - start);
}

void SystemClock_Config(void)
{
	RCC_ClkInitTypeDef RCC_ClkInitStruct;
	RCC_OscInitTypeDef RCC_OscInitStruct;
	HAL_StatusTypeDef ret = HAL_OK;

	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLM = 25;
	RCC_OscInitStruct.PLL.PLLN = 432;
	RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
	RCC_OscInitStruct.PLL.PLLQ = 9;

	ret = HAL_RCC_OscConfig(&RCC_OscInitStruct);
	if (ret != HAL_OK) {
		while (1) {
			;
		}
	}

	ret = HAL_PWREx_EnableOverDrive();
	if (ret != HAL_OK) {
		while (1) {
			;
		}
	}

	RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK
			| RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

	ret = HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7);
	if (ret != HAL_OK) {
		while (1) {
			;
		}
	}
}


void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}
