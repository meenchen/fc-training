/*
 * yoloOutput.h
 *
 *  Created on: Nov 15, 2021
 *      Author: wmchen
 */

typedef struct box{
	float x0;
	float y0;
	float x1;
	float y1;
	float score;
} det_box;

det_box** postprocessing(signed char *input_data[3], signed char y_zero[3], float y_scale[3],
		unsigned char *data_buf, int w, int h, int output_c, int num_classes, const int anchors[3][3][2], int outputs,
		const float NMS_threshold, const float VALID_THRESHOLD, int* box_ret, det_box** ret_box);

det_box** postprocessing_fp(float *input_data[3], signed char y_zero[3], float y_scale[3],
		unsigned char *data_buf, int w, int h, int output_c, int num_classes, const int anchors[3][3][2], int outputs,
		const float NMS_threshold, const float VALID_THRESHOLD, int* box_ret, det_box** ret_box);
