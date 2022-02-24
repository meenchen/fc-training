/*
 * detectionUtility.h
 *
 *  Created on: Jan 26, 2021
 *      Author: wmchen
 */

#ifndef TINYENGINE_INCLUDE_DETECTIONUTILITY_H_
#define TINYENGINE_INCLUDE_DETECTIONUTILITY_H_

int postProcessing(signed char *input, unsigned char* runtime_buffer,
		int y_zero, float y_scale, int shape_x, int shape_y, int shape_c, int resolution,
		int width, int height , float conf_thresh, float out_boxes[10][6]);


#endif /* TINYENGINE_INCLUDE_DETECTIONUTILITY_H_ */
