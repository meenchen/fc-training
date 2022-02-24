/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 * 	- Ji Lin, jilin@mit.ed
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#include <stdint.h>
typedef int8_t q7_t;
typedef uint8_t q8_t;
typedef int16_t q15_t;
typedef uint16_t q16_t;
typedef int32_t q31_t;
typedef uint32_t q32_t;

typedef enum {
	STATE_SUCCESS = 0, /* No error */
	PARAM_NO_SUPPORT = 1, /* Unsupported parameters */
} tinyengine_status;


typedef struct add_params{
	int input_h, input_w, input_c, left_shift;
	int input1_offset, input1_multiplier, input1_shift;
	int input2_offset, input2_multiplier, input2_shift;
	int output_offset, output_multiplier, output_shift;
	int quantized_activation_max, quantized_activation_min;

} ADD_params;

#define TN_MAX(A,B) ((A) > (B) ? (A) : (B))
#define TN_MIN(A,B) ((A) < (B) ? (A) : (B))


#include "genInclude.h"
