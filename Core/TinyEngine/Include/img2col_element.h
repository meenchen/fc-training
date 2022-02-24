/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author: Wei-Ming Chen, wmchen@mit.edu
 * -------------------------------------------------------------------- */

#ifndef ARMNN_INCLUDE_IMG2COL_ELEMENT_H_
#define ARMNN_INCLUDE_IMG2COL_ELEMENT_H_

#define q7_q15_offset_ele(src,dst)													\
/* convert from q7 to q15 and then store the results in the destination buffer */	\
in_q7x4 = arm_nn_read_q7x4_ia((const q7_t **)&src);									\
/* Extract and sign extend each of the four q7 values to q15 */					 	\
in_q15x2_1 = __SXTB16(__ROR(in_q7x4, 8));										 	\
in_q15x2_2 = __SXTB16(in_q7x4);													 	\
																					\
out_q15x2_2 = __PKHTB(in_q15x2_1, in_q15x2_2, 16);									\
/* Maximum of 9 bits from the addition is expected */								\
out_q15x2_2 = __SADD16(out_q15x2_2, offset_q15x2);									\
																					\
out_q15x2_1 = __PKHBT(in_q15x2_2, in_q15x2_1, 16);									\
out_q15x2_1 = __SADD16(out_q15x2_1, offset_q15x2);									\
																					\
write_q15x2_ia(&dst, out_q15x2_1);													\
write_q15x2_ia(&dst, out_q15x2_2);

#define q8_q15_offset_ele(src,dst)													\
/* convert from q8 to q15 and then store the results in the destination buffer */	\
in_q7x4 = arm_nn_read_q7x4_ia((const q8_t **)&src);											 	\
/* Extend each of the four q8 values to q15 */					 	\
in_q15x2_1 = __UXTB16(__ROR(in_q7x4, 8));										 	\
in_q15x2_2 = __UXTB16(in_q7x4);													 	\
																					\
out_q15x2_2 = __PKHTB(in_q15x2_1, in_q15x2_2, 16);									\
/* Maximum of 9 bits from the addition is expected */								\
out_q15x2_2 = __SADD16(out_q15x2_2, offset_q15x2);									\
																					\
out_q15x2_1 = __PKHBT(in_q15x2_2, in_q15x2_1, 16);									\
out_q15x2_1 = __SADD16(out_q15x2_1, offset_q15x2);									\
																					\
write_q15x2_ia(&dst, out_q15x2_1);													\
write_q15x2_ia(&dst, out_q15x2_2);

#define q7_q15_offset_reordered_ele(src,dst)\
/* convert from q7 to q15 and then store the results in the destination buffer */\
in_q7x4 = arm_nn_read_q7x4_ia((const q7_t **)&src);\
\
/* Extract and sign extend each of the four q7 values to q15 */\
out_q15x2_1 = __SXTB16(__ROR(in_q7x4, 8));\
out_q15x2_2 = __SXTB16(in_q7x4);\
\
out_q15x2_1 = __SADD16(out_q15x2_1, offset_q15x2);\
out_q15x2_2 = __SADD16(out_q15x2_2, offset_q15x2);\
\
write_q15x2_ia(&dst, out_q15x2_2);\
write_q15x2_ia(&dst, out_q15x2_1);

#endif /* ARMNN_INCLUDE_IMG2COL_ELEMENT_H_ */
