/*
 * File:
 *  linuxcamera.c
 *
 * Description:
 *  Linux camera support for Lua
 *
 * Author:
 *  Jonghoon Jin // jhjin0@gmail.com
 *  Marko Vitez // marko@vitez.it
 */

#include <luaT.h>
#include <TH/TH.h>
#include <stdio.h>
#include <unistd.h>
#include "videocap.h"

#define BYTE2FLOAT 0.003921568f // 1/255

/* yuv420p-to-rgbp lookup table */
static short TB_YUR[256], TB_YUB[256], TB_YUGU[256], TB_YUGV[256], TB_Y[256];
static unsigned char TB_SAT[1024 + 1024 + 256];

static void *vcap;
static int vcap_w, vcap_h, vcap_fps, vcap_nframes;

// This function calculates a lookup table for yuv420p-to-rgbp conversion
static void yuv420p_rgbp_LUT()
{
	int i;

	for (i = 0; i < 256; i++) {
		TB_YUR[i]  =  459 * (i-128) / 256;
		TB_YUB[i]  =  541 * (i-128) / 256;
		TB_YUGU[i] = -137 * (i-128) / 256;
		TB_YUGV[i] = - 55 * (i-128) / 256;
		TB_Y[i]    = (i-16) * 298 / 256;
	}
	for (i = 0; i < 1024; i++) {
		TB_SAT[i] = 0;
		TB_SAT[i + 1024 + 256] = 255;
	}
	for (i = 0; i < 256; i++)
		TB_SAT[i + 1024] = i;
}

// YUYV to RGB byte
static void yuyv2torchRGB(const unsigned char *frame, unsigned char *dst_byte, int imgstride, int rowstride, int w, int h)
{
	int i, j, w2 = w / 2;
	unsigned char *dst;
	const unsigned char *src;

	/* convert for R channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_byte + i * rowstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUR[ src[3] ] + 1024];
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUR[ src[3] ] + 1024];
			src += 4;
		}
	}

	/* convert for G channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_byte + i * rowstride + imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024];
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024];
			src += 4;
		}
	}

	/* convert for B channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_byte + i * rowstride + 2*imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUB[ src[1] ] + 1024];
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUB[ src[1] ] + 1024];
			src += 4;
		}
	}
}

// YUYV to RGB float
static void yuyv2torchfloatRGB(const unsigned char *frame, float *dst_float, int imgstride, int rowstride, int w, int h)
{
	int i, j, w2 = w / 2;
	float *dst;
	const unsigned char *src;

	/* convert for R channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_float + i * rowstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUR[ src[3] ] + 1024] * BYTE2FLOAT;
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUR[ src[3] ] + 1024] * BYTE2FLOAT;
			src += 4;
		}
	}

	/* convert for G channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_float + i * rowstride + imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024] * BYTE2FLOAT;
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024] * BYTE2FLOAT;
			src += 4;
		}
	}

	/* convert for B channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_float + i * rowstride + 2*imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUB[ src[1] ] + 1024] * BYTE2FLOAT;
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUB[ src[1] ] + 1024] * BYTE2FLOAT;
			src += 4;
		}
	}
}

// Initialize camera: device, width, height, fps, number of buffers
static int capture(lua_State *L)
{
	const char *device = lua_tostring(L, 1);
	int w = lua_tointeger(L, 2);
	int h = lua_tointeger(L, 3);
	int nbuffers = lua_tointeger(L, 5);
	int rc;
	
	if(vcap)
		videocap_close(vcap);
	vcap_nframes = 0;
	vcap = videocap_open(device);
	vcap_fps = lua_tointeger(L, 4);
	if(!vcap)
		luaL_error(L, "Error opening device %s", device);
	rc = videocap_startcapture(vcap, w, h, V4L2_PIX_FMT_YUYV, vcap_fps, nbuffers ? nbuffers : 1);
	if(rc < 0)
	{
		videocap_close(vcap);
		vcap = 0;
		luaL_error(L, "Error %d starting capture", rc);
	}
	vcap_w = w;
	vcap_h = h;
	return 0;
}

// Get next frame from camera
static int frame_rgb(lua_State * L)
{
	int dim = 0;
	long *stride = NULL;
	long *size = NULL;
	unsigned char *dst_byte = NULL;
	float *dst_float = NULL;
	char *frame;
	struct timeval tv;

	const char *tname = luaT_typename(L, 1);
	if (strcmp("torch.ByteTensor", tname) == 0)
	{
		THByteTensor *frame = luaT_toudata(L, 1, luaT_typenameid(L, "torch.ByteTensor"));

		// get tensor's Info
		dst_byte = THByteTensor_data(frame);
		dim = frame->nDimension;
		stride = &frame->stride[0];
		size = &frame->size[0];

	} else if (strcmp("torch.FloatTensor", tname) == 0)
	{
		THFloatTensor *frame = luaT_toudata(L, 1, luaT_typenameid(L, "torch.FloatTensor"));

		// get tensor's Info
		dst_float = THFloatTensor_data(frame);
		dim = frame->nDimension;
		stride = &frame->stride[0];
		size = &frame->size[0];
	} else luaL_error(L, "<linuxcamera>: cannot process tensor type %s", tname);

	if ((3 != dim) || (3 != size[0]))
		luaL_error(L, "<linuxcamera>: cannot process tensor of this dimension and size");

	if(!vcap)
		luaL_error(L, "<linuxcamera>: call capture first");

	// Get the frame from the V4L2 device using our videocap library
	int rc = videocap_getframe(vcap, &frame, &tv);
	if(rc < 0)
		luaL_error(L, "<linuxcamera>: videocap_getframe returned error %d", rc);
	// Convert image from YUYV to RGB torch tensor
	if(dst_byte)
		yuyv2torchRGB((unsigned char *)frame, dst_byte, stride[0], stride[1], vcap_w, vcap_h);
	else yuyv2torchfloatRGB((unsigned char *)frame, dst_float, stride[0], stride[1], vcap_w, vcap_h);
	return 0;
}

// Close camera
static int stop(lua_State * L)
{
	if(vcap)
	{
		videocap_close(vcap);
		vcap = 0;
	}
	return 0;
}

static const struct luaL_reg linuxcamera[] = {
	{"capture", capture},
	{"frame_rgb", frame_rgb},
	{"stop", stop},
	{NULL, NULL}
};

// Initialize the library
int luaopen_linuxcamera(lua_State * L)
{
	luaL_register(L, "linuxcamera", linuxcamera);
	yuv420p_rgbp_LUT();
	return 1;
}
