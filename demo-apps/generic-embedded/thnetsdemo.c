#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <ft2build.h>
#include <libswscale/swscale.h>
#include FT_FREETYPE_H
#include "thnets.h"
#include "libgl.h"
#include "videocap.h"

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
#define NBUFFERS 4
#define MAX_OBJECTS 20
static void *vcap;
static int frame_width, frame_height, win_width, win_height;
static THNETWORK *net;
static char *frames[NBUFFERS];
static int curframe;
const int eye = 231;
const int motion_threshold = 20;
const int motion_downscale = 8;
const int motion_minsize = 64;
const int decaylimit = 3;
const int minstillframes = 3;
const float pdt = 0.5;

#define MAXITEMS 100
static int nitems = 0;
typedef struct {
	int x, y, w, h;
} RECT;
struct ITEM {
	int type;
	RECT rect;
	int width, color;
	char text[100];
	uint8_t *bitmap;
};
static struct ITEM items[MAXITEMS];
static FT_Library ft_lib;
static FT_Face ft_face;

static int rescale(unsigned char *src, int srcw, int srch, int src_stride, int format, unsigned char *dst, int dstw, int dsth, int dst_stride)
{
	struct SwsContext *sws_ctx;
	const uint8_t *srcslice[3];
	uint8_t *dstslice[3];
	int srcstride[3], dststride[3];

	srcslice[0] = src;
	srcstride[0] = src_stride;
	dstslice[0] = dst;
	dststride[0] = dst_stride;

	sws_ctx = sws_getContext(srcw, srch, AV_PIX_FMT_YUYV422, dstw, dsth, format, SWS_FAST_BILINEAR, 0, 0, 0);
	sws_scale(sws_ctx, srcslice, srcstride, 0, srch, dstslice, dststride);
	sws_freeContext(sws_ctx);
	return 0;
}

static void PrepareText(struct ITEM *item)
{
	int i, j, k, stride, bwidth, bheight, asc, desc;
	const char *s = item->text;
	int size = item->width;
	unsigned color = item->color;
	unsigned char alpha = (color >> 24);
	unsigned char red = (color >> 16);
	unsigned char green = (color >> 8);
	unsigned char blue = color;
	FT_Set_Char_Size(ft_face, 0, item->width * 64, 0, 0 );
	int scale = ft_face->size->metrics.y_scale;
	asc = FT_MulFix(ft_face->ascender, scale) / 64;
	desc = -FT_MulFix(ft_face->descender, scale) / 64;
	bheight = asc + desc + 1;
	bwidth = 2 * size * strlen(s);
	unsigned char *bitmap = (unsigned char *)calloc(bheight * bwidth, 4);
	stride = strlen(s) * size * 2 * 4;
	FT_Set_Char_Size(ft_face, 0, 64 * size, 0, 0);
	unsigned char *pbitmap = bitmap;
	for(i = 0; s[i]; i++)
	{
		FT_Load_Char(ft_face, s[i], FT_LOAD_RENDER);
		FT_Bitmap *bmp = &ft_face->glyph->bitmap;
		for(j = 0; j < bmp->rows; j++)
			for(k = 0; k < bmp->width; k++)
			{
				pbitmap[(j + asc - ft_face->glyph->bitmap_top) * stride + 4 * (k + ft_face->glyph->bitmap_left) + 0] = red;
				pbitmap[(j + asc - ft_face->glyph->bitmap_top) * stride + 4 * (k + ft_face->glyph->bitmap_left) + 1] = green;
				pbitmap[(j + asc - ft_face->glyph->bitmap_top) * stride + 4 * (k + ft_face->glyph->bitmap_left) + 2] = blue;
				pbitmap[(j + asc - ft_face->glyph->bitmap_top) * stride + 4 * (k + ft_face->glyph->bitmap_left) + 3] =
					bmp->buffer[j * bmp->pitch + k] * alpha / 255;
			}
		pbitmap += 4 * (ft_face->glyph->advance.x / 64);
	}
	//bwidth = (pbitmap - bitmap) / 4;
	item->rect.w = bwidth;
	item->rect.h = bheight;
	if(item->bitmap)
		free(item->bitmap);
	item->bitmap = bitmap;
	item->type = 3;
}

static void DrawItem(struct ITEM *item)
{
	unsigned color;

	switch(item->type)
	{
	case 1:
		color = (item->color & 0xff00ff00) | ((item->color & 0xff0000)>>16) | ((item->color & 0xff)<<16);
		Blt(&color, 0, 1, 1, item->rect.x, item->rect.y, item->rect.w, item->width);
		Blt(&color, 0, 1, 1, item->rect.x, item->rect.y+item->width, item->width, item->rect.h - 2 * item->width);
		Blt(&color, 0, 1, 1, item->rect.x + item->rect.w - item->width, item->rect.y + item->width, item->width, item->rect.h - 2 * item->width);
		Blt(&color, 0, 1, 1, item->rect.x, item->rect.y + item->rect.h - item->width, item->rect.w, item->width);
		break;
	case 2:
		PrepareText(item);
		if(item->type != 3)
			break;
	case 3:
		Blt(item->bitmap, 0, item->rect.w, item->rect.h, item->rect.x, item->rect.y, item->rect.w, item->rect.h);
		break;
	}
}

static void loadfont()
{
	if(FT_Init_FreeType(&ft_lib))
	{
		fprintf(stderr, "Error initializing the FreeType library\n");
		return;
	}
	if(FT_New_Face(ft_lib, "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 0, &ft_face))
	{
		fprintf(stderr, "Error loading FreeSans font\n");
		return;
	}
}

static double seconds()
{
	static double base;
	struct timeval tv;

	gettimeofday(&tv, 0);
	if(!base)
		base = tv.tv_sec + tv.tv_usec * 1e-6;
	return tv.tv_sec + tv.tv_usec * 1e-6 - base;
}

int rectangle(int x, int y, int w, int h, int width, int color)
{
	if(nitems == MAXITEMS)
		return -1;
	items[nitems].type = 1;
	items[nitems].rect.x = x;
	items[nitems].rect.y = y;
	items[nitems].rect.w = w;
	items[nitems].rect.h = h;
	items[nitems].width = width;
	items[nitems].color = color | 0xff000000;
	nitems++;
	return 0;
}

static int text(int x, int y, const char *text, int width, int color)
{
	if(!ft_face)
		return -1;
	if(nitems == MAXITEMS)
		return -1;
	items[nitems].type = 2;
	items[nitems].rect.x = x;
	items[nitems].rect.y = y;
	strncpy(items[nitems].text, text, sizeof(items[nitems].text)-1);
	items[nitems].width = width;
	items[nitems].color = color | 0xff000000;
	nitems++;
	return 0;
}


static void *rendering_thread(void *dummy)
{
	struct timeval tv;
	int i;

	StartWindow();
	for(;;)
	{
		int rc;
		int fn = 0;

		rc = videocap_getframe(vcap, &frames[fn], &tv);
		if(rc < 0)
			break;
		curframe = fn;
		fn = (fn + 1) % NBUFFERS;
		Blt(frames[curframe], 1, frame_width, frame_height, 0, 0, win_width, win_height);
		for(i = 0; i < nitems; i++)
			DrawItem(items+i);
		Present();
	}
	fprintf(stderr, "Failed to capture from camera\n");
	return 0;
}

struct catp {
	float p;
	char *cat;
};

int catpcmp(const void *a, const void *b)
{
	return (((struct catp *)b)->p - ((struct catp *)a)->p) * 1e8;
}

char **categories;
int ncat;

int loadcategories(const char *modelsdir)
{
	char path[200], s[200], *p;
	FILE *fp;
	
	sprintf(path, "%s/categories.txt", modelsdir);
	fp = fopen(path, "r");
	if(!fp)
		THError("Cannot load %s", path);
	ncat = 0;
	if(fgets(s, sizeof(s), fp))
	while(fgets(s, sizeof(s), fp))
	{
		p = strchr(s, ',');
		if(!p)
			continue;
		ncat++;
	}
	rewind(fp);
	categories = calloc(ncat, sizeof(*categories));
	ncat = 0;
	if(fgets(s, sizeof(s), fp))
	while(fgets(s, sizeof(s), fp))
	{
		p = strchr(s, ',');
		if(!p)
			continue;
		*p = 0;
		categories[ncat++] = strdup(s);
	}
	fclose(fp);
	return 0;
}

void dilate(unsigned char *dst, unsigned char *src, int w, int h, int size)
{
	int x, y, x1, x2, y1, y2;
	
	memset(dst, 0, w*h);
	for(y = 0; y < h; y++)
		for(x = 0; x < w; x++)
			if(src[x + w*y])
			{
				y1 = y - size/2;
				if(y1 < 0)
					y1 = 0;
				y2 = y + size/2;
				if(y2 > h)
					y2 = h;
				while(y1 < y2)
				{
					x1 = x - size/2;
					if(x1 < 0)
						x1 = 0;
					x2 = x + size/2;
					if(x2 > w)
						x2 = w;
					memset(dst + y1*w + x1, 1, x2-x1);
					y1++;
				}
			}
}

void erode(unsigned char *dst, unsigned char *src, int w, int h, int size)
{
	int x, y, x1, x2, y1, y2;
	
	memset(dst, 1, w*h);
	for(y = 0; y < h; y++)
		for(x = 0; x < w; x++)
			if(!src[x + w*y])
			{
				y1 = y - size/2;
				if(y1 < 0)
					y1 = 0;
				y2 = y + size/2;
				if(y2 > h)
					y2 = h;
				while(y1 < y2)
				{
					x1 = x - size/2;
					if(x1 < 0)
						x1 = 0;
					x2 = x + size/2;
					if(x2 > w)
						x2 = w;
					memset(dst + y1*w + x1, 0, x2-x1);
					y1++;
				}
			}
}

int connectedComponent(unsigned char *image, int* coordinates, int coordinatesSize, int height, int width);

void expandrect(RECT *r, int minsize, int image_width, int image_height)
{
	if(r->w < minsize)
	{
		r->x -= (minsize - r->w) / 2;
		r->w = minsize;
	}
	if(r->h < minsize)
	{
		r->y -= (minsize - r->h) / 2;
		r->h = minsize;
	}
	// Expand the motion rectangle to be a square
	if(r->w < r->h)
	{
		r->x -= (r->h - r->w) / 2;
		r->w = r->h;
	} else {
		r->y -= (r->w - r->h) / 2;
		r->h = r->w;
	}
	// Reduce the square, if it's higher of the frame
	if(r->h > image_height)
	{
		r->x += (r->h - image_height) / 2;
		r->y += (r->h - image_height) / 2;
		r->w = r->h = image_height;
	}
	// Put the square inside the frame
	if(r->x + r->w > image_width)
		r->x = image_width - r->w;
	else if(r->x < 0)
		r->x = 0;
	if(r->y + r->h > image_height)
		r->y = image_height - r->h;
	else if(r->y < 0)
		r->y = 0;
}

int nobjects;
unsigned oids;
struct object {
	RECT r;
	unsigned id;
	unsigned color;
	char target, valid, decay, still, permastill;
} objects[MAX_OBJECTS];

float *process(RECT r, unsigned char *rgb)
{
	float *result;
	int outwidth, outheight;

	expandrect(&r, motion_minsize, frame_width, frame_height);
	rescale((unsigned char *)frames[curframe] + 2 * (r.x + r.y * frame_width),
		r.w, r.h, frame_width * 2, AV_PIX_FMT_RGB24, rgb, eye, eye, 3 * eye);
	THProcessImages(net, &rgb, 1, eye, eye, 3*eye, &result, &outwidth, &outheight, 0);
	return result;
}

void run_motion()
{
	unsigned char *y[2], *diffs[4];
	int cury = 0;
	int rw = frame_width / motion_downscale, rh = frame_height / motion_downscale;
	int i, j, n;
	RECT rects[10], *r;
	float *result;
	int personidx = -1;
	unsigned char *rgb = malloc(eye * eye * 3);
	const unsigned colors[] = {
		0xffffff, 0x0000ff, 0x00ff00, 0x008000, 0xff8000, 0x00ffff,
		0x008080, 0xff00ff, 0x800080, 0x8000ff, 0x808000,
		0x808080, 0x404040, 0xff0000, 0x800000, 0xffff00, 0x808000};
		
	
	y[0] = malloc(rw * rh);
	y[1] = malloc(rw * rh);
	for(i = 0; i < 4; i++)
		diffs[i] = malloc(rw * rh);
	// diffs[0] and diffs[1] contain the last two binary images
	// diffs[2] contains the sum of diffs[0] and diffs[1] and then the eroded
	// diffs[3] contains the dilated sum
	for(i = 0; i < ncat; i++)
		if(!strcmp(categories[i], "person"))
		{
			personidx = i;
			break;
		}
	if(personidx == -1)
		THError("Fatal: this network does not contain a category called person");
	for(;;)
	{
		rescale((unsigned char *)frames[curframe], frame_width, frame_height, frame_width * 2, AV_PIX_FMT_GRAY8, y[cury], rw, rh, rw);
		for(i = 0; i < rw*rh; i++)
		{
			diffs[cury][i] = abs(y[cury][i] - y[1-cury][i]) > motion_threshold;
			diffs[2][i] = diffs[0][i] + diffs[1][i];
		}
		dilate(diffs[3], diffs[2], rw, rh, 9);
		erode(diffs[2], diffs[3], rw, rh, 5);
		n = connectedComponent(diffs[2], (int *)rects, sizeof(rects)/sizeof(rects[0]), rh, rw);
		
		for(i = 0; i < n; i++)
		{
			// Bring the rectangle to actual frame view
			r = rects + i;
			r->w = (r->w - r->x + 3) * motion_downscale;
			r->h = (r->h - r->y + 3) * motion_downscale;
			r->x = (r->x - 1) * motion_downscale;
			r->y = (r->y - 1) * motion_downscale;
			if(r->w < motion_minsize || r->h < motion_minsize)
				continue;
			result = process(*r, rgb);
			int b2o = -1;
			for(j = 0; j < nobjects; j++)
			{
				RECT *r2 = &objects[j].r;
				int x1 = MAX(r->x, r2->x);
				int x2 = MIN(r->x + r->w, r2->x + r2->w);
				int y1 = MAX(r->y, r2->y);
				int y2 = MIN(r->y + r->h, r2->y + r2->h);
				int minarea = MIN(r->w * r->h, r2->w * r2->h);
				float overlap = abs(y2-y1) * abs(x2-x1) / (float)minarea;
				if(overlap > 0.5 && y2 - y1 > 0 && x2 - x1 > 0)
				{
					if(b2o >= 0)
					{
						memmove(objects + b2o, objects + b2o + 1, (nobjects - b2o - 1) * sizeof(*objects));
						nobjects--;
						j--;
					} else {
						b2o = j;
						// If the new bounding box area suddenly decreases
						// the object is probably becoming still, as only some
						// parts are still in motion
						if(r2->w * r2->h / (float)(r->w * r->h) > 2)
							r = r2;
					}
				}
			}
			char target = result[personidx] >= pdt;
			struct object *o = b2o >= 0 ? objects + b2o : 0;
			if(!o)
			{
				// New object
				o = objects + nobjects++;
				o->color = 0;
				o->id = oids++;
				o->r = *r;
				o->valid = 1;
				o->decay = decaylimit;
				o->still = 0;
				o->permastill = 0;
				o->target = target;
			} else {
				if(o->still)
					o->valid = 0;
				else {
					o->valid = 1;
					o->r = *r;
				}
				o->decay = decaylimit;
				if(target && !o->target)
				{
					o->id = oids++;
					o->target = 1;
				}
			}
		}
		for(j = 0; j < nobjects; j++)
		{
			struct object *o = objects + j;
			if(o->valid)
				o->valid = 0;
			else {
				RECT *r = &o->r;
				result = process(*r, rgb);
				char target = result[personidx] >= pdt;
				if(target)
					o->still = o->permastill > minstillframes;
				else if(!o->decay)
				{
					memmove(objects + j, objects + j + 1, (nobjects - j - 1) * sizeof(*objects));
					nobjects--;
					j--;
				} else o->decay--;
				o->permastill++;
			}
		}
		nitems = 0;
		for(j = 0; j < nobjects; j++)
		{
			struct object *o = objects + j;
			RECT *r = &o->r;
			if(o->target)
			{
				if(!o->color)
					o->color = colors[rand() * sizeof(colors) / sizeof(colors[0]) / RAND_MAX];
				rectangle(r->x, r->y, r->w, r->h, 8, o->color);
				text(r->x + 8, r->y + 8, "person", 16, o->color);
			} else rectangle(r->x, r->y, r->w, r->h, 2, 0x808080);
		}
		cury = 1-cury;
	}
}

void run_simple()
{
	unsigned char *rgb = malloc(frame_width * frame_height * 3);
	int i, offset = (frame_width - frame_height) / 2 * 2;
	struct catp *res = malloc(sizeof(*res) * ncat);
	
	for(;;)
	{
		float *result;
		double t;
		int outwidth, outheight, n;
		char s[300];

		t = seconds();
		rescale((unsigned char *)frames[curframe] + offset, frame_height, frame_height, frame_width * 2, AV_PIX_FMT_RGB24, rgb, eye, eye, 3 * eye);
		n = THProcessImages(net, &rgb, 1, eye, eye, 3*eye, &result, &outwidth, &outheight, 0);
		if(n / outwidth != ncat)
			THError("Bug: wrong number of outputs received: %d != %d", n / outwidth, ncat);
		if(outheight != 1)
			THError("Bug: outheight expected 1");
		t = 1.0 / (seconds() - t);
		for(i = 0; i < ncat; i++)
		{
			res[i].p = result[i];
			res[i].cat = categories[i];
		}
		qsort(res, ncat, sizeof(*res), catpcmp);
		nitems = 0;
		sprintf(s, "%.2f fps", t);
		text(10, 10, s, 16, 0xff0000);
		for(i = 0; i < 5; i++)
		{
			text(10, 40 + i * 20, res[i].cat, 16, 0x0000a0);
			sprintf(s, "(%.0f %%)", res[i].p * 100);
			text(100, 40 + i * 20, s, 16, 0x0000a0);
		}
	}
}

int main(int argc, char **argv)
{
	int rc, alg = 2, i, fullscreen = 0, motion = 0;
	char camera[6] = "cam0";
	const char *modelsdir = 0, *inputfile = camera;

	frame_width = 640;
	frame_height = 480;
	for(i = 1; i < argc; i++)
	{
		if(argv[i][0] != '-')
			continue;
		switch(argv[i][1])
		{
		case 'm':
			if(i+1 < argc)
				modelsdir = argv[++i];
			break;
		case 'i':
			if(i+1 < argc)
				inputfile = argv[++i];
			break;
		case 'a':
			if(i+1 < argc)
				alg = atoi(argv[++i]);
			break;
		case 'f':
			fullscreen = 1;
			break;
		case 'M':
			motion = 1;
			break;
		case 'r':
			if(i+1 < argc)
			{
				i++;
				if(!strcasecmp(argv[i], "QVGA"))
				{
					frame_width = 320;
					frame_height = 240;
				} else if(!strcasecmp(argv[i], "HD"))
				{
					frame_width = 1280;
					frame_height = 720;
				} else if(!strcasecmp(argv[i], "FHD"))
				{
					frame_width = 1920;
					frame_height = 1080;
				}
			}
			break;
		}
	}
	if(!modelsdir || !inputfile)
	{
		fprintf(stderr, "Syntax: thnetsdemo -m <models directory> [-i <input file (default cam0)>]\n");
		fprintf(stderr, "                   [-a <alg=0:norm,1:MM,2:virtMM (default),3:cuDNN,4:cudNNhalf>]\n");
		fprintf(stderr, "                   [-r <QVGA,VGA (default),HD,FHD] [-f(ullscreen)]\n");
		fprintf(stderr, "                   [-M(otion mode)\n");
		return -1;
	}
	if(alg == 4)
	{
		alg = 3;
		THCudaHalfFloat(1);
	}
	loadfont();
	THInit();
	net = THLoadNetwork(modelsdir);
	loadcategories(modelsdir);
	if(net)
	{
		THMakeSpatial(net);
		if(alg == 0)
			THUseSpatialConvolutionMM(net, 0);
		else if(alg == 1 || alg == 2)
			THUseSpatialConvolutionMM(net, alg);
		else if(alg == 3)
		{
			THNETWORK *net2 = THCreateCudaNetwork(net);
			if(!net2)
				THError("CUDA not compiled in");
			THFreeNetwork(net);
			net = net2;
		}
		char videodev[20] = "/dev/video";
		if(!memcmp(inputfile, "cam", 3))
			strcpy(videodev+10, inputfile+3);
		else THError("Only cams supported for now");
		vcap = videocap_open(videodev);
		if(!vcap)
			THError("Error opening camera");
		rc = videocap_startcapture(vcap, frame_width, frame_height, V4L2_PIX_FMT_YUYV, 10, NBUFFERS);
		if(rc)
			THError("startcapture error %d", rc);
		win_width = frame_width;
		win_height = frame_height;
		if(fullscreen)
			win_width = win_height = 0;
		rc = CreateWindow("thnetsdemo", 0, 0, win_width, win_height);
		if(rc)
			THError("CreateWindow failed");
		GetWindowSize(&win_width, &win_height);
		pthread_t tid;
		pthread_create(&tid, 0, rendering_thread, 0);
		pthread_detach(tid);
		while(!frames[curframe])
			usleep(10000);
		if(motion)
			run_motion();
		else run_simple();
	} else printf("The network could not be loaded: %d\n", THLastError());
	return 0;
}
