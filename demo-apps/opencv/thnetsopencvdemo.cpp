#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <opencv2/opencv.hpp>
extern "C" {
#include "thnets.h"
}

using namespace cv;

#define NBUFFERS 4
#define MAX_OBJECTS 20
static int frame_width, frame_height, win_width, win_height;
static FT_Library ft_lib;
static FT_Face ft_face;
static THNETWORK *net;
const char *winname = "thnets opencv demo";
const int eye = 231;
const int motion_threshold = 20;
const int motion_downscale = 8;
const int motion_minsize = 64;
const int decaylimit = 3;
const int minstillframes = 3;
const float pdt = 0.5;

typedef struct {
	int x, y, w, h;
} RECT;

static double seconds()
{
	static double base;
	struct timeval tv;

	gettimeofday(&tv, 0);
	if(!base)
		base = tv.tv_sec + tv.tv_usec * 1e-6;
	return tv.tv_sec + tv.tv_usec * 1e-6 - base;
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
	categories = (char **)calloc(ncat, sizeof(*categories));
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

static int text(Mat *frame, int x, int y, const char *text, int size, int color)
{
	if(!ft_face)
		return -1;

	int i, stride, asc;
	unsigned j, k;
	unsigned char red = (color >> 16);
	unsigned char green = (color >> 8);
	unsigned char blue = color;
	FT_Set_Char_Size(ft_face, 0, size * 64, 0, 0 );
	int scale = ft_face->size->metrics.y_scale;
	asc = FT_MulFix(ft_face->ascender, scale) / 64;
	stride = frame->step1(0);
	FT_Set_Char_Size(ft_face, 0, 64 * size, 0, 0);
	unsigned char *pbitmap = frame->data + stride * y + x * 3, *p;
	int str_len = strlen(text);
	//printf("%s .... str_len = %d\n" ,text,str_len);

	for(i = 0; i < str_len; i++)
	{
		FT_Load_Char(ft_face, text[i], FT_LOAD_RENDER);
		FT_Bitmap *bmp = &ft_face->glyph->bitmap;

		int left = ft_face->glyph->bitmap_left;
		int top = ft_face->glyph->bitmap_top;

		left = left < 0 ? 0 : left;
		
		for(j = 0; j < bmp->rows; j++)
			for(k = 0; k < bmp->width; k++)
			{
				p = pbitmap + (j + asc - top) * stride + 3 * (k + left);
				p[0] = (bmp->buffer[j * bmp->pitch + k] * red + (255 - bmp->buffer[j * bmp->pitch + k]) * p[0]) / 255;
				p[1] = (bmp->buffer[j * bmp->pitch + k] * green + (255 - bmp->buffer[j * bmp->pitch + k]) * p[1]) / 255;
				p[2] = (bmp->buffer[j * bmp->pitch + k] * blue + (255 - bmp->buffer[j * bmp->pitch + k]) * p[2]) / 255;
			}
		pbitmap += 3 * (ft_face->glyph->advance.x / 64);
	}
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

extern "C" int connectedComponent(unsigned char *image, int* coordinates, int coordinatesSize, int height, int width);

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

float *process(RECT r, Mat frame, unsigned char *rgb)
{
	float *result;
	int outwidth, outheight;

	expandrect(&r, motion_minsize, frame_width, frame_height);
	Rect roi(r.x, r.y, r.w, r.h);
	Mat cropped = frame(roi);
	Mat resized;
	resize(frame, resized, Size(eye, eye));
	THProcessImages(net, &resized.data, 1, eye, eye, 3*eye, &result, &outwidth, &outheight, 1);
	return result;
}

void run_motion(VideoCapture &cap)
{
	Mat frame, y[2];
	unsigned char *diffs[4];
	int cury = 0;
	int rw = frame_width / motion_downscale, rh = frame_height / motion_downscale;
	int i, j, n;
	RECT rects[10], *r;
	float *result;
	int personidx = -1;
	unsigned char *rgb = (unsigned char *)malloc(eye * eye * 3);
	const unsigned colors[] = {
		0xffffff, 0x0000ff, 0x00ff00, 0x008000, 0xff8000, 0x00ffff,
		0x008080, 0xff00ff, 0x800080, 0x8000ff, 0x808000,
		0x808080, 0x404040, 0xff0000, 0x800000, 0xffff00, 0x808000};
		
	for(i = 0; i < 4; i++)
		diffs[i] = (unsigned char *)malloc(rw * rh);
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
		cap >> frame;
		Mat resized;
		resize(frame, resized, Size(rw, rh));
		cvtColor(resized, y[cury], CV_BGR2GRAY);
		if(!y[1-cury].data)
		{
			cury = 1-cury;
			continue;
		}
		for(i = 0; i < rw*rh; i++)
		{
			diffs[cury][i] = abs(y[cury].data[i] - y[1-cury].data[i]) > motion_threshold;
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
			result = process(*r, frame, rgb);

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
				result = process(*r, frame, rgb);
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
		for(j = 0; j < nobjects; j++)
		{
			struct object *o = objects + j;
			RECT *r = &o->r;
			Rect r1(r->x, r->y, r->w, r->h);
			if(o->target)
			{
				if(!o->color)
					o->color = colors[rand() * sizeof(colors) / sizeof(colors[0]) / RAND_MAX];
				Scalar col((o->color >> 16) &  0xff, (o->color >> 8) & 0xff, o->color & 0xff);
				rectangle(frame, r1, col, 8);
				text(&frame, r->x + 8, r->y + 8, "person", 16, o->color);
			} else {
				Scalar col(0x80, 0x80, 0x80);
				rectangle(frame, r1, col, 1);
			}
		}
		imshow(winname, frame);
		waitKey(1);
		cury = 1-cury;
	}
}

void run_simple(VideoCapture &cap)
{
	int offset = (frame_width - frame_height) / 2;
	struct catp *res = (struct catp *)malloc(sizeof(*res) * ncat);
	int i;
	
	Mat frame;
	float fps = 1;
	for(;;)
	{
		float *result;
		double t;
		int outwidth, outheight, n;
		char s[300];

		t = seconds();
		cap >> frame;
		//frame = imread("/home/baughg/test/opencv_img/image1.jpg", CV_LOAD_IMAGE_COLOR);
		Rect roi(offset, 0, frame_height, frame_height);
		Mat cropped = frame(roi);
		Mat resized;
		resize(frame, resized, Size(eye, eye));
		n = THProcessImages(net, &resized.data, 1, eye, eye, 3*eye, &result, &outwidth, &outheight, 1);
		if(n / outwidth != ncat)
			THError("Bug: wrong number of outputs received: %d != %d", n / outwidth, ncat);
		if(outheight != 1)
			THError("Bug: outheight expected 1");
		for(i = 0; i < ncat; i++)
		{
			res[i].p = result[i];
			res[i].cat = categories[i];
		}
		qsort(res, ncat, sizeof(*res), catpcmp);
		sprintf(s, "%.2f fps", fps);
		text(&cropped, 10, 10, s, 16, 0x0000ff);
		for(i = 0; i < 5; i++)
		{
			text(&cropped, 10, 40 + i * 20, res[i].cat, 16, 0xa00000);
			sprintf(s, "(%.0f %%)", res[i].p * 100);
			text(&cropped, 100, 40 + i * 20, s, 16, 0xa00000);
		}
		imshow(winname, cropped);
		waitKey(1);
		fps = 1.0 / (seconds() - t);
	}
}

int main(int argc, char **argv)
{
	int alg = 2, i, motion = 0;
	char camera[6] = "cam0";
	const char *modelsdir = 0, *inputfile = camera;

	frame_width = 640;
	frame_height = 480;
	loadfont();
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
		if(memcmp(inputfile, "cam", 3))
			THError("Only cams supported for now");
		VideoCapture cap(atoi(inputfile+3));
		if(!cap.isOpened())
			THError("Cannot open video device");
		cap.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
		win_width = frame_width;
		win_height = frame_height;
		if(motion)
			run_motion(cap);
		else run_simple(cap);
	} else printf("The network could not be loaded: %d\n", THLastError());
	return 0;
}
