#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void print_image(int* image, int height, int width) // image를 보기 위한 함수
{
   for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
         printf("%3d ", image[i*width + j]);
      }
      printf("\n");
   }
}

void convolution(int* output, int* image, int* kernel, int image_Height, int image_Width, int kernel_Height, int kernel_Width, int pad_Top, int pad_Bottom, int pad_Left, int pad_Right, int stride_Height, int stride_Width)
{
   int H = image_Height;
   int W = image_Width;
   int kH = kernel_Height;
   int kW = kernel_Width;
   int pT = pad_Top;
   int pB = pad_Bottom;
   int pL = pad_Left;
   int pR = pad_Right;
   int sH = stride_Height;
   int sW = stride_Width;
   int oH, oW; // output_Height, output_Width

   oH = (H + pT + pB - kH) / sH + 1;
   oW = (W + pL + pR - kW) / sW + 1;

   //pad image memory allocation
   int* pad_image;
   
   int pad_height = H + pT + pB ;
   int pad_width = W + pL + pR ;
   
   pad_image = (int*)malloc((pad_height * pad_width) * sizeof(int));
   memset(pad_image, 0, (pad_height * pad_width) * sizeof(int));

   //fill pad image 
   for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
         int pad_index = i * (W + pL + pR) + j + pL + pT * ( W + pL + pR);
         pad_image[pad_index] = image[i * W + j];
      }
   }

   print_image(pad_image, pad_height, pad_width);
   
   //convolution
   for (int i = 0; i < oH; i++) {
      for (int j = 0; j < oW; j++) { 
         int sum = 0;
         for (int h = 0; h < kH; h++) {
            for (int w = 0; w < kW; w++) {
               int kernel_index = h * kW + w;
               int sum_index = i * ( W + pL + pR) * sH + j * sW + h * ( W + pL + pR) + w;
               sum += pad_image[sum_index] * kernel[kernel_index];
            }
         }
         int output_index = i * oW  + j;
         output[output_index] = sum;
      }
   }
}

int main()
{
   int imagesize, kernelsize;
   int image[25];
   int kernel[9];

   FILE *fp = fopen("5533convolution.bin", "r");

   //int imagesize, kernelsize;
   //int image[9];
   //int kernel[4];

   //FILE *fp = fopen("3322convolution.bin", "r");

   fread(&imagesize, sizeof(imagesize), 1, fp);
   fread(&kernelsize, sizeof(kernelsize), 1, fp);
   fread(&image, sizeof(image), 1, fp);
   fread(&kernel, sizeof(kernel), 1, fp);

   fclose(fp);

   printf("imagesize: %d\n", imagesize);
   printf("============================\n");

   printf("kernelsize: %d\n", kernelsize);
   printf("============================\n");

   print_image(image, imagesize, imagesize);
   printf("============================\n");

   print_image(kernel, kernelsize, kernelsize);
   printf("============================\n");

   int image_height = imagesize;
   int image_width = imagesize;
   int kernel_height = kernelsize;
   int kernel_width = kernelsize;
   int pad_top = 1;
   int pad_bottom = 1;
   int pad_left = 1;
   int pad_right = 1;
   int stride_height = 1;
   int stride_width = 1;

   int output_height = ( (image_height + pad_top + pad_bottom - kernel_height) / stride_height ) + 1;
   int output_width = ( (image_width + pad_left + pad_right - kernel_width) / stride_width) + 1 ;

   printf("output_height : %d, output_width : %d\n", output_height, output_width);
   printf("============================\n");

   int *output;
   output = (int*)malloc((output_height *  output_width) * sizeof(int));
   memset(output, 0, (output_height  * output_width) * sizeof(int));

   convolution(output, image, kernel, image_height, image_width, kernel_height, kernel_width, pad_top, pad_bottom, pad_left, pad_right, stride_height, stride_width);

   printf("============================\n");

   print_image(output, output_height, output_width);

   free(output);

   return 0;
}