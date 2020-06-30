#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

int main()
{
   int buffer[36] = { 5, 3, 1, 0, 0, 1, 0, 1, 2, 3, 1, 2, 1, 3, 1, 3, 3, 4, 2, 1, 1, 6, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0};

   FILE *fp = fopen("5533convolution.bin", "wb");

   fwrite(buffer, sizeof(buffer), 1, fp);

   //printf("%s\n", buffer[0]);

   fclose(fp);

   return 0;
}