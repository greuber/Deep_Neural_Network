// Create the data

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <Create_Data.h>
#include <math.h>

// g++ -I/home/greuber/Programmieren/Deep_Learning/Deep_Learning_Parabolar_2 Create_Data.c

int main(void)
{
  int epochs = 1e3;
  int ijk;
  double Data[epochs][6];
  double n;
  double x[5] = { 0, 0.25, 0.5, 0.75, 1 };

  FILE *f = fopen("Data.txt", "w");
  if (f == NULL)
  {
      printf("Error opening file!\n");
      exit(1);
  }

  // Configure rand
  srand(time(NULL));

  for (ijk = 0;ijk<epochs;ijk++)
  {
	// n = (rand()%6)+1;
	n = (rand()) / (double)RAND_MAX;

	Data[ijk][0] = pow(x[0],n) ;
	Data[ijk][1] = pow(x[1],n) ;
	Data[ijk][2] = pow(x[2],n) ;
	Data[ijk][3] = pow(x[3],n) ;
	Data[ijk][4] = pow(x[4],n) ;
	Data[ijk][5] = n;

	fprintf(f, "%f %f %f %f %f %f\n", Data[ijk][0],Data[ijk][1],Data[ijk][2],Data[ijk][3],Data[ijk][4],Data[ijk][5]);
	// printf("%f %f %f %f %f %f\n\n", Data[ijk][0],Data[ijk][1],Data[ijk][2],Data[ijk][3],Data[ijk][4],Data[ijk][5]);
  }

  fclose(f);
}


