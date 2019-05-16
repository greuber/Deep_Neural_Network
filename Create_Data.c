// Create the data

#include <stdio.h>
#include <stdlib.h>
#include <Create_Data.h>
#include <math.h>

// gcc -I/home/greuber/Programmieren/Deep_Learning Create_Data.c -lm

int main(void)
{
  int n=2001;
  int m = 2;
  int i,j;
  double Data[n][m];

  FILE *f = fopen("Data.txt", "w");
  if (f == NULL)
  {
      printf("Error opening file!\n");
      exit(1);
  }

  // Configure rand
  time_t t;
	srand((unsigned) time(&t));

  for(i=0;i<n;i++)
  {
    for(j=0;j<m;j++)
    {
      Data[i][j] = round((double)rand() / (double)RAND_MAX) ;
    }
    // Data[i][4] = Fun(Data[i][0],Data[i][1],Data[i][2],Data[i][3]);
    Data[i][2] = (Data[i][0] == Data[i][1]);
    fprintf(f, "%f %f %f\n", Data[i][0],Data[i][1],Data[i][2]);
  }
  fclose(f);
}


/*double Fun(double x1,double x2,double x3, double x4)
{
  double y;
  y = (x1*x1) + 3*x2 + 4*x3 + 2*x4;
  return y;
}*/

/*double Fun(double x1)
{
  double y;
  y = (x1*x1);
  return y;
}*/
