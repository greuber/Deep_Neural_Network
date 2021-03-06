#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Deep_learning.h>

// gcc -I/home/greuber/Programmieren/Deep_Learning Deep_learning.c -lm
// neuron[ineuron][ilayer]
// w[thislayer][previouslayer][ilayer]
// jin = layer that comes as input (e.g. l-1)
// jout = layer 'output' (e.g. l)

// todo: Bias neuron is currently not taken into account

int main(void)
{
	int iepoch, ilayer, ineuron, error, neuron, it, npx, t, i, j ,n, m, nData, output;
	Neurons neurons;
	NeuronData neurondat;
	double C[_MAXNODES_];
	double delta;
	double Data[_MAXSAMPLES_][_MAXSAMPLESIN_];
	double DataEpoch[_MAXSAMPLES_];

	output = 1000;  // output every epochs % output = 0
	neurondat.eta = 0.5;   // learning rate
	neurondat.bias = 1;
	neurondat.tol = 1e-3;
	neurondat.maxepoch = 1e4;
	neurondat.momentum = 0.0;   // learning rate
	neurondat.nlayers = 3;
	neurondat.nneuronshid = 2;
	neurondat.nneuronsin = 2;
	neurondat.nneuronsout = 2;
	for(i=0;i<neurondat.nlayers;i++)
	{
		if(i == 0)                   neurondat.nneuronslayers[i] = neurondat.nneuronsin;
		else if(i == neurondat.nlayers-1) neurondat.nneuronslayers[i] = neurondat.nneuronsout;
		else                         neurondat.nneuronslayers[i] = neurondat.nneuronshid;
	}

	/*// Load Data
	FILE *f = fopen("Data.txt", "r");
	if (f == NULL)
	{
			printf("Error opening file!\n");
			return 1;
	}
	for(i=0;i<n;i++)
	{
		// fscanf(f, "%lf %lf %lf %lf %lf\n", &Data[i][0],&Data[i][1],&Data[i][2],&Data[i][3],&Data[i][4]);
		fscanf(f, "%lf %lf %lf\n", &Data[i][0],&Data[i][1],&Data[i][2]);
	}
	fclose(f);*/

	// Create Data
	for(i=0;i<neurondat.maxepoch;i++)
	{
		for(j=0;j<neurondat.nneuronsin;j++)
		{
			Data[i][j] = round((double)rand() / (double)RAND_MAX) ;
			/*// Test data set
			Data[i][0] = 0.05;
			Data[i][1] = 0.1;
			Data[i][2] = 0.01;
			Data[i][3] = 0.99;*/
		}
		Data[i][2] = (double)(Data[i][0] == Data[i][1]);
	}

	// Initialize network
	error = init(&neurondat, &neurons);

	// Train the actual network with GD
	iepoch = 0;
	neurons.RMS = 1e100;

	// Create Data
	while(neurons.RMS > neurondat.tol || iepoch<neurondat.maxepoch)
	{
		// Assign data to 1D array
	  for(ineuron=0;ineuron<neurondat.nneuronsin;ineuron++) {
			DataEpoch[ineuron] = Data[iepoch][ineuron+neurondat.nneuronsin];  // output has to be stored
			neurons.a[ineuron][0] = Data[iepoch][ineuron];  // input
		}

		// printf("feedforward ...\n");
		error = feedforward(&neurons, &neurondat);

		// Compute first (RMS) error of cost function
		neurons.RMS = 0;
		for(ineuron=0;ineuron<neurondat.nneuronsout;ineuron++)
		{
			delta = (DataEpoch[ineuron] - neurons.a[ineuron][neurondat.nlayers-1]);
			neurons.RMS += 0.5*(delta*delta) ;
		}

		// printf("backpropagate ...\n");
	  error = backpropagate(&neurons, &neurondat, DataEpoch);

		// printf("update w and b ...\n");
		error = UpdateWandB(&neurons, &neurondat);

		if((iepoch+1) % output == 0 || iepoch == 0)
		{
			printf("%i. Error = %.20f (1. Val example: Input = %.2f ; NN = %.5f, True = %.2f)\n",(iepoch+1),neurons.RMS,Data[iepoch][0],neurons.a[0][neurondat.nlayers-1],DataEpoch[0]);
			// error = PlotNetwork(&neurons, &neurondat);
		}
		iepoch+=1;
	}

	return 0;


}
// ---------------------------------------------------------------//
int init(NeuronData *neurondat, Neurons *neurons)
{
	// ATTENTION: still a bit weird because the weights are here initiated a bit counterintuitive
	// compared to the rest of the alg since jin and jout are flipped ..

	int i, jin, jout;
	time_t t;

	srand((unsigned) time(&t));

	for(i=0;i<neurondat->nlayers;i++)
	{
		if(i == 0)
		{
			for(jout=0;jout<neurondat->nneuronslayers[i];jout++)
			{
				neurons->weight[jout][0][i] = (double)rand() / (double)RAND_MAX ;
			}
		}
		else if(i == neurondat->nlayers-1)
		{
			for(jin=0;jin<neurondat->nneuronslayers[i-1];jin++)
			{
				neurons->weight[0][jin][i] = (double)rand() / (double)RAND_MAX ;
			}
		}
		else
		{
			for(jout=0;jout<neurondat->nneuronslayers[i];jout++)
			{
				for(jin=0;jin<neurondat->nneuronslayers[i-1];jin++)
				{
					neurons->weight[jout][jin][i] = (double)rand() / (double)RAND_MAX ;
				}
			}
		}
	}

	/*// Test data
	neurons->weight[0][0][0] = 0.15;
	neurons->weight[1][0][0] = 0.20;
	neurons->weight[0][1][0] = 0.25;
	neurons->weight[1][1][0] = 0.30;

	neurons->weight[0][0][1] = 0.40;
	neurons->weight[1][0][1] = 0.45;
	neurons->weight[0][1][1] = 0.50;
	neurons->weight[1][1][1] = 0.55;*/
}
// ---------------------------------------------------------------//
int feedforward(Neurons *neurons, NeuronData *neurondat)
{
	int jout, jin, i;

	for(i=1;i<neurondat->nlayers-1;i++) {
		neurons->a[neurondat->nneuronsin][i] = neurondat->bias;  // bias neuron has constant output 1
	}

	/*// Test bias
	neurons->a[neurondat->nneuronsin][0] = 0.35;  // bias neuron has constant output 1
	neurons->a[neurondat->nneuronshid][1] = 0.6;  // bias neuron has constant output 1*/

	// Hidden neurons
	for(i=1;i<neurondat->nlayers;i++)
	{
		for(jout=0;jout<neurondat->nneuronslayers[i];jout++)
		{
			neurons->z[jout][i] = 0;
			for(jin=0;jin<neurondat->nneuronslayers[i-1];jin++)
			{
				neurons->z[jout][i] += neurons->weight[jin][jout][i-1] * neurons->a[jin][i-1];
			}
			neurons->a[jout][i] = sigmoid(neurons->z[jout][i]+(neurons->a[neurondat->nneuronslayers[i-1]][i-1]*1));
		}
	}

	return 0;
}
// ---------------------------------------------------------------//
int backpropagate(Neurons *neurons, NeuronData *neurondat, double *DataEpoch)
{
	int jout, jin, jmid, i;

	// Output layer gradients
	for(jout=0;jout<neurondat->nneuronsout;jout++)
	{
		for(jin=0;jin<neurondat->nneuronshid;jin++)
		{
			neurons->d[jin][jout][neurondat->nlayers-1]           = -(DataEpoch[jout] - neurons->a[jout][neurondat->nlayers-1]) *sigmoid_prime(neurons->a[jout][neurondat->nlayers-1]) ;
			neurons->deltaweight[jin][jout][neurondat->nlayers-1] = neurons->d[jin][jout][neurondat->nlayers-1] * neurons->a[jin][neurondat->nlayers-2];
		}
	}


	// Hidden layer gradients
	for(i=neurondat->nlayers-2;i>0;i--)
	{
		for(jin=0;jin<neurondat->nneuronslayers[i-1];jin++)
		{
			for(jmid=0;jmid<neurondat->nneuronslayers[i];jmid++)
			{
				neurons->temp[jmid][i] = 0;
				for(jout=0;jout<neurondat->nneuronslayers[i+1];jout++)
				{
					neurons->temp[jmid][i] += neurons->weight[jmid][jout][i] * neurons->d[jmid][jout][i+1];
				}
				neurons->deltaweight[jin][jmid][i] = neurons->temp[jmid][i]*sigmoid_prime(neurons->a[jmid][i])*neurons->a[jin][i-1];
			}
		}
	}

	return 0;
}
// ---------------------------------------------------------------//
int UpdateWandB(Neurons *neurons, NeuronData *neurondat)
{
	int jout, jin, i;

	for(i=neurondat->nlayers-2;i>-1;i--)
	{
		for(jout=0;jout<neurondat->nneuronsout;jout++)
		{
			for(jin=0;jin<neurondat->nneuronshid;jin++)
			{
				neurons->weight[jin][jout][i] -= neurondat->eta * neurons->deltaweight[jin][jout][i+1];
			}
		}
	}

	return 0;
}
// ---------------------------------------------------------------//
int PlotNetwork(Neurons *neurons, NeuronData *neurondat)
// This is asimple plot function to visualize the network
{
	int i,j;

	// Only loop over the hidden layers which are 2 less than total and hidden layers should always have the most neurons
	//for(i=0; i<neurondat->nlayers-3;i++)
	//{
		for(j=0;j<neurondat->nneuronshid;j++)
		{
			if(neurondat->nneuronsin-1>j)
			{
				fprintf(stdout, "%.5f | %.5f | %.5f\n", neurons->a[j][0],neurons->a[j][1],neurons->a[j][2]);
			}
			else if(neurondat->nneuronsout-1<j && neurondat->nneuronsin-1>=j)
			{
				fprintf(stdout, "%.5f | %.5f |      \n", neurons->a[j][0],neurons->a[j][1]);
			}
			else
			{
				fprintf(stdout, "        | %.5f |      \n", neurons->a[j][1]);
			}
		}
	//}

	fflush(stdout);

	return 0;
}
// ---------------------------------------------------------------//
double sigmoid(double z)
{
	return (1.0/(1.0+exp(-z)));
}
// ---------------------------------------------------------------//
double sigmoid_prime(double z)
{
	return z*(1-z);
}
