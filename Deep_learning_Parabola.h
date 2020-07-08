#ifndef DEEP_LEARNING_H
#define DEEP_LEARNING_H

#define _MAXNODES_ 201
#define _MAXSAMPLES_ 1001
#define _MAXSAMPLESIN_ 6
#define _MAXNETWORK_ 4

typedef struct
{
	double RMS;
	double Aer;
	double weight[_MAXNODES_][_MAXNODES_][_MAXNETWORK_];
	double deltaweight[_MAXNODES_][_MAXNODES_][_MAXNETWORK_];
	 // double bias[_MAXNODES_][_MAXNETWORK_];
	double a[_MAXNODES_][_MAXNETWORK_];   // activation
	double z[_MAXNODES_][_MAXNETWORK_];   // activation inner factor
	double d[_MAXNODES_][_MAXNODES_][_MAXNETWORK_];   // gradient
	double temp[_MAXNODES_][_MAXNETWORK_];   // for temporary computation
}Neurons;

typedef struct
{
	int nneuronsin;
	int nneuronsout;
	int nneuronshid;
	int samplesperepoch;
	int nlayers;
	int maxepoch;
	int nneuronslayers[_MAXNETWORK_];
	double eta;   // learning rate
	double bias;   // bias
	double tol;   // bias
	double momentum;
}NeuronData;

int init(NeuronData *neurondat, Neurons *neurons);
// int HandData(Neurons *neurons, double (*Data)[5], NeuronData *neurondat);
int feedforward(Neurons *neurons, NeuronData *neurondat);
int backpropagate(Neurons *neurons, NeuronData *neurondat, double *DataEpoch);
int UpdateWandB(Neurons *neurons, NeuronData *neurondat);
int PlotNetwork(Neurons *neurons, NeuronData *neurondat);
double sigmoid(double z);
double sigmoid_prime(double z);


#endif
