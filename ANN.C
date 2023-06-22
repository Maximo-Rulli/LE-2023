#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// We define some constants of mi algorithm
#define m 10       // Number of training examples
#define n_x 2      // Number of features
#define alpha 0.001 // Learning rate
#define RATE_CHANGE 0.00001
#define EPOCHS 1000001 // Training epochs

// Definimos la estructura de la neurona
typedef struct {
  // Parameters
  float **W;
  float *b;

  // Derivatives
  float **dJdW;
  float *dJdb;
  float **dJdA;
  float **dJdZ;

  // Cache and activation
  char activation[10];
  float **Z;
  float **A;

  // Dimensions
  unsigned int prev_neurons;
  unsigned int neurons;
} Layer;

// We define the functions' prototipes
float sigmoid(float x);
float linear(float x);
float relu(float x);

void layer_setup(Layer *layer, char *activation, unsigned int previous_neurons,
                 unsigned int layer_neurons);

void input_prop(Layer *layer, unsigned int neurons,
                unsigned int previous_neurons,
                float **input);
void forw_prop(Layer *layer, float **a);

void back_pred(Layer *layer, Layer *prev_layer, float total_cost, float lr,
               const float *y_true);
void back_prop(Layer *layer, Layer *next_layer,
               float **A_prev, float lr);

float sqr(float x);
float compute_cost(float **y_pred, const float *y_true);

// Main
int main() {
  // Inicialize the seed
  srand(0);

  // Create a dataset
  float X_data[n_x][m] = {
        {0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0},
        {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}};

  float** X = (float**)malloc(n_x * sizeof(float*));
    for (int i = 0; i < n_x; i++) {
        X[i] = (float*)malloc(m * sizeof(float));
    }

  for(unsigned int n = 0; n<n_x; n++){
    for(unsigned int i = 0; i<m; i++){
      X[n][i] = X_data[n][i];
    }
  }
  

  const float y[m] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0};

  // Create neurons
  Layer layer1;
  Layer layer2;
  Layer layer3;
  Layer layer4;

  // The function to create a layer requires the user to send the previous
  // layer's n. of neurons
  layer_setup(&layer1, "relu", n_x, 3);
  layer_setup(&layer2, "relu", layer1.neurons, 4);
  layer_setup(&layer3, "relu", layer2.neurons, 2);
  layer_setup(&layer4, "linear", layer3.neurons, 1);

  printf("Capas inicializadas! \n");

  // Propagate forward
  input_prop(&layer1, layer1.neurons, layer1.prev_neurons, X);
  forw_prop(&layer2, layer1.A);
  forw_prop(&layer3, layer2.A);
  forw_prop(&layer4, layer3.A);

  float total_cost = compute_cost(layer4.A, y);
  printf("Costo inicial: %f \n", total_cost);

  // Train the model
  for (unsigned int i = 0; i < EPOCHS; i++) {
    // Backpropagate and update weights
    back_pred(&layer4, &layer3, total_cost, alpha, y);
    back_prop(&layer3, &layer4, layer2.A, alpha);
    back_prop(&layer2, &layer3, layer1.A, alpha);
    back_prop(&layer1, &layer2, X, alpha);

    // Forward porpagate and update cost
    input_prop(&layer1, layer1.neurons, layer1.prev_neurons, X);
    forw_prop(&layer2, layer1.A);
    forw_prop(&layer3, layer2.A);
    forw_prop(&layer4, layer3.A);

    total_cost = compute_cost(layer4.A, y);

    if (i % 100000 == 0)
      printf("Costo %i: %f\n", i, total_cost);
  }

  // //Compute cost
  // printf("Predicción: %f\n", neurona1.A[0]);
  // printf("Pesos %f\n", neurona1.w);
  // printf("Bias %f\n", neurona1.b);
  // printf("Costo: %f\n", total_cost);

  return 0;
}

// We define the square of a number
float sqr(float x) { return pow(x, 2); }

// We define the relu function
float relu(float x) {
  if (x < 0.0)
    return 0.0;
  else {
    return x;
  }
}

// We define the sigmoid function
float sigmoid(float x) { return 1 / (1 + exp(-x)); }

// We define the linear function
float linear(float x) { return x; }

// Function to inizialice a layer
void layer_setup(Layer *layer, char *activation, unsigned int previous_neurons,
                 unsigned int layer_neurons) {
  layer->prev_neurons = previous_neurons;
  layer->neurons = layer_neurons;

  // Allocate memory for the arrays
  layer->W = (float **)malloc(layer_neurons * sizeof(float *));
  layer->dJdW = (float **)malloc(layer_neurons * sizeof(float *));
  layer->dJdA = (float **)malloc(layer_neurons * sizeof(float *));
  layer->dJdZ = (float **)malloc(layer_neurons * sizeof(float *));
  layer->b = (float *)malloc(layer_neurons * sizeof(float));
  layer->dJdb = (float *)malloc(layer_neurons * sizeof(float));
  layer->A = (float **)malloc(layer_neurons * sizeof(float *));
  layer->Z = (float **)malloc(layer_neurons * sizeof(float *));

  for (unsigned int n = 0; n < layer_neurons; n++) {
    layer->W[n] = (float *)malloc(previous_neurons * sizeof(float));
    layer->dJdW[n] = (float *)malloc(previous_neurons * sizeof(float));
    layer->A[n] = (float *)malloc(m * sizeof(float));
    layer->Z[n] = (float *)malloc(m * sizeof(float));
    layer->dJdA[n] = (float *)malloc(m * sizeof(float));
    layer->dJdZ[n] = (float *)malloc(m * sizeof(float));

    for (unsigned int i = 0; i < m; i++) {
      layer->A[n][i] = 0.0;
      layer->Z[n][i] = 0.0;
      layer->dJdA[n][i] = 0.0;
      layer->dJdZ[n][i] = 0.0;
    }
  }

  for (unsigned int n = 0; n < layer_neurons; n++) {
    layer->b[n] = 0;
    for (unsigned int j = 0; j < previous_neurons; j++) {
      layer->W[n][j] = (float)rand() / (float)(RAND_MAX / 2) * 0.01;
      layer->dJdW[n][j] = 0.0;
    }
  }

  strcpy(layer->activation, activation);
}

// Function to forward propagate the input on the first Layer
void input_prop(Layer *layer, unsigned int neurons,
                unsigned int previous_neurons,
                float **input) {
  static char sigmoid_n[] = "sigmoid";
  static char relu_n[] = "relu";
  static char linear_n[] = "linear";
  float z = 0.0;
  for (unsigned int n = 0; n < layer->neurons; n++) {
    for (char i = 0; i < m; i++) {
      z = 0.0;
      for (char nf = 0; nf < layer->prev_neurons; nf++) {
        z += layer->W[n][nf] * input[nf][i];
      }
      layer->Z[n][i] = z + layer->b[n];
      if (!strcmp(layer->activation, sigmoid_n))
        layer->A[n][i] = sigmoid(layer->Z[n][i]);

      if (!strcmp(layer->activation, relu_n))
        layer->A[n][i] = relu(layer->Z[n][i]);

      if (!strcmp(layer->activation, linear_n))
        layer->A[n][i] = linear(layer->Z[n][i]);
      //printf("Activación de neurona %d ejemplo %d: %f\n", n, i,
      //layer->A[n][i]);
    }
  }
}

// Function to forward propagate on a layer
void forw_prop(Layer *layer, float **a) {
  static char sigmoid_n[] = "sigmoid";
  static char relu_n[] = "relu";
  static char linear_n[] = "linear";

  float z = 0.0;
  for (unsigned int n = 0; n < layer->neurons; n++) {
    for (char i = 0; i < m; i++) {
      z = 0.0;
      for (char nf = 0; nf < layer->prev_neurons; nf++) {
        z += layer->W[n][nf] * a[nf][i];
      }
      layer->Z[n][i] = z + layer->b[n];
      if (!strcmp(layer->activation, sigmoid_n))
        layer->A[n][i] = sigmoid(layer->Z[n][i]);

      if (!strcmp(layer->activation, relu_n))
        layer->A[n][i] = relu(layer->Z[n][i]);

      if (!strcmp(layer->activation, linear_n))
        layer->A[n][i] = linear(layer->Z[n][i]);

      //printf("Activación de neurona %d ejemplo %d: %f\n", n, i,
      //layer->A[n][i]);
    }
  }
}

// Backpropagation computed on the output layer
void back_pred(Layer *layer, Layer *prev_layer, float total_cost, float lr,
               const float *y_true) {
  static char sigmoid_n[] = "sigmoid";
  static char relu_n[] = "relu";
  static char linear_n[] = "linear";

  // Compute derivative of A with respect to J
  // It is expected for the last layer to have only 1 neuron
  for (unsigned int i = 0; i < m; i++) {
    layer->dJdA[0][i] = (layer->A[0][i] * (layer->A[0][i] - y_true[i])) / m;

    if (!strcmp(layer->activation, sigmoid_n))
      layer->dJdZ[0][i] =
          (sigmoid(layer->Z[0][i]) * (1 - sigmoid(layer->Z[0][i]))) *
          layer->dJdA[0][i];

    else if (!strcmp(layer->activation, relu_n))
      layer->dJdZ[0][i] = (layer->dJdA[0][i]) * (layer->Z[0][i] > 0);

    else if (!strcmp(layer->activation, linear_n))
      layer->dJdZ[0][i] = layer->dJdA[0][i];
  }

  // We define W transposed (W_t)
  float **W_t;
  W_t = (float **)malloc(layer->prev_neurons * sizeof(float *));
  for (unsigned int n = 0; n < layer->prev_neurons; n++) {
    W_t[n] = (float *)malloc(layer->neurons * sizeof(float));
    for (unsigned int j = 0; j < layer->neurons; j++) {
      W_t[n][j] = layer->W[j][n];
    }
  }

  // We define A transposed (A_t) for the previous layer activations
  float **A_t;
  A_t = (float **)malloc(m * sizeof(float *));
  for (unsigned int i = 0; i < m; i++) {
    A_t[i] = (float *)malloc(layer->prev_neurons * sizeof(float));
    for (unsigned int j = 0; j < layer->prev_neurons; j++) {
      A_t[i][j] = prev_layer->A[j][i];
    }
  }

  // Compute dJdW for the final Layer
  for (unsigned int n = 0; n < layer->neurons; n++) {
    for (unsigned int j = 0; j < layer->prev_neurons; j++) {
      layer->dJdW[n][j] = 0.0;
      for (unsigned int i = 0; i < m; i++) {
        layer->dJdW[n][j] += layer->dJdZ[n][i] * A_t[i][j];
      }

      layer->dJdW[n][j] = layer->dJdW[n][j] / m;

      // Update weights
      layer->W[n][j] -= lr * layer->dJdW[n][j];

      // printf("Weights: %f \n", layer->W[n][j]);
    }
  }

  // Compute dJdA for the current layer
  for (unsigned int n = 0; n < prev_layer->neurons; n++) {
    for (unsigned int i = 0; i < m; i++) {
      prev_layer->dJdA[n][i] = 0.0;
      for (unsigned int j = 0; j < layer->neurons; j++) {
        prev_layer->dJdA[n][i] += W_t[n][j] * layer->dJdZ[j][i];
      }
      // printf("Derivatives: %f \n", prev_layer->dJdA[n][i]);
    }
  }

  // Compute dJdb for the last Layer
  for (unsigned int n = 0; n < layer->neurons; n++) {
    layer->dJdb[n] = 0.0;
    for (unsigned int i = 0; i < m; i++) {
      layer->dJdb[n] += layer->dJdZ[n][i] / m;
    }
    layer->b[n] -= lr * layer->dJdb[n];
    // printf("Bias: %f \n", layer->b[n]);
  }

  // Free memory
  for (unsigned int n = 0; n < layer->prev_neurons; n++) {
    free(W_t[n]);
  }
  free(W_t);

  for (unsigned int i = 0; i < m; i++) {
    free(A_t[i]);
  }
  free(A_t);
}

// Backpropagation computed on a hidden layer
void back_prop(Layer *layer, Layer *next_layer,
               float **A_prev, float lr) {
  static char sigmoid_n[] = "sigmoid";
  static char relu_n[] = "relu";
  static char linear_n[] = "linear";

  // We define W transposed of next layer (W_t)
  float **W_t;
  W_t = (float **)malloc(next_layer->prev_neurons * sizeof(float *));
  for (unsigned int n = 0; n < next_layer->prev_neurons; n++) {
    W_t[n] = (float *)malloc(next_layer->neurons * sizeof(float));
    for (unsigned int j = 0; j < next_layer->neurons; j++) {
      W_t[n][j] = next_layer->W[j][n];
    }
  }

  // Compute dJdA for the current layer
  for (unsigned int n = 0; n < layer->neurons; n++) {
    for (unsigned int i = 0; i < m; i++) {

      // Reset derivatives
      layer->dJdA[n][i] = 0.0;

      for (unsigned int j = 0; j < next_layer->neurons; j++) {
        layer->dJdA[n][i] += W_t[n][j] * next_layer->dJdZ[j][i];
      }
      if (!strcmp(layer->activation, sigmoid_n))
        layer->dJdZ[n][i] =
            (sigmoid(layer->Z[n][i]) * (1 - sigmoid(layer->Z[n][i]))) *
            layer->dJdA[n][i];

      else if (!strcmp(layer->activation, relu_n))
        layer->dJdZ[n][i] = (layer->dJdA[n][i]) * (layer->Z[n][i] > 0);

      else if (!strcmp(layer->activation, linear_n))
        layer->dJdZ[n][i] = layer->dJdA[n][i];
      // printf("Derivatives: %f \n", prev_layer->dJdA[n][i]);
    }
  }

  // We define A transposed (A_t) for the previous layer activations
  float **A_t;
  A_t = (float **)malloc(m * sizeof(float *));
  for (unsigned int i = 0; i < m; i++) {
    A_t[i] = (float *)malloc(layer->prev_neurons * sizeof(float));
    for (unsigned int j = 0; j < layer->prev_neurons; j++) {
      A_t[i][j] = A_prev[j][i];
    }
  }

  // Compute dJdW for the current Layer
  for (unsigned int n = 0; n < layer->neurons; n++) {
    for (unsigned int j = 0; j < layer->prev_neurons; j++) {

      // Reset values
      layer->dJdW[n][j] = 0.0;

      for (unsigned int i = 0; i < m; i++) {
        layer->dJdW[n][j] += layer->dJdZ[n][i] * A_t[i][j];
      }

      layer->dJdW[n][j] = layer->dJdW[n][j] / m;

      // Update weights
      layer->W[n][j] -= lr * layer->dJdW[n][j];

      // printf("Weights: %f \n", layer->W[n][j]);
    }
  }

  // Compute dJdb for the Layer
  for (unsigned int n = 0; n < layer->neurons; n++) {

    // Reset values
    layer->dJdb[n] = 0.0;

    for (unsigned int i = 0; i < m; i++) {
      layer->dJdb[n] += layer->dJdZ[n][i] / m;
    }

    layer->b[n] -= lr * layer->dJdb[n];
    // printf("Bias: %f \n", layer->b[n]);
  }

  // Free memory
  for (unsigned int n = 0; n < next_layer->prev_neurons; n++) {
    free(W_t[n]);
  }
  free(W_t);

  for (unsigned int i = 0; i < m; i++) {
    free(A_t[i]);
  }
  free(A_t);
}

// Compute cost function
float compute_cost(float **y_pred, const float *y_true) {
  float total_cost = 0.0;
  for (unsigned char i = 0; i < m; i++) {
    total_cost += sqr(y_pred[0][i] - y_true[i]);
  }
  return total_cost / (2 * m);
}
