#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;
float learning_rate =0.99;
#define Epochs 1000
#define training_images 128 //upper limit is 60000
#define inference_images 10000 //upper limit is 10000
#define batchSize 32
#define HiddenLayer1_Size 32 //lower limit is 10
#define Input_Size 28 //do not chnage
#define Output_Size 10 //do not change
// Forward proporgation matrices
int no_of_images = training_images;
vector<vector<float>> A0(Input_Size*Input_Size,vector<float>(training_images));
vector<vector<float>> W1(HiddenLayer1_Size,vector<float>(Input_Size*Input_Size));
vector<float> B1(HiddenLayer1_Size);
vector<vector<float>> Z1(HiddenLayer1_Size,vector<float>(training_images));
vector<vector<float>> A1(HiddenLayer1_Size,vector<float>(training_images));
vector<vector<float>> W2(Output_Size,vector<float>(HiddenLayer1_Size));
vector<float> B2(Output_Size);    
vector<vector<float>> Z2(Output_Size,vector<float>(training_images));
vector<vector<float>> A2(Output_Size,vector<float>(training_images));
vector<vector<float>> labels(Output_Size,vector<float>(training_images,0));
// Backward proporgation matrices
vector<vector<float>> dZ2(Output_Size,vector<float>(training_images));
vector<vector<float>> dW2(Output_Size,vector<float>(HiddenLayer1_Size));
vector<float> dB2(Output_Size);
vector<vector<float>> dZ1(HiddenLayer1_Size,vector<float>(training_images));
vector<vector<float>> dW1(HiddenLayer1_Size,vector<float>(Input_Size*Input_Size));
vector<float> dB1(HiddenLayer1_Size);
vector<vector<float>> W2_T(HiddenLayer1_Size,vector<float>(Output_Size));
vector<vector<float>> A0_T(training_images,vector<float>(Input_Size*Input_Size));
vector<vector<float>> A1_T(training_images,vector<float>(HiddenLayer1_Size));


void matrix_multiply(vector<vector<float>>& output,vector<vector<float>>& weight,vector<vector<float>>& input,int row1,int col1,int row2,int col2){
    for(int i=0;i<row1;i++){
        for(int j=0;j<col2;j++){
            output[i][j]=0;
            for(int k=0;k<col1;k++){
                output[i][j]+=(weight[i][k]*input[k][j]);
            }
        }
    }
    
}
void bias_addition(vector<vector<float>>& input,vector<float>& bias,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            input[i][j]+=bias[i];
        }
    }
}
void ReLU(vector<vector<float>>& output,vector<vector<float>>& input,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(input[i][j]<0) output[i][j]=0;
            else output[i][j]=input[i][j];
        }
    }
}
void SoftMax(vector<vector<float>>& output,vector<vector<float>>& input,int row,int col){
    for(int j=0;j<col;j++){
        float sum=0;
        float maxval=input[0][j];
        for(int i=1;i<row;i++){
            if(maxval<input[i][j]) maxval=input[i][j];
        }
        for(int i=0;i<row;i++){
            output[i][j]=expf(input[i][j]-maxval);
            sum+=output[i][j];
        }
        for(int i=0;i<row;i++){
            output[i][j]/=sum;
        }
    }
}
void forward_proporgation(){
    matrix_multiply(Z1,W1,A0,HiddenLayer1_Size,Input_Size*Input_Size,Input_Size*Input_Size,no_of_images);
    bias_addition(Z1,B1,HiddenLayer1_Size,no_of_images);
    ReLU(A1,Z1,HiddenLayer1_Size,no_of_images);
    matrix_multiply(Z2,W2,A1,Output_Size,HiddenLayer1_Size,HiddenLayer1_Size,no_of_images);
    bias_addition(Z2,B2,Output_Size,no_of_images);
    SoftMax(A2,Z2,Output_Size,no_of_images); 
}
// void matrix_subtraction(float **output,float **input1,float **input2,int row,int col){
    
// }
// void ReLU_Derivative(float **gradient,int row,int col){
    
// }
// void elementwise_matrix_multiply(float **output,float **input,int row,int col){
    
// }
// int max(float **matrix,int col){
//     float max=matrix[0][col],index=0;
//     for(int y=1;y<Output_Size;y++){
//         if(matrix[y][col]>max){
//             max=matrix[y][col];
//             index=y;
//         }
//     }
//     return index;
// }

// void backward_proporgation(Forward_Matrices *forward,Backward_Matrices *backward,int no_of_images){
//     for(int i=0;i<HiddenLayer1_Size;i++){
//         for(int j=0;j<no_of_images;j++){
//             backward->A1_T[j][i]=forward->A1[i][j];
//         }
//     }
//     for(int i=0;i<Output_Size;i++){
//         for(int j=0;j<HiddenLayer1_Size;j++){
//             backward->W2_T[j][i]=forward->W2[i][j];
//         }
//     }
//     matrix_subtraction(backward->dZ2,forward->A2,forward->labels,Output_Size,no_of_images);
//     matrix_multiply(backward->dW2,backward->dZ2,backward->A1_T,Output_Size,no_of_images,no_of_images,HiddenLayer1_Size);
//     for(int i=0;i<Output_Size;i++){
//         for(int j=0;j<HiddenLayer1_Size;j++){
//             backward->dW2[i][j]/=no_of_images;
//         }
//     }
//     for(int i=0;i<Output_Size;i++){
//         backward->dB2[i]=0;
//         for(int j=0;j<no_of_images;j++){
//             backward->dB2[i]+=(backward->dZ2[i][j]/(float)no_of_images);
//         }
//     }
//     matrix_multiply(backward->dZ1,backward->W2_T,backward->dZ2,HiddenLayer1_Size,Output_Size,Output_Size,no_of_images);
//     ReLU_Derivative(forward->Z1,HiddenLayer1_Size,no_of_images);
//     elementwise_matrix_multiply(backward->dZ1,forward->Z1,HiddenLayer1_Size,no_of_images);
//     matrix_multiply(backward->dW1,backward->dZ1,forward->X,HiddenLayer1_Size,no_of_images,no_of_images,Input_Size*Input_Size);
//     for(int i=0;i<HiddenLayer1_Size;i++){
//         for(int j=0;j<Input_Size*Input_Size;j++){
//             backward->dW1[i][j]/=no_of_images;
//         }
//     }
//     for(int i=0;i<HiddenLayer1_Size;i++){
//         backward->dB1[i]=0;
//         for(int j=0;j<no_of_images;j++){
//             backward->dB1[i]+=(backward->dZ1[i][j]/(float)no_of_images);
//         }
//     }
// }
// void update_parameter(Forward_Matrices *forward,Backward_Matrices *backward){
//     for(int i=0;i<HiddenLayer1_Size;i++){
//         forward->B1[i]=(forward->B1[i]-learning_rate*backward->dB1[i]);
//         for(int j=0;j<Input_Size*Input_Size;j++){
//             forward->W1[i][j]=(forward->W1[i][j]-learning_rate*backward->dW1[i][j]);
//         }
//     }

//     for(int i=0;i<Output_Size;i++){
//         forward->B2[i]=(forward->B2[i]-learning_rate*backward->dB2[i]);
//         for(int j=0;j<HiddenLayer1_Size;j++){
//             forward->W2[i][j]=(forward->W2[i][j]-learning_rate*backward->dW2[i][j]);
//         }
//     }
// }

// void Writing_Trained_data(Forward_Matrices *forward){
//     //writing data to files
//     FILE *weight1=fopen("weight.txt","w");
//     FILE *bias1=fopen("bias.txt","w");
//     if (!weight1||!bias1) {
//         printf("Training Failed , data could not be saved.\n");
//         exit(EXIT_FAILURE);
//     }

//     for(int i=0;i<HiddenLayer1_Size;i++){
//         for(int j = 0; j < Input_Size*Input_Size; j++) {
//             fprintf(weight1, "%.16f ", forward->W1[i][j]);
//         }
//     }
//     for(int i = 0; i < Output_Size ; i++) {
//         for(int j=0;j< HiddenLayer1_Size ;j++){
//             fprintf(weight1, "%.16f ", forward->W2[i][j]);
//         }
//     }

//     for(int i=0;i<HiddenLayer1_Size;i++){
//         fprintf(bias1, "%.16f ", forward->B1[i]);
//     }
//     for(int i=0;i<Output_Size;i++){
//         fprintf(bias1, "%.16f ", forward->B2[i]);
//     }   
//     fclose(weight1);
//     fclose(bias1);
// } 
void reading_dataset(bool Training){
    FILE *image=NULL ;
    FILE *label=NULL ;
    int no_of_images=0;
    if(Training){
        image = fopen("train-images-idx3-ubyte","rb");
        label = fopen("train-labels-idx1-ubyte","rb");
        no_of_images=training_images;
    }
    else {
        image = fopen("t10k-images-idx3-ubyte","rb");
        label = fopen("t10k-labels-idx1-ubyte","rb");
        no_of_images=inference_images;
    }

    if (!image||!label) {
        printf("Error opening file \nTraining Failed.\n");
        exit(EXIT_FAILURE);
    }
    fseek(image,16,SEEK_SET);
    fseek(label,8,SEEK_SET);
    
    
    unsigned char curr_label;
    vector<unsigned char> input(Input_Size*Input_Size,0);

    for(int currimage=0;currimage<no_of_images;currimage++){ 
        // reading label
        fread(&curr_label,sizeof(unsigned char),1,label);
        labels[curr_label][currimage]=1;
        
        // reading dataset
        fread(input.data(),sizeof(unsigned char),Input_Size*Input_Size,image);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                A0[i * 28 + j][currimage] = input[i * 28 + j]/255.0f;
                A0_T[currimage][i * 28 + j] = input[i * 28 + j]/255.0f;
            }
        }
    }
    fclose(image);
    fclose(label);
}
void initalize_weights_bias(){
    //Generating initalize random bias and weights
    srand(time(NULL));
    
    //initalize weights
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            W1[i][j]=((float)rand() / (float)RAND_MAX - 0.5f) * sqrtf(2.0f / (Input_Size * Input_Size));
        }
    }

    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            W2[i][j]=((float)rand() / (float)RAND_MAX - 0.5f) * sqrtf(2.0f / (HiddenLayer1_Size));
        }
    }
    
    // initilaizing bias 
    for(int i=0;i<HiddenLayer1_Size;i++){
        B1[i]=0.0;
    }
    for(int i=0;i<Output_Size;i++){
        B2[i]=0.0;
    }
}

void Training_Mode(){
    // run this part only one time for clean start from ground up
    reading_dataset(true);
    initalize_weights_bias();
    forward_proporgation();
    for(int j=0;j<training_images;j++){
        printf("Image:%d\n",j);
        for(int i=0;i<Output_Size;i++){
            printf("%d:%.3f ",i,A2[i][j]);
        }
        printf("\n");
        for(int i=0;i<Output_Size;i++){
            printf("%d:%.3f ",i,labels[i][j]);
        }
        printf("\n");
    }

    // backward_proporgation();
    // update_parameter();
    // learning_rate=learning_rate*0.5*(1 + cos(M_PI*Epoch/Epochs));
    // Writing_Trained_data(&forward);
    // free_forward_memory(&forward,no_of_images);
    // free_backward_memory(&backward,no_of_images);
}

int main() {
    Training_Mode();   
} 