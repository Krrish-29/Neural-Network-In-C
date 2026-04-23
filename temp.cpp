#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;
float learning_rate =0.09;
#define Epochs 1000
#define training_images 320 //upper limit is 60000
// #define inference_images 10000 //upper limit is 10000
#define batchSize 32
#define HiddenLayer1_Size 32 //lower limit is 10
#define Input_Size 28 //do not chnage
#define Output_Size 10 //do not change
// Forward proporgation matrices
vector<vector<float>> A0(Input_Size*Input_Size,vector<float>(training_images,0));
vector<vector<float>> W1(HiddenLayer1_Size,vector<float>(Input_Size*Input_Size,0));
vector<float> B1(HiddenLayer1_Size);
vector<vector<float>> Z1(HiddenLayer1_Size,vector<float>(training_images,0));
vector<vector<float>> A1(HiddenLayer1_Size,vector<float>(training_images,0));
vector<vector<float>> W2(Output_Size,vector<float>(HiddenLayer1_Size,0));
vector<float> B2(Output_Size);    
vector<vector<float>> Z2(Output_Size,vector<float>(training_images,0));
vector<vector<float>> A2(Output_Size,vector<float>(training_images,0));
vector<vector<float>> labels(Output_Size,vector<float>(training_images,0));
// Backward proporgation matrices
vector<vector<float>> dZ2(Output_Size,vector<float>(training_images,0));
vector<vector<float>> dW2(Output_Size,vector<float>(HiddenLayer1_Size,0));
vector<float> dB2(Output_Size);
vector<vector<float>> dZ1(HiddenLayer1_Size,vector<float>(training_images,0));
vector<vector<float>> dW1(HiddenLayer1_Size,vector<float>(Input_Size*Input_Size,0));
vector<float> dB1(HiddenLayer1_Size);
vector<vector<float>> W2_T(HiddenLayer1_Size,vector<float>(Output_Size,0));
vector<vector<float>> A0_T(training_images,vector<float>(Input_Size*Input_Size,0));
vector<vector<float>> A1_T(training_images,vector<float>(HiddenLayer1_Size,0));


void matrix_multiply(vector<vector<float>>& output,vector<vector<float>>& weight,vector<vector<float>>& input,int start_row1,int end_row1,int start_col1,int end_col1,int start_col2,int end_col2){
    for(int i=start_row1;i<end_row1;i++){
        for(int j=start_col2;j<end_col2;j++){
            output[i][j]=0;
            for(int k=start_col1;k<end_col1;k++){
                output[i][j]+=(weight[i][k]*input[k][j]);
            }
        }
    }
    
}
void bias_addition(vector<vector<float>>& input,vector<float>& bias,int row,int start_col,int end_col){
    for(int i=0;i<row;i++){
        for(int j=start_col;j<end_col;j++){
            input[i][j]+=bias[i];
        }
    }
}
void ReLU(vector<vector<float>>& output,vector<vector<float>>& input,int row,int start_col,int end_col){
    for(int i=0;i<row;i++){
        for(int j=start_col;j<end_col;j++){
            if(input[i][j]<0) output[i][j]=0;
            else output[i][j]=input[i][j];
        }
    }
}
void SoftMax(vector<vector<float>>& output,vector<vector<float>>& input,int row,int start_col,int end_col){
    for(int j=start_col;j<end_col;j++){
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
void forward_proporgation(int start,int end){
    matrix_multiply(Z1,W1,A0,0,HiddenLayer1_Size,0,Input_Size*Input_Size,start,end);
    bias_addition(Z1,B1,HiddenLayer1_Size,start,end);
    ReLU(A1,Z1,HiddenLayer1_Size,start,end);
    matrix_multiply(Z2,W2,A1,0,Output_Size,0,HiddenLayer1_Size,start,end);
    bias_addition(Z2,B2,Output_Size,start,end);
    SoftMax(A2,Z2,Output_Size,start,end); 
}
void matrix_subtraction(vector<vector<float>>& output,vector<vector<float>>& input1,vector<vector<float>>& input2,int row,int start_col,int end_col){
    for(int i=0;i<row;i++){
        for(int j=start_col;j<end_col;j++){
            output[i][j]=input1[i][j]-input2[i][j];
        }
    }
    
}
void ReLU_Derivative(vector<vector<float>>& gradient,int row,int start_col,int end_col){
    for(int i=0;i<row;i++){
        for(int j=start_col;j<end_col;j++){
            if(gradient[i][j]<0) gradient[i][j]=0;
            else gradient[i][j]=1;
        }
    }    
}
void elementwise_matrix_multiply(vector<vector<float>>& output,vector<vector<float>>& input,int row,int start_col,int end_col){
        for(int i=0;i<row;i++){
            for(int j=start_col;j<end_col;j++){
                output[i][j]*=input[i][j];
            }
        }
}

void backward_proporgation(int start,int end){
    
    matrix_subtraction(dZ2,A2,labels,Output_Size,start,end);

    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=start;j<end;j++){
            A1_T[j][i]=A1[i][j];
        }
    }
    
    matrix_multiply(dW2,dZ2,A1_T,0,Output_Size,start,end,0,HiddenLayer1_Size);

    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            dW2[i][j]/=batchSize;
        }
    }

    for(int i=0;i<Output_Size;i++){
        dB2[i]=0;
        for(int j=start;j<end;j++){
            dB2[i]+=(dZ2[i][j]/batchSize);
        }
    }

    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            W2_T[j][i]=W2[i][j];
        }
    }

    matrix_multiply(dZ1,W2_T,dZ2,0,HiddenLayer1_Size,0,Output_Size,start,end);
    ReLU_Derivative(Z1,HiddenLayer1_Size,start,end);
    elementwise_matrix_multiply(dZ1,Z1,HiddenLayer1_Size,start,end);
    matrix_multiply(dW1,dZ1,A0_T,0,HiddenLayer1_Size,start,end,0,Input_Size*Input_Size);
    int currBatchSize=end-start;
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=0;j<Input_Size*Input_Size;j++){
            dW1[i][j]/=currBatchSize;
        }
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        dB1[i]=0;
        for(int j=start;j<end;j++){
            dB1[i]+=(dZ1[i][j]/currBatchSize);
        }
    }
}
void update_parameter(int start,int end){
    for(int i=0;i<HiddenLayer1_Size;i++){
        B1[i]=(B1[i]-learning_rate*dB1[i]);
        for(int j=0;j<Input_Size*Input_Size;j++){
            W1[i][j]=(W1[i][j]-learning_rate*dW1[i][j]);
        }
    }

    for(int i=0;i<Output_Size;i++){
        B2[i]=(B2[i]-learning_rate*dB2[i]);
        for(int j=0;j<HiddenLayer1_Size;j++){
            W2[i][j]=(W2[i][j]-learning_rate*dW2[i][j]);
        }
    }
}

void Writing_Trained_data(){
    //writing data to files
    FILE *weight=fopen("weight.txt","w");
    FILE *bias=fopen("bias.txt","w");
    if (!weight||!bias) {
        printf("Training Failed , data could not be saved.\n");
        exit(EXIT_FAILURE);
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fprintf(weight, "%.16f ", W1[i][j]);
        }
    }
    for(int i = 0; i < Output_Size ; i++) {
        for(int j=0;j< HiddenLayer1_Size ;j++){
            fprintf(weight, "%.16f ", W2[i][j]);
        }
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        fprintf(bias, "%.16f ", B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fprintf(bias, "%.16f ", B2[i]);
    }   
    fclose(weight);
    fclose(bias);
} 
void reading_dataset(){
    FILE *image=NULL ;
    FILE *label=NULL ;
    image = fopen("train-images-idx3-ubyte","rb");
    label = fopen("train-labels-idx1-ubyte","rb");

    if (!image||!label) {
        printf("Error opening file \nTraining Failed.\n");
        exit(EXIT_FAILURE);
    }
    fseek(image,16,SEEK_SET);
    fseek(label,8,SEEK_SET);
    
    unsigned char curr_label;
    vector<unsigned char> input(Input_Size*Input_Size,0);

    for(int currimage=0;currimage<training_images;currimage++){ 
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
    for(int Epoch=0;Epoch<Epochs;Epoch++){

        for(int start=0;start<training_images;start+=batchSize){
            printf("batch:%d",batch+1);
            int end=min(start+batchSize,training_images);
            forward_proporgation(start,end);
            backward_proporgation(start,end);
            update_parameter(start,end);
        }
        learning_rate=learning_rate*0.5*(1 + cos(M_PI*Epoch/Epochs));

        int correct=0;
        for(int j=0;j<training_images;j++){
            int pred=0;
            float maxv=A2[0][j];
            for(int i=1;i<Output_Size;i++){
                if(A2[i][j] > maxv) {
                    maxv = A2[i][j];
                    pred = i;
                }
            }
            int actual = 0;
            for(int i = 0; i < Output_Size; i++) {
                if(labels[i][j] == 1) {
                    actual = i;
                    break;
                }
            }
            if(pred == actual) correct++;
        }
        printf("Epoch %d | Accuracy: %.2f%%\n", Epoch,100.0f * correct / training_images);
    }

    Writing_Trained_data();
}

int main() {
    Training_Mode();   
} 
