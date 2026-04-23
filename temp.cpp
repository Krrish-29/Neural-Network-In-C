#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
using namespace std;

float learning_rate =0.99;
#define Epochs 1000
#define training_images 32000 //upper limit is 60000
#define batchSize 32
#define HiddenLayer1_Size 32 //lower limit is 10

#define inference_images 10000 //upper limit is 10000

#define Input_Size 28 //do not chnage
#define Output_Size 10 //do not change
// Forward proporgation matrices
class Forward{
public:
    vector<vector<float>> A0;
    vector<vector<float>> W1;
    vector<float> B1;
    vector<vector<float>> Z1;
    vector<vector<float>> A1;
    vector<vector<float>> W2;
    vector<float> B2;
    vector<vector<float>> Z2;
    vector<vector<float>> A2;
    vector<vector<float>> labels;
    Forward(int images){
        A0.resize(Input_Size*Input_Size,vector<float>(images,0));
        W1.resize(HiddenLayer1_Size,vector<float>(Input_Size*Input_Size,0));
        B1.resize(HiddenLayer1_Size);
        Z1.resize(HiddenLayer1_Size,vector<float>(images,0));
        A1.resize(HiddenLayer1_Size,vector<float>(images,0));
        W2.resize(Output_Size,vector<float>(HiddenLayer1_Size,0));
        B2.resize(Output_Size);    
        Z2.resize(Output_Size,vector<float>(images,0));
        A2.resize(Output_Size,vector<float>(images,0));
        labels.resize(Output_Size,vector<float>(images,0));
    }
};
// Backward proporgation matrices
class Backward{
public:
    vector<vector<float>> dZ2;
    vector<vector<float>> dW2;
    vector<float> dB2;
    vector<vector<float>> dZ1;
    vector<vector<float>> dW1;
    vector<float> dB1;
    vector<vector<float>> W2_T;
    vector<vector<float>> A0_T;
    vector<vector<float>> A1_T;
    Backward(int images){
        dZ2.resize(Output_Size,vector<float>(images,0));
        dW2.resize(Output_Size,vector<float>(HiddenLayer1_Size,0));
        dB2.resize(Output_Size);
        dZ1.resize(HiddenLayer1_Size,vector<float>(images,0));
        dW1.resize(HiddenLayer1_Size,vector<float>(Input_Size*Input_Size,0));
        dB1.resize(HiddenLayer1_Size);
        W2_T.resize(HiddenLayer1_Size,vector<float>(Output_Size,0));
        A0_T.resize(images,vector<float>(Input_Size*Input_Size,0));
        A1_T.resize(images,vector<float>(HiddenLayer1_Size,0));
    }
};

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
void forward_proporgation(Forward *forward,int start,int end){
    matrix_multiply(forward->Z1,forward->W1,forward->A0,0,HiddenLayer1_Size,0,Input_Size*Input_Size,start,end);
    bias_addition(forward->Z1,forward->B1,HiddenLayer1_Size,start,end);
    ReLU(forward->A1,forward->Z1,HiddenLayer1_Size,start,end);
    matrix_multiply(forward->Z2,forward->W2,forward->A1,0,Output_Size,0,HiddenLayer1_Size,start,end);
    bias_addition(forward->Z2,forward->B2,Output_Size,start,end);
    SoftMax(forward->A2,forward->Z2,Output_Size,start,end); 
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

void backward_proporgation(Forward *forward,Backward *backward,int start,int end){
    
    matrix_subtraction(backward->dZ2,forward->A2,forward->labels,Output_Size,start,end);

    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=start;j<end;j++){
            backward->A1_T[j][i]=forward->A1[i][j];
        }
    }
    
    matrix_multiply(backward->dW2,backward->dZ2,backward->A1_T,0,Output_Size,start,end,0,HiddenLayer1_Size);

    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            backward->dW2[i][j]/=batchSize;
        }
    }

    for(int i=0;i<Output_Size;i++){
        backward->dB2[i]=0;
        for(int j=start;j<end;j++){
            backward->dB2[i]+=(backward->dZ2[i][j]/batchSize);
        }
    }

    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            backward->W2_T[j][i]=forward->W2[i][j];
        }
    }

    matrix_multiply(backward->dZ1,backward->W2_T,backward->dZ2,0,HiddenLayer1_Size,0,Output_Size,start,end);
    ReLU_Derivative(forward->Z1,HiddenLayer1_Size,start,end);
    elementwise_matrix_multiply(backward->dZ1,forward->Z1,HiddenLayer1_Size,start,end);
    matrix_multiply(backward->dW1,backward->dZ1,backward->A0_T,0,HiddenLayer1_Size,start,end,0,Input_Size*Input_Size);

    int currBatchSize=end-start;
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=0;j<Input_Size*Input_Size;j++){
            backward->dW1[i][j]/=currBatchSize;
        }
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        backward->dB1[i]=0;
        for(int j=start;j<end;j++){
            backward->dB1[i]+=(backward->dZ1[i][j]/currBatchSize);
        }
    }
}
void update_parameter(Forward *forward,Backward *backward,int start,int end){
    for(int i=0;i<HiddenLayer1_Size;i++){
        forward->B1[i]=(forward->B1[i]-learning_rate*backward->dB1[i]);
        for(int j=0;j<Input_Size*Input_Size;j++){
            forward->W1[i][j]=(forward->W1[i][j]-learning_rate*backward->dW1[i][j]);
        }
    }

    for(int i=0;i<Output_Size;i++){
        forward->B2[i]=(forward->B2[i]-learning_rate*backward->dB2[i]);
        for(int j=0;j<HiddenLayer1_Size;j++){
            forward->W2[i][j]=(forward->W2[i][j]-learning_rate*backward->dW2[i][j]);
        }
    }
}
int max(vector<vector<float>> &matrix,int col){
    float max=matrix[0][col],index=0;
    for(int y=1;y<Output_Size;y++){
        if(matrix[y][col]>max){
            max=matrix[y][col];
            index=y;
        }
    }
    return index;
}
void read_weights_bias(Forward *forward){
    FILE *weight=fopen("weight.txt","r");
    FILE *bias=fopen("bias.txt","r");
    if (!weight||!bias) {
        printf("Error opening file.\n");
        exit(EXIT_FAILURE);
    }
    
    // initilaizing weigths 
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fscanf(weight, "%f", &forward->W1[i][j]);
        }
    }
    
    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            fscanf(weight, "%f", &forward->W2[i][j]);
        }
    }
    
    // initilaizing bias 
    for(int i=0;i<HiddenLayer1_Size;i++){
        fscanf(bias, "%f", &forward->B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fscanf(bias, "%f", &forward->B2[i]);
    }
    
    fclose(bias);
    fclose(weight);
}
void Writing_Trained_data(Forward *forward){
    //writing data to files
    FILE *weight=fopen("weight.txt","w");
    FILE *bias=fopen("bias.txt","w");
    if (!weight||!bias) {
        printf("Training Failed , data could not be saved.\n");
        exit(EXIT_FAILURE);
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fprintf(weight, "%.16f ", forward->W1[i][j]);
        }
    }
    for(int i = 0; i < Output_Size ; i++) {
        for(int j=0;j< HiddenLayer1_Size ;j++){
            fprintf(weight, "%.16f ", forward->W2[i][j]);
        }
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        fprintf(bias, "%.16f ", forward->B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fprintf(bias, "%.16f ", forward->B2[i]);
    }   
    fclose(weight);
    fclose(bias);
} 
void initalize_weights_bias(Forward *forward){
    //Generating initalize random bias and weights
    srand(time(NULL));
    
    //initalize weights
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            forward->W1[i][j]=((float)rand() / (float)RAND_MAX - 0.5f) * sqrtf(2.0f / (Input_Size * Input_Size));
        }
    }

    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            forward->W2[i][j]=((float)rand() / (float)RAND_MAX - 0.5f) * sqrtf(2.0f / (HiddenLayer1_Size));
        }
    }
    
    // initilaizing bias 
    for(int i=0;i<HiddenLayer1_Size;i++){
        forward->B1[i]=0.0;
    }
    for(int i=0;i<Output_Size;i++){
        forward->B2[i]=0.0;
    }
}
void reading_dataset(bool training,Forward *forward,Backward *backward=NULL){
    FILE *image=NULL;
    FILE *label=NULL;
    if(training){
        image = fopen("train-images-idx3-ubyte","rb");
        label = fopen("train-labels-idx1-ubyte","rb");
    }
    else {
        image = fopen("t10k-images-idx3-ubyte","rb");
        label = fopen("t10k-labels-idx1-ubyte","rb");
    }

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
        forward->labels[curr_label][currimage]=1;
        
        // reading dataset
        fread(input.data(),sizeof(unsigned char),Input_Size*Input_Size,image);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                forward->A0[i * 28 + j][currimage] = input[i * 28 + j]/255.0f;
                if(training) backward->A0_T[currimage][i * 28 + j] = input[i * 28 + j]/255.0f;
            }
        }
    }
    fclose(image);
    fclose(label);
}
void Training_Mode(){
    // run this part only one time for clean start from ground up
    Forward *forward=new Forward(training_images);
    Backward *backward=new Backward(training_images);
    reading_dataset(true,forward,backward);
    initalize_weights_bias(forward);
    printf("     ----- Traning Started -----\n");
    for(int Epoch=0;Epoch<Epochs;Epoch++){
        printf("Epoch:%d/%d |",Epoch,Epochs);
        
        for(int start=0;start<training_images;start+=batchSize){
            int end=min(start+batchSize,training_images);
            forward_proporgation(forward,start,end);
            backward_proporgation(forward,backward,start,end);
            update_parameter(forward,backward,start,end);
        }
        learning_rate=learning_rate*0.5*(1 + cos(M_PI*Epoch/Epochs));
        
        float Accuracy=0.0;
        int predIdx,actualIdx;
        for(int x=0;x<inference_images;x++){
            predIdx=max(forward->A2,x);
            actualIdx=max(forward->labels,x);
            if(actualIdx==predIdx) Accuracy++;
            else printf("     Wrong Prediction:-> \n     Sample Number :%d\n     Actual Value : %d\n     Model Prediction : %d\n\n",x+1,predIdx,actualIdx);
        }
        printf("Epoch: %d | Accuracy: %.2f%%\n", Epoch,100 * Accuracy / training_images);
    }
    printf("     ----- Training Completed! -----\n");
    
    Writing_Trained_data(forward);
}
void Inference_Mode(){
    Forward *forward=new Forward(inference_images);
    reading_dataset(false,forward);
    read_weights_bias(forward);

    printf("     ----- Inference Started -----\n");
    forward_proporgation(forward,0,inference_images);
    float Accuracy=0.0;
    int predIdx,actualIdx;
    for(int x=0;x<inference_images;x++){
        predIdx=max(forward->A2,x);
        actualIdx=max(forward->labels,x);
        if(actualIdx==predIdx) Accuracy++;
        else printf("     Wrong Prediction:-> \n     Sample Number :%d\n     Actual Value : %d\n     Model Prediction : %d\n\n",x+1,predIdx,actualIdx);
    }
    printf("     ----- Inference Completed! -----\n     Accuracy:%.2f\n",100*Accuracy/inference_images);
}
int main() {
    Training_Mode(); 
    Inference_Mode();  
} 
