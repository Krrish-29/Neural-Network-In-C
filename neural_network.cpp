#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <cblas.h>
#include <time.h>
using namespace std;
float learning_rate = 0.1;
#define Epochs 100
#define training_images 60000 //upper limit is 60000
#define batchSize 32
#define HiddenLayer1_Size 32 //lower limit is 10
#define inference_images 10000 //upper limit is 10000

#define Input_Size 28 //do not chnage
#define Output_Size 10 //do not change
// Forward proporgation matrices
class Forward{
public:
    vector<float> X;
    vector<float> A0;
    vector<float> W1;
    vector<float> B1;
    vector<float> Z1;
    vector<float> A1;
    vector<float> W2;
    vector<float> B2;
    vector<float> Z2;
    vector<float> A2;
    vector<float> labels;
    Forward(int images,int batchLimit){
        X.resize(Input_Size*Input_Size*images,0);
        A0.resize(Input_Size*Input_Size*batchLimit,0);
        W1.resize(HiddenLayer1_Size*Input_Size*Input_Size,0);
        B1.resize(HiddenLayer1_Size);
        Z1.resize(HiddenLayer1_Size*batchLimit,0);
        A1.resize(HiddenLayer1_Size*batchLimit,0);
        W2.resize(Output_Size*HiddenLayer1_Size,0);
        B2.resize(Output_Size);
        Z2.resize(Output_Size*batchLimit,0);
        A2.resize(Output_Size*batchLimit,0);
        labels.resize(Output_Size*images,0);
    }
};
// Backward proporgation matrices
class Backward{
public:
    vector<float> dZ2;
    vector<float> dW2;
    vector<float> dB2;
    vector<float> dZ1;
    vector<float> dW1;
    vector<float> dB1;
    Backward(int batchLimit){
        dZ2.resize(Output_Size*batchLimit,0);
        dW2.resize(Output_Size*HiddenLayer1_Size,0);
        dB2.resize(Output_Size);
        dZ1.resize(HiddenLayer1_Size*batchLimit,0);
        dW1.resize(HiddenLayer1_Size*Input_Size*Input_Size,0);
        dB1.resize(HiddenLayer1_Size);
    }
};

void matrix_multiply(vector<float>& output,vector<float>& input1,vector<float>& input2,int row1,int col1,int col2,bool transpose1,bool transpose2){
    // this is explictly made that the transpose1 =  true will use the input1 as transpose and the matrix passed in input1 will be normal not transposed
    // and same thing for input2 and transpose2
    CBLAS_TRANSPOSE transA = transpose1 ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = transpose2 ? CblasTrans : CblasNoTrans;

    int M = row1;   // rows of output
    int N = col2;   // cols of output
    int K = col1;   // shared dimension

    cblas_sgemm(
        CblasRowMajor,
        transA,
        transB,
        M,             // rows of A and C
        N,             // cols of B and C
        K,             // cols of A / rows of B
        1.0f,          // alpha
        input1.data(), // A
        transpose1 ? M : K, // lda
        input2.data(), // B
        transpose2 ? K : N, // ldb
        0.0f,          // beta
        output.data(), // C
        N              // ldc
    );
    // #pragma omp parallel for
    // for(int i=0;i<row1;i++){
    //     for(int j=0;j<col2;j++){
    //         output[(i*col2)+j]=0;
    //         float a,b;
    //         for(int k=0;k<col1;k++){
    //             a = transpose1 ? input1[(k*row1)+i] : input1[(i*col1)+k];
    //             b = transpose2 ? input2[(j*col1)+k] : input2[(k*col2)+j];
    //             output[(i*col2)+j] += a * b;
    //         }
    //     }
    // }
}
void bias_addition(vector<float>& input,vector<float>& bias,int row,int col){
    #pragma omp parallel for
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            input[(i*col)+j]+=bias[i];
        }
    }
}
void ReLU(vector<float>& output,vector<float>& input,int row,int col){
    #pragma omp parallel for
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(input[(i*col)+j]<0) output[(i*col)+j]=0;
            else output[(i*col)+j]=input[(i*col)+j];
        }
    }
}
void SoftMax(vector<float>& output,vector<float>& input,int row,int col){
    #pragma omp parallel for
    for(int j=0;j<col;j++){
        float sum=0;
        float maxval=input[(0*col)+j];
        for(int i=1;i<row;i++){
            if(maxval<input[(i*col)+j]) maxval=input[(i*col)+j];
        }
        for(int i=0;i<row;i++){
            output[(i*col)+j]=expf(input[(i*col)+j]-maxval);
            sum+=output[(i*col)+j];
        }
        for(int i=0;i<row;i++){
            output[(i*col)+j]/=sum;
        }
    }
}
void forward_proporgation(Forward *forward,int start,int end){
    int batchLimit=end-start;
    matrix_multiply(forward->Z1,forward->W1,forward->A0,HiddenLayer1_Size,Input_Size*Input_Size,batchLimit,false,false);
    bias_addition(forward->Z1,forward->B1,HiddenLayer1_Size,batchLimit);
    ReLU(forward->A1,forward->Z1,HiddenLayer1_Size,batchLimit);
    matrix_multiply(forward->Z2,forward->W2,forward->A1,Output_Size,HiddenLayer1_Size,batchLimit,false,false);
    bias_addition(forward->Z2,forward->B2,Output_Size,batchLimit);
    SoftMax(forward->A2,forward->Z2,Output_Size,batchLimit); 
}
void matrix_subtraction(vector<float>& output,vector<float>& input1,vector<float>& input2,int row,int col,int batchStart){
    #pragma omp parallel for
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            output[(i*col)+j]=input1[(i*col)+j]-input2[(i*training_images)+batchStart+j];
        }
    }
}
void ReLU_Derivative(vector<float>& gradient,int row,int col){
    #pragma omp parallel for
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(gradient[(i*col)+j]<0) gradient[(i*col)+j]=0;
            else gradient[(i*col)+j]=1;
        }
    }    
}
void elementwise_matrix_multiply(vector<float>& output,vector<float>& input,int row,int col){
    #pragma omp parallel for
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            output[(i*col)+j]*=input[(i*col)+j];
        }
    }
}

void backward_proporgation(Forward *forward,Backward *backward,int start,int end){
    int batchLimit=end-start;
    matrix_subtraction(backward->dZ2,forward->A2,forward->labels,Output_Size,batchLimit,start);

    matrix_multiply(backward->dW2,backward->dZ2,forward->A1,Output_Size,batchLimit,HiddenLayer1_Size,false,true);

    #pragma omp parallel for
    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            backward->dW2[(i*HiddenLayer1_Size)+j]/=batchLimit;
        }
    }

    for(int i=0;i<Output_Size;i++){
        backward->dB2[i]=0;
        for(int j=0;j<batchLimit;j++){
            backward->dB2[i]+=(backward->dZ2[(i*batchLimit)+j]/batchLimit);
        }
    }

    matrix_multiply(backward->dZ1,forward->W2,backward->dZ2,HiddenLayer1_Size,Output_Size,batchLimit,true,false);
    ReLU_Derivative(forward->Z1,HiddenLayer1_Size,batchLimit);
    elementwise_matrix_multiply(backward->dZ1,forward->Z1,HiddenLayer1_Size,batchLimit);
    matrix_multiply(backward->dW1,backward->dZ1,forward->A0,HiddenLayer1_Size,batchLimit,Input_Size*Input_Size,false,true);

    #pragma omp parallel for
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=0;j<Input_Size*Input_Size;j++){
            backward->dW1[(i*Input_Size*Input_Size)+j]/=batchLimit;
        }
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        backward->dB1[i]=0;
        for(int j=0;j<batchLimit;j++){
            backward->dB1[i]+=(backward->dZ1[(i*batchLimit)+j]/batchLimit);
        }
    }
}
void update_parameter(Forward *forward,Backward *backward){
    for(int i=0;i<HiddenLayer1_Size;i++){
        forward->B1[i]=(forward->B1[i]-learning_rate*backward->dB1[i]);
        for(int j=0;j<Input_Size*Input_Size;j++){
            forward->W1[(i*Input_Size*Input_Size)+j]=(forward->W1[(i*Input_Size*Input_Size)+j]-learning_rate*backward->dW1[(i*Input_Size*Input_Size)+j]);
        }
    }

    for(int i=0;i<Output_Size;i++){
        forward->B2[i]=(forward->B2[i]-learning_rate*backward->dB2[i]);
        for(int j=0;j<HiddenLayer1_Size;j++){
            forward->W2[(i*HiddenLayer1_Size)+j]=(forward->W2[(i*HiddenLayer1_Size)+j]-learning_rate*backward->dW2[(i*HiddenLayer1_Size)+j]);
        }
    }
}
int max(vector<float> &matrix,int row,int col,int total_images){
    float max=-1;
    int index=-1;
    for(int i=0;i<row;i++){
        if(matrix[(i*total_images)+col]>max){
            max=matrix[(i*total_images)+col];
            index=i;
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
            fscanf(weight, "%f", &forward->W1[(i*Input_Size*Input_Size)+j]);
        }
    }
    
    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            fscanf(weight, "%f", &forward->W2[(i*HiddenLayer1_Size)+j]);
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
            fprintf(weight, "%.16f ", forward->W1[(i*Input_Size*Input_Size)+j]);
        }
    }
    for(int i=0; i<Output_Size ; i++) {
        for(int j=0;j<HiddenLayer1_Size ;j++){
            fprintf(weight, "%.16f ", forward->W2[(i*HiddenLayer1_Size)+j]);
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
void batch_loader(Forward *forward,int start,int end,int total_images){
    int batchLimit=end-start;
    #pragma omp parallel for
    for(int image=0;image<batchLimit;image++){ 
        for (int i = 0; i < Input_Size*Input_Size; i++) {
            forward->A0[(i*batchLimit)+image] = forward->X[(i*total_images)+start+image];
        }
    }
}
void initalize_weights_bias(Forward *forward){
    //Generating initalize random bias and weights
    srand(time(NULL));
    
    //initalize weights
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            forward->W1[(i*Input_Size*Input_Size)+j]=((float)rand() / (float)RAND_MAX - 0.5f) * sqrtf(2.0f / (Input_Size * Input_Size));
        }
    }

    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            forward->W2[(i*HiddenLayer1_Size)+j]=((float)rand() / (float)RAND_MAX - 0.5f) * sqrtf(2.0f / (HiddenLayer1_Size));
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
void reading_dataset(bool training,Forward *forward){
    FILE *image=NULL;
    FILE *label=NULL;
    int no_of_images;
    if(training){
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
        forward->labels[(curr_label*no_of_images)+currimage]=1;
        
        // reading dataset
        fread(input.data(),sizeof(unsigned char),Input_Size*Input_Size,image);
        for (int i = 0; i < Input_Size*Input_Size; i++) {
            forward->X[(i*no_of_images)+currimage] = input[i]/255.0f;
        }
    }
    fclose(image);
    fclose(label);
}
void Training_Mode(){
    Forward *forward=new Forward(training_images,batchSize);
    Backward *backward=new Backward(batchSize);
    reading_dataset(true,forward);
    initalize_weights_bias(forward);
    printf("     ----- Traning Started -----\n");
    for(int Epoch=0;Epoch<Epochs;Epoch++){
        float Accuracy=0.0;
        int predIdx,actualIdx;
        
        for(int start=0;start<training_images;start+=batchSize){
            int end=min(start+batchSize,training_images);
            batch_loader(forward,start,end,training_images);
            forward_proporgation(forward,start,end);
            backward_proporgation(forward,backward,start,end);
            update_parameter(forward,backward);
            
            for(int x=start;x<end;x++){
                predIdx=max(forward->A2,Output_Size,x%batchSize,end-start);
                actualIdx=max(forward->labels,Output_Size,x,training_images);
                if(actualIdx==predIdx) Accuracy++;
            }
        }
        learning_rate=learning_rate*0.5*(1 + cos(M_PI*Epoch/Epochs));
        
        printf("Epoch:%d/%d | Accuracy: %.2f%%\n", Epoch,Epochs,100 * Accuracy / training_images);
    }
    printf("     ----- Training Completed! -----\n");
    
    Writing_Trained_data(forward);
    delete(forward);
    delete(backward);
}
void Inference_Mode(){
    Forward *forward=new Forward(inference_images,inference_images);
    reading_dataset(false,forward);
    read_weights_bias(forward);
    
    printf("     ----- Inference Started -----\n");
    batch_loader(forward,0,inference_images,inference_images);
    forward_proporgation(forward,0,inference_images);
    float Accuracy=0.0;
    int predIdx,actualIdx;
    for(int x=0;x<inference_images;x++){
        predIdx=max(forward->A2,Output_Size,x,inference_images);
        actualIdx=max(forward->labels,Output_Size,x,inference_images);
        if(actualIdx==predIdx) Accuracy++;
        else printf("     Wrong Prediction:-> \n     Sample Number :%d\n     Actual Value : %d\n     Model Prediction : %d\n\n",x+1,predIdx,actualIdx);
    }
    printf("     ----- Inference Completed! -----\n     Accuracy:%.2f\n",100*Accuracy/inference_images);
    delete(forward);
}
int main() {
    omp_set_num_threads(2);
    auto start = chrono::high_resolution_clock::now();
    int mode=0;
    printf("-----Welcome From Neural Network Made From Scratch In C-----\n");
    printf("     There are two operations modes :\n");
    printf("     1. Training Mode  (Recommended when first starting the network for warm up.)\n");
    printf("     2.Inference Mode  (Use only after Training is successfully completed.)\n");
    printf("     Press 1 for Training to commence , Press 2 for Inference to commence. : ");
    scanf("%d",&mode);
    if(mode==1) {
        Training_Mode();   
        Inference_Mode();  
    }
    else if(mode==2) {
        Inference_Mode();
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    printf("\ntime used :%f",diff.count());
    exit(EXIT_SUCCESS);
} 