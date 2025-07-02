#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#define HiddenLayer1_Size 32 //lower limit is 10
float learning_rate =0.1;
#define Epochs 1000 
#define training_images 1000 //upper limit is 60000
#define inference_images 1000 //upper limit is 10000


#define Input_Size 28 //do not chnage
#define Output_Size 10 //do not change
typedef struct {
    // Forward proporgation matrices
    unsigned char *input;
    float **X,**Z1,**W1,**A0,*B1;
    float **Z2,**W2,**A1,*B2,**A2;
    float **labels;
}Forward_Matrices;
typedef struct {
    // Backward proporgation matrices
    float **dZ2,**dW2,**A1_T,*dB2;
    float **dZ1,**W2_T,**dW1,*dB1;
}Backward_Matrices;
void initalize_matrices_forward(Forward_Matrices *forward,int no_of_images){
    forward->input=(unsigned char*)calloc(Input_Size*Input_Size,sizeof(unsigned char));
    forward->X=(float**)calloc(no_of_images,sizeof(float*));

    forward->Z1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    forward->W1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    forward->A0=(float**)calloc(Input_Size*Input_Size,sizeof(float*));
    forward->B1=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    
    forward->Z2=(float**)calloc(Output_Size,sizeof(float*));
    forward->W2=(float**)calloc(Output_Size,sizeof(float*));
    forward->A1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    forward->B2=(float*)calloc(Output_Size,sizeof(float)); 
    
    forward->A2=(float**)calloc(Output_Size,sizeof(float*));
    
    forward->labels=(float**)calloc(Output_Size,sizeof(float*));

    for(int i=0;i<no_of_images;i++){
        forward->X[i]=(float*)calloc(Input_Size*Input_Size,sizeof(float));
    }
    for(int i=0;i<Input_Size*Input_Size;i++){
        forward->A0[i]=(float*)calloc(no_of_images,sizeof(float));
    }

    for(int i=0;i<Output_Size;i++){
        forward->W2[i]=(float*)calloc(HiddenLayer1_Size,sizeof(float));
        forward->A2[i]=(float*)calloc(no_of_images,sizeof(float));
        forward->Z2[i]=(float*)calloc(no_of_images,sizeof(float));
        forward->labels[i]=(float*)calloc(no_of_images,sizeof(float));
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        forward->W1[i]=(float*)calloc(Input_Size*Input_Size,sizeof(float));
        forward->A1[i]=(float*)calloc(no_of_images,sizeof(float));
        forward->Z1[i]=(float*)calloc(no_of_images,sizeof(float));
    }
}
void initalize_matrices_backward(Backward_Matrices *backward,int no_of_images){
    backward->dZ2=(float**)calloc(Output_Size,sizeof(float*));
    backward->dW2=(float**)calloc(Output_Size,sizeof(float*));
    backward->A1_T=(float**)calloc(no_of_images,sizeof(float*));
    backward->dB2=(float*)calloc(Output_Size,sizeof(float));

    backward->dZ1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    backward->W2_T=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    backward->dW1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    backward->dB1=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    
    for(int i=0;i<Output_Size;i++){
        backward->dZ2[i]=(float*)calloc(no_of_images,sizeof(float));
        backward->dW2[i]=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        backward->dZ1[i]=(float*)calloc(no_of_images,sizeof(float));
        backward->W2_T[i]=(float*)calloc(Output_Size,sizeof(float));
        backward->dW1[i]=(float*)calloc(Input_Size*Input_Size,sizeof(float));
    }
    for(int i=0;i<no_of_images;i++){
        backward->A1_T[i]=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    }
}
void matrix_multiply(float **output,float **weight,float **input,int row1,int col1,int row2,int col2){
    for(int i=0;i<row1;i++){
        for(int j=0;j<col2;j++){
            output[i][j]=0;
            for(int k=0;k<col1;k++){
                output[i][j]+=(weight[i][k]*input[k][j]);
            }
        }
    }
}
void bias_addition(float **input,float *bias,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            input[i][j]+=bias[i];
        }
    }
}
void matrix_subtraction(float **output,float **input1,float **input2,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            output[i][j]=input1[i][j]-input2[i][j];
        }
    }
}
void ReLU(float **output,float **input,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(input[i][j]<=0) output[i][j]=0;
            else output[i][j]=input[i][j];
        }
    }
}
void SoftMax(float **output,float **input,int row,int col){
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
void ReLU_Derivative(float **gradient,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            if(gradient[i][j]<0) gradient[i][j]=0;
            else gradient[i][j]=1;
        }
    }
}
void elementwise_matrix_multiply(float **output,float **input,int row,int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col;j++){
            output[i][j]*=(input[i][j]);
        }
    }
}
void forward_proporgation(Forward_Matrices *forward,int no_of_images){
    matrix_multiply(forward->Z1,forward->W1,forward->A0,HiddenLayer1_Size,Input_Size*Input_Size,Input_Size*Input_Size,no_of_images);
    bias_addition(forward->Z1,forward->B1,HiddenLayer1_Size,no_of_images);
    ReLU(forward->A1,forward->Z1,HiddenLayer1_Size,no_of_images);
    matrix_multiply(forward->Z2,forward->W2,forward->A1,Output_Size,HiddenLayer1_Size,HiddenLayer1_Size,no_of_images);
    bias_addition(forward->Z2,forward->B2,Output_Size,no_of_images);
    SoftMax(forward->A2,forward->Z2,Output_Size,no_of_images); 

}
void backward_proporgation(Forward_Matrices *forward,Backward_Matrices *backward,int no_of_images){
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=0;j<no_of_images;j++){
            backward->A1_T[j][i]=forward->A1[i][j];
        }
    }
    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            backward->W2_T[j][i]=forward->W2[i][j];
        }
    }
    matrix_subtraction(backward->dZ2,forward->A2,forward->labels,Output_Size,no_of_images);
    matrix_multiply(backward->dW2,backward->dZ2,backward->A1_T,Output_Size,no_of_images,no_of_images,HiddenLayer1_Size);
    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<HiddenLayer1_Size;j++){
            backward->dW2[i][j]/=no_of_images;
        }
    }
    for(int i=0;i<Output_Size;i++){
        for(int j=0;j<no_of_images;j++){
            backward->dB2[i]+=backward->dZ2[i][j];
        }
        backward->dB2[i]/=(float)no_of_images;
    }
    matrix_multiply(backward->dZ1,backward->W2_T,backward->dZ2,HiddenLayer1_Size,Output_Size,Output_Size,no_of_images);
    ReLU_Derivative(forward->Z1,HiddenLayer1_Size,no_of_images);
    elementwise_matrix_multiply(backward->dZ1,forward->Z1,HiddenLayer1_Size,no_of_images);
    matrix_multiply(backward->dW1,backward->dZ1,forward->X,HiddenLayer1_Size,no_of_images,no_of_images,Input_Size*Input_Size);
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=0;j<Input_Size*Input_Size;j++){
            backward->dW1[i][j]/=no_of_images;
        }
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j=0;j<no_of_images;j++){
            backward->dB1[i]+=backward->dZ1[i][j];
        }
        backward->dB1[i]/=(float)no_of_images;
    }
}
void update_parameter(Forward_Matrices *forward,Backward_Matrices *backward){
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
void free_forward_memory(Forward_Matrices *forward,int no_of_images){
    for(int i=0;i<no_of_images;i++){
        free(forward->X[i]);
    }
    for(int i=0;i<Input_Size*Input_Size;i++){
        free(forward->A0[i]);
    }
    for(int i=0;i<Output_Size;i++){
        free(forward->W2[i]);
        free(forward->A2[i]);
        free(forward->Z2[i]);
        free(forward->labels[i]);
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        free(forward->W1[i]);
        free(forward->A1[i]);
        free(forward->Z1[i]);
    }
    free(forward->X);
    free(forward->Z1);
    free(forward->W1);
    free(forward->A0);
    free(forward->B1);
    free(forward->Z2);
    free(forward->W2);
    free(forward->A1);
    free(forward->B2);
    free(forward->A2);
    free(forward->labels);
}
void free_backward_memory(Backward_Matrices *backward,int no_of_images){
    for(int i=0;i<no_of_images;i++){
        free(backward->A1_T[i]);
    }
    for(int i=0;i<Output_Size;i++){
        free(backward->dZ2[i]);
        free(backward->dW2[i]);
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        free(backward->dZ1[i]);
        free(backward->W2_T[i]);
        free(backward->dW1[i]);
    }
    free(backward->dZ2);
    free(backward->dW2);
    free(backward->A1_T);
    free(backward->dB2);
    free(backward->dZ1);
    free(backward->W2_T);
    free(backward->dW1);
    free(backward->dB1);
}
void Writing_Trained_data(Forward_Matrices *forward){
    //writing data to files
    FILE *weight1=fopen("weight.txt","w");
    FILE *bias1=fopen("bias.txt","w");
    if (!weight1||!bias1) {
        printf("Training Failed , data could not be saved.\n");
        exit(EXIT_FAILURE);
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fprintf(weight1, "%.16f ", forward->W1[i][j]);
        }
    }
    for(int i = 0; i < Output_Size ; i++) {
        for(int j=0;j< HiddenLayer1_Size ;j++){
            fprintf(weight1, "%.16f ", forward->W2[i][j]);
        }
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        fprintf(bias1, "%.16f ", forward->B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fprintf(bias1, "%.16f ", forward->B2[i]);
    }   
    fclose(weight1);
    fclose(bias1);
} 
void reading_dataset(Forward_Matrices *forward,bool Training,bool Inference){
    FILE *image ;
    FILE *label ;
    int no_of_images=0;
    if(Training){
        image = fopen("train-images-idx3-ubyte","rb");
        label = fopen("train-labels-idx1-ubyte","rb");
        no_of_images=training_images;
    }
    else if(Inference){
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
    
    
    for(int currimage=0;currimage<no_of_images;currimage++){ 
        unsigned char curr_label;
        fread(&curr_label,sizeof(unsigned char),1,label);
        forward->labels[curr_label][currimage]=1;
        
        fread(forward->input,sizeof(unsigned char),Input_Size*Input_Size,image);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                forward->X[currimage][i * 28 + j] = forward->input[i * 28 + j]/255.0f;
                forward->A0[i * 28 + j][currimage] = forward->input[i * 28 + j]/255.0f;
            }
        }
        
    }
    free(forward->input);
    fclose(image);
    fclose(label);
}
void read_weigths_bias(Forward_Matrices *forward){
    FILE *weight_=fopen("weight.txt","r");
    FILE *bias_=fopen("bias.txt","r");
    if (!weight_||!bias_) {
        printf("Error opening file.\n");
        exit(EXIT_FAILURE);
    }
    // initilaizing weigths 
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fscanf(weight_, "%f", &forward->W1[i][j]);
        }
    }
    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            fscanf(weight_, "%f", &forward->W2[i][j]);
        }
    }
    
    // initilaizing bias 
    for(int i=0;i<HiddenLayer1_Size;i++){
        fscanf(bias_, "%f", &forward->B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fscanf(bias_, "%f", &forward->B2[i]);
    }
    
    fclose(bias_);
    fclose(weight_);
}
int max(float **matrix,int col){
    float max=matrix[0][col],index=0;
    for(int y=1;y<Output_Size;y++){
        if(matrix[y][col]>max){
            max=matrix[y][col];
            index=y;
        }
    }
    return index;
}
void Inference_Mode(){
    Forward_Matrices forward;
    int no_of_images=inference_images;
    initalize_matrices_forward(&forward,no_of_images);
    read_weigths_bias(&forward);
    reading_dataset(&forward,false,true);
    printf("     ----- Inference Started -----\n");
    forward_proporgation(&forward,no_of_images);
    float Accuracy=0.0;
    for(int x=0;x<no_of_images;x++){
        if(max(forward.labels,x)==max(forward.A2,x)) Accuracy++;
        else printf("     Wrong Prediction:-> \n     Sample Number :%d\n     Actual Value : %d\n     Model Prediction : %d\n\n",x+1,max(forward.labels,x),max(forward.A2,x));
    }
    printf("     ----- Inference Completed! -----\n     Accuracy:%.2f\n",100*Accuracy/no_of_images);
    free_forward_memory(&forward,no_of_images);
}
void Training_Mode(){
    Forward_Matrices forward;
    Backward_Matrices backward;
    int no_of_images=training_images;
    initalize_matrices_forward(&forward,no_of_images);
    initalize_matrices_backward(&backward,no_of_images);
    
    //Generating initalize random bias and weights
    // run this part only one time for clean start from ground up
    
    srand(time(NULL));
    
    FILE *weight=fopen("weight.txt","w");
    FILE *bias=fopen("bias.txt","w");
    
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            float rand_float = ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / (Input_Size * Input_Size));
            fprintf(weight, "%.16f ", rand_float);
        }
        fprintf(weight, "\n");
    }

    for(int i = 0; i < Output_Size ; i++) {
        for(int j=0;j< HiddenLayer1_Size ;j++){
            float rand_float =  ((float)rand() / RAND_MAX - 0.5f) * sqrtf(2.0f / (HiddenLayer1_Size));
            fprintf(weight, "%.16f ", rand_float);
        }
        fprintf(weight, "\n");
    }
    
    for(int i = 0; i < HiddenLayer1_Size ; i++) {
        fprintf(bias, "%.16f\n", 0.0);
    }
    for(int i = 0; i < Output_Size ; i++) {
        fprintf(bias, "%.16f\n", 0.0);
    }
    
    fclose(bias);
    fclose(weight);
    read_weigths_bias(&forward);
    reading_dataset(&forward,true,false);
    printf("     ---- Training Started -----\n");
    for(int Epoch=0;Epoch<=Epochs;Epoch++){
        forward_proporgation(&forward,no_of_images);

        if(Epoch%100==0){
            float Accuracy=0.0;
            float loss = 0.0;
            for(int x=0;x<no_of_images;x++){       
                if(max(forward.labels,x)==max(forward.A2,x)) Accuracy++;
                loss -= logf(forward.A2[max(forward.labels,x)][x] + 1e-8);  //to avoid log(0)
            }        
            printf("     ----Epoch %d/%d----\n     Accuracy:%.2f\n     Loss:%.2f\n",Epoch,Epochs,100*Accuracy/no_of_images,loss/no_of_images);
            printf("\n");
        }  

        backward_proporgation(&forward,&backward,no_of_images);
        update_parameter(&forward,&backward);
        learning_rate *= 0.99f;
    }

    Writing_Trained_data(&forward);
    free_forward_memory(&forward,no_of_images);
    free_backward_memory(&backward,no_of_images);
    printf("     ---- Training Successful! -----");
}

int main() {
    clock_t start,end;
    start = clock();
    int mode=0;
    printf("-----Welcome From Neural Network Made From Scratch In C-----\n");
    printf("     There are two operations modes :\n");
    printf("     1. Training Mode  (Recommended when first starting the network for warm up.)\n");
    printf("     2.Inference Mode  (Use only after Training is successfully completed.)\n");
    printf("     Press 1 for Training to commence , Press 2 for Inference to commence. : ");
    scanf("%d",&mode);
    if(mode==1) {
        Training_Mode();   
    }
    else if(mode==2) {
        Inference_Mode();
    }
    end =clock();
    printf("\ntime used :%f",(double)(end-start)/CLOCKS_PER_SEC);
    exit(EXIT_SUCCESS);
} 