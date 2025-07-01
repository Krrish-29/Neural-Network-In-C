#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define Input_Size 28
#define HiddenLayer1_Size 128
// #define HiddenLayer2_Size 10
#define Output_Size 10
#define no_of_images 10
#define Epochs 100
float learning_rate =0.1;
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
int main() {
    //Forward proporgation
    unsigned char *input=(unsigned char*)calloc(Input_Size*Input_Size,sizeof(unsigned char));
    float **X=(float**)calloc(no_of_images,sizeof(float*));

    float **Z1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    float **W1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    float **A0=(float**)calloc(Input_Size*Input_Size,sizeof(float*));
    float *B1=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    
    float **Z2=(float**)calloc(Output_Size,sizeof(float*));
    float **W2=(float**)calloc(Output_Size,sizeof(float*));
    float **A1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    float *B2=(float*)calloc(Output_Size,sizeof(float)); 
    
    float **A2=(float**)calloc(Output_Size,sizeof(float*));
    
    float **labels=(float**)calloc(Output_Size,sizeof(float*));

    for(int i=0;i<no_of_images;i++){
        X[i]=(float*)calloc(Input_Size*Input_Size,sizeof(float));
    }
    for(int i=0;i<Input_Size*Input_Size;i++){
        A0[i]=(float*)calloc(no_of_images,sizeof(float));
    }

    for(int i=0;i<Output_Size;i++){
        W2[i]=(float*)calloc(HiddenLayer1_Size,sizeof(float));
        A2[i]=(float*)calloc(no_of_images,sizeof(float));
        Z2[i]=(float*)calloc(no_of_images,sizeof(float));
        labels[i]=(float*)calloc(no_of_images,sizeof(float));
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        W1[i]=(float*)calloc(Input_Size*Input_Size,sizeof(float));
        A1[i]=(float*)calloc(no_of_images,sizeof(float));
        Z1[i]=(float*)calloc(no_of_images,sizeof(float));
    }

    //backproporgation
    float **dZ2=(float**)calloc(Output_Size,sizeof(float*));
    float **dW2=(float**)calloc(Output_Size,sizeof(float*));
    float **A1_T=(float**)calloc(no_of_images,sizeof(float*));
    float *dB2=(float*)calloc(Output_Size,sizeof(float));

    float **dZ1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    float **W2_T=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    float **dW1=(float**)calloc(HiddenLayer1_Size,sizeof(float*));
    float *dB1=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    
    for(int i=0;i<Output_Size;i++){
        dZ2[i]=(float*)calloc(no_of_images,sizeof(float));
        dW2[i]=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        dZ1[i]=(float*)calloc(no_of_images,sizeof(float));
        W2_T[i]=(float*)calloc(Output_Size,sizeof(float));
        dW1[i]=(float*)calloc(Input_Size*Input_Size,sizeof(float));
    }
    for(int i=0;i<no_of_images;i++){
        A1_T[i]=(float*)calloc(HiddenLayer1_Size,sizeof(float));
    }

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

    FILE *weight_=fopen("weight.txt","r");
    FILE *bias_=fopen("bias.txt","r");
    if (!weight_||!bias_) {
        printf("Error opening file.\n");
        return 1;
    }
    // initilaizing weigths 
    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fscanf(weight_, "%f", &W1[i][j]);
        }
    }
    for(int i=0;i<Output_Size;i++){
        for(int j = 0; j < HiddenLayer1_Size; j++) {
            fscanf(weight_, "%f", &W2[i][j]);
        }
    }
    
    // initilaizing bias 
    for(int i=0;i<HiddenLayer1_Size;i++){
        fscanf(bias_, "%f", &B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fscanf(bias_, "%f", &B2[i]);
    }

    //reading from dataset
    FILE *image = fopen("t10k-images-idx3-ubyte","rb");
    FILE *label = fopen("t10k-labels-idx1-ubyte","rb");
    if (!image||!label) {
        printf("Error opening file.\n");
        return 1;
    }
    fseek(image,16,SEEK_SET);
    fseek(label,8,SEEK_SET);
    
    
    for(int currimage=0;currimage<no_of_images;currimage++){//for all the 10,000 images
        
        unsigned char curr_label;
        fread(&curr_label,sizeof(unsigned char),1,label);
        labels[curr_label][currimage]=1;
        
        fread(input,sizeof(unsigned char),Input_Size*Input_Size,image);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                X[currimage][i * 28 + j] = input[i * 28 + j]/255.0f;
                A0[i * 28 + j][currimage] = input[i * 28 + j]/255.0f;
            }
        }
        
    }
    
    free(input);
    fclose(image);
    fclose(label);
    fclose(bias_);
    fclose(weight_);
    
    for(int Epoch=0;Epoch<Epochs;Epoch++){
        //Forward proporgation
        matrix_multiply(Z1,W1,A0,HiddenLayer1_Size,Input_Size*Input_Size,Input_Size*Input_Size,no_of_images);
        bias_addition(Z1,B1,HiddenLayer1_Size,no_of_images);
        ReLU(A1,Z1,HiddenLayer1_Size,no_of_images);
        matrix_multiply(Z2,W2,A1,Output_Size,HiddenLayer1_Size,HiddenLayer1_Size,no_of_images);
        bias_addition(Z2,B2,Output_Size,no_of_images);
        SoftMax(A2,Z2,Output_Size,no_of_images); 

        for(int x=0;x<no_of_images;x++){
            // for(int i=0;i<28;i++){
            //     for(int j=0;j<28;j++){
            //         printf("%.1f ",transposed_input_layer[i * 28 + j][x] );
            //     }
            //     printf("\n");
            // }
            for(int k=0;k<Output_Size;k++){
                // printf("%d : %f   ",k,labels[k][x]);
                if(labels[k][x]!=0){
                    printf("Actually Value: %d\n",k);
                    printf("Models Guess Probability : %f \n" ,100*A2[k][x]);
                }
            }
            for(int row=0;row<Output_Size;row++){
                printf("%d : %.3f  ",row,A2[row][x]);
            }
            printf("\n");
            printf("\n");
        }   

        // backproporgation
        for(int i=0;i<HiddenLayer1_Size;i++){
            for(int j=0;j<no_of_images;j++){
                A1_T[j][i]=A1[i][j];
            }
        }
        for(int i=0;i<Output_Size;i++){
            for(int j=0;j<HiddenLayer1_Size;j++){
                W2_T[j][i]=W2[i][j];
            }
        }
        matrix_subtraction(dZ2,A2,labels,Output_Size,no_of_images);
        matrix_multiply(dW2,dZ2,A1_T,Output_Size,no_of_images,no_of_images,HiddenLayer1_Size);
        for(int i=0;i<Output_Size;i++){
            for(int j=0;j<HiddenLayer1_Size;j++){
                dW2[i][j]/=no_of_images;
            }
        }
        for(int i=0;i<Output_Size;i++){
            for(int j=0;j<no_of_images;j++){
                dB2[i]+=dZ2[i][j];
            }
            dB2[i]/=(float)no_of_images;
        }
        matrix_multiply(dZ1,W2_T,dZ2,HiddenLayer1_Size,Output_Size,Output_Size,no_of_images);
        ReLU_Derivative(Z1,HiddenLayer1_Size,no_of_images);
        elementwise_matrix_multiply(dZ1,Z1,HiddenLayer1_Size,no_of_images);
        matrix_multiply(dW1,dZ1,X,HiddenLayer1_Size,no_of_images,no_of_images,Input_Size*Input_Size);
        for(int i=0;i<HiddenLayer1_Size;i++){
            for(int j=0;j<Input_Size*Input_Size;j++){
                dW1[i][j]/=no_of_images;
            }
        }
        for(int i=0;i<HiddenLayer1_Size;i++){
            for(int j=0;j<no_of_images;j++){
                dB1[i]+=dZ1[i][j];
            }
            dB1[i]/=(float)no_of_images;
        }

        // updating params
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
        
        learning_rate *= 0.99f;
    }

    //writing data to files
    FILE *weight1=fopen("weight.txt","w");
    FILE *bias1=fopen("bias.txt","w");
    if (!weight1||!bias1) {
        printf("Error opening file.\n");
        return 1;
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        for(int j = 0; j < Input_Size*Input_Size; j++) {
            fprintf(weight1, "%.16f ", W1[i][j]);
        }
    }
    for(int i = 0; i < Output_Size ; i++) {
        for(int j=0;j< HiddenLayer1_Size ;j++){
            fprintf(weight1, "%.16f ", W2[i][j]);
        }
    }

    for(int i=0;i<HiddenLayer1_Size;i++){
        fprintf(bias1, "%.16f ", B1[i]);
    }
    for(int i=0;i<Output_Size;i++){
        fprintf(bias1, "%.16f ", B2[i]);
    }   

    // //freeing memory
    for(int i=0;i<no_of_images;i++){
        free(X[i]);
    }
    for(int i=0;i<Input_Size*Input_Size;i++){
        free(A0[i]);
    }
    for(int i=0;i<Output_Size;i++){
        free(W2[i]);
        free(A2[i]);
        free(Z2[i]);
        free(labels[i]);
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        free(W1[i]);
        free(A1[i]);
        free(Z1[i]);
    }
    free(X);
    free(Z1);
    free(W1);
    free(A0);
    free(B1);
    free(Z2);
    free(W2);
    free(A1);
    free(B2);
    free(A2);
    free(labels);
    for(int i=0;i<no_of_images;i++){
        free(A1_T[i]);
    }
    for(int i=0;i<Output_Size;i++){
        free(dZ2[i]);
        free(dW2[i]);
    }
    for(int i=0;i<HiddenLayer1_Size;i++){
        free(dZ1[i]);
        free(W2_T[i]);
        free(dW1[i]);
    }
    free(dZ2);
    free(dW2);
    free(A1_T);
    free(dB2);
    free(dZ1);
    free(W2_T);
    free(dW1);
    free(dB1);
} 