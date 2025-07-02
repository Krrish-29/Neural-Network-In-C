# Neural-Network-In-C
This is a project where i created a Neural Network from scratch in C
EveryThing is done from scratch execpt a few functions.
Its pure maths and logic.
How to use :
1. git clone  https://github.com:Krrish-29/Neural-Network-In-C.git in a folder
2. Make sure to have gcc installed in your system 
   To verify run : gcc in the terminal u should be seeing something like : gcc: fatal error: no input files compilation terminated.

3. Tweaking parameters: (Optional)
    #define HiddenLayer1_Size 32 //recommended upper limit is 32 , lower limit is 10
    float learning_rate =0.1; // upper limit is 1 , lower limit is 0.01
    #define Epochs 1000 // recommended upper limit is 1000 , lower limit is 1
    #define training_images 1000 //upper limit is 60000 , lower limit is 1
    #define inference_images 1000 //upper limit is 10000 , lower limit is 1
    these parameters in the program can be tweaked lower limit and upper limit are mentioned above. 
    changing the above values higher or lower than mentioned will lead to errors.
    Other than parameter mention should not be changed for error free results. 
4. compile the file : gcc -g -O0 -Wall -fsanitize=address -o image image.c -lm
5. run the exe : ./image
6. Select 1 or training at start for warming up as some important files are created during training procedure. 
7. Then 2 or inference afterwards for analyzing performace.  
