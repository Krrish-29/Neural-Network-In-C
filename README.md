# Neural-Network-In-C
🧠 Neural Network in C – MNIST From Scratch
Welcome to my from-scratch implementation of a simple feedforward Neural Network in pure C, trained on the classic MNIST handwritten digits dataset.

This project is built on raw logic and math, without any high-level libraries or AI-generated code. Just me, VS Code, and a lot of hours debugging memory errors deep into the night. 🌙 Along the way, I gained a deep understanding of:

🧩 Neural Network architecture and math

🧠 Activation functions like ReLU and SoftMax

🧠 Gradients and backpropagation

💾 Pointers, arrays, and dynamic memory allocation in C

🚀 Features
Full training and inference pipeline

Reads binary MNIST data files

Implements matrix operations, ReLU, SoftMax, and backpropagation by hand

Configurable architecture with tunable parameters

No third-party AI libraries involved—written completely from scratch

🛠️ How to Use
1. Clone the Repository
bash
git clone https://github.com/Krrish-29/Neural-Network-In-C.git
cd Neural-Network-In-C
2. Verify You Have GCC Installed
bash
gcc
You should see something like:

gcc: fatal error: no input files
compilation terminated.

3. (Optional) Tweak Parameters
Modify the following values in image.c:

c
#define HiddenLayer1_Size 32     // Recommended: 10 to 32
float learning_rate = 0.1;       // Recommended: 0.01 to 1
#define Epochs 1000              // Recommended: 1 to 1000
#define training_images 1000     // Recommended: 1 to 60000
#define inference_images 1000    // Recommended: 1 to 10000
⚠️ Do not exceed upper/lower bounds to avoid runtime errors. Other parts of the code should remain unchanged for stable execution.

4. Compile the Program
bash
gcc -g -O0 -Wall -fsanitize=address -o image image.c -lm
5. Run the Program
bash
./image
6. Train the Network
At the prompt, choose:

1 for Training mode
This creates essential files required for future inference.

7. Run Inference
After training, run the program again and choose:

2  for Inference mode
This lets you test model accuracy and view predictions.

📈 Roadmap
This is just the beginning! I’ll be improving the architecture, experimenting with deeper layers, refining training logic, and exploring visualizations in upcoming iterations.

📂 Dataset
The model uses the MNIST dataset in raw binary format. Make sure the files are in the correct directory:

train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
🤝 Contributions & Feedback
If you’re also diving into low-level machine learning or curious about how neural nets actually work under the hood—feel free to explore, fork, and share thoughts.

🌟 Author
Krrish – Proudly learning by doing. Built this with late-night debugging, real math, and a whole lot of curiosity.
