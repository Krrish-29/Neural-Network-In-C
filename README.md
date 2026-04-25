# 🧠 Neural Network in C/C++ — MNIST From Scratch

A fully from-scratch implementation of a feedforward neural network trained on the MNIST handwritten digits dataset.

This project focuses on understanding **how neural networks actually work under the hood**, by implementing everything manually — from matrix multiplication to backpropagation — without relying on high-level ML frameworks.

---

## 🚀 Highlights

- 📦 End-to-end training and inference pipeline
- 🧮 Manual implementation of:
  - Forward propagation
  - Backpropagation
  - ReLU activation
  - Softmax output layer
- 📊 Mini-batch gradient descent
- ⚡ Optimized matrix operations using **OpenBLAS (CBLAS)**
- 🧵 Parallelism with OpenMP (where applicable)
- 🧠 Achieves **>93% accuracy on MNIST**
- 🔁 Significant improvement over previous version:
  - Training accuracy: **~85% → 93%+**
  - Inference accuracy: **~10–15% → 93%+**

---

## 🧠 What I Learned

This project helped me build a deep understanding of:

- Neural network fundamentals (forward/backward pass)
- Gradient computation and optimization
- Memory layout and cache efficiency
- Performance trade-offs (naive loops vs BLAS)
- Low-level debugging (segfaults, indexing, numerical stability)

---

## 🛠️ Tech Stack

- C / C++
- OpenBLAS (CBLAS)
- OpenMP
- GCC / Clang

---

## ⚙️ Configuration

You can tweak the model in `neural_network.cpp`:

```cpp
float learning_rate = 0.99;
#define Epochs 100
#define training_images 20000
#define batchSize 32
#define HiddenLayer1_Size 32
#define inference_images 10000
```

## 🧪 Results
128 hidden layer -> training - 99.97, inference - 97.99
64 hidden layer -> training - 99.92, inference - 97.62
32 hidden layer -> training - 99.37, inference - 96.63
16 hidden layer -> training - 97.62, inference - 95.82
10 hidden layer -> training - 95.39, inference - 94.28



## Compile
bash
g++ -O3 -march=native -fopenmp neural_network.cpp -lopenblas -o neural_network

Note: use clang for MacOS
## Run
bash
./neural_network

## 📈 Future Improvements
Deeper architectures (multi-layer networks)
SIMD/vectorization optimizations
Better memory layout for cache efficiency
Loss tracking & visualization
GPU acceleration experiments


## 🤝 Contributions

Feel free to fork, experiment, or suggest improvements.
This project is meant for learning and exploration.

## 🌟 Author

Krrish