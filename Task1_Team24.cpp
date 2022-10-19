//SPDX-License-Identifier: MIT
//Date: Date: October 1, 2021
//Start code
//CEG4135/CEG4536 Fall 2021
//Author: Dr Mohamed Ali Ibrahim
// This is a starting code to implement Lab1. 
// This is to implement the layer 1 output.


#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

using namespace sycl;
using namespace std;

int main() {
    // Set up queue on any available device
    queue Q;
    /* Code given by the prof that indicates the size of the matrix, but we used the value instead of the variable
    // Matrix X
    constexpr size_t rowX = 1;
    constexpr size_t colX = 2;

    // Matrix W
    constexpr size_t rowW = 2;
    constexpr size_t colW = 3;

    // Matrix C
    constexpr size_t rowC = 1;
    constexpr size_t colC = 3;
    */
    
     
    vector<double>  C(3); //creating a new matrix for the answers from the second computation

    //setting all matrices to prepare for the computation
    vector<double> X = { 1.0, 0.5 };
    vector<double> W = { 0.5, 0.3, 0.5, 0.2, 0.4, 0.6 };
    vector<double> B = { 0.1, 0.2, 0.3 };

    std::fill(C.begin(), C.end(), 0.0); //populating the matrice with zeroes

    {
        // Create buffers associated with inputs and output
        buffer<double, 2> 
            X_buf(X.data(), range<2>(1, 2)), //X Buffer will read the 1D Matrix as a 2D Matrix (1X2)
            B_buf(B.data(), range<2>(1, 3)), //B Buffer will read the 1D Matrix as a 2D Matrix (1X3)
            W_buf(W.data(), range<2>(2, 3)), //W Buffer will read the 1D Matrix as a 2D Matrix (2X3)
            C_buf(C.data(), range<2>(1, 3)); //C Buffer will read the 1D Matrix as a 2D Matrix (1X3)

        // Submit the kernel to the queue
        Q.submit([&](handler& h) {
            //accessors access the buffers
            accessor X{ X_buf, h };
            accessor W{ W_buf, h };
            accessor C{ C_buf, h };
            accessor B{ B_buf, h };

            // START CODE SNIP
            h.parallel_for(range{ 1, 3 }, [=](id<2> idx) {
                //declare our constant indexes for the rows and columns
                int j = idx[0];
                int i = idx[1];
                //for loop to calculate our matrix
                for (int k = 0; k < 2; ++k) {
                    C[j][i] += X[j][k] * W[k][i]; //Matrix multiplication
                }
                C[idx] = C[idx] + B[idx];//Add the bias

                C[idx] = 1 / (1 + exp(-C[idx])); //Applying the sigmoid activation function, 
                                                 //we obtain Z1
                });
            // END CODE SNIP
            });
    }
    //displays the output for task 1
    for (auto i : C)
    {
        cout << i << " ";

    }

    cout << "\nResults for Task 1\n";
}