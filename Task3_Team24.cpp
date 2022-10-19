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
    /*
    //displays the output for task 1
    for (auto i : C)
    {
        cout << i << " ";

    }

    cout << "\nResults For Task 1\n";
    */

    //Task 2

    //setting all matrices to prepare for the computation 
    vector<double> W2 = { 0.1, 0.4, 0.2, 0.5, 0.3, 0.6 };
    vector<double> B2 = { 0.1, 0.2 };

    vector<double> Z = C; //setting the value from task 1 to Z
    vector<double> C2(2); //creating a new matrix for the answers from the second computation


    std::fill(C2.begin(), C2.end(), 0.0); //populating the second matrice with zeroes

    {
        // Create buffers associated with inputs and output
        buffer<double, 2>
            B2_buf(B2.data(), range<2>(1, 2)), //B2 Buffer will read the 1D Matrix as a 2D Matrix (1X2)
            W2_buf(W2.data(), range<2>(3, 2)), //W2 Buffer will read the 1D Matrix as a 2D Matrix (3X2)
            Z_buf(Z.data(), range<2>(1, 3)), //Z Buffer will read the 1D Matrix as a 2D Matrix (1X3)
            C2_buf(C2.data(), range<2>(1, 2)); //C2 Buffer will read the 1D Matrix as a 2D Matrix (1X2)

        // Submit the kernel to the queue
        Q.submit([&](handler& h) {
            //accessors access the buffers
            accessor W2{ W2_buf, h };
            accessor Z{ Z_buf, h };
            accessor B2{ B2_buf, h };
            accessor C2{ C2_buf, h };

            // START CODE SNIP
            h.parallel_for(range{ 1, 2 }, [=](id<2> idx) {
                //declare our constant indexes for the rows and columns
                int j = idx[0];
                int i = idx[1];
                //for loop to calculate our matrix
                for (int k = 0; k < 3; ++k) {
                    C2[j][i] += Z[j][k] * W2[k][i]; //Matrix multiplication
                }
                C2[idx] = C2[idx] + B2[idx];//Add with biases

                C2[idx] = 1 / (1 + exp(-C2[idx])); //Applying the sigmoid activation function, 
                                                 //we obtain Z1
                });
            // END CODE SNIP
            });
    }
/*
    //displays the output for task 2
    for (auto i : C2)
    {
        cout << i << " ";

    }

    cout << "\nResults for Task 2\n";
}
*/
    //Task 3
    //setting all matrices to prepare for the computation
    vector<double> W3 = { 0.1, 0.3, 0.2, 0.4 };
    vector<double> B3 = { 0.1, 0.2 };

    vector<double> Z2 = C2; //setting the value from task 2 to Z2
    vector<double> C3(2); //creating a new matrix for the answers from the third computation


    std::fill(C3.begin(), C3.end(), 0.0); //populating the second matrice with zeroes

    {
        // Create buffers associated with inputs and output
        buffer<double, 2>
            B3_buf(B3.data(), range<2>(1, 2)), //B3 Buffer will read the 1D Matrix as a 2D Matrix (1X2)
            W3_buf(W3.data(), range<2>(2, 2)), //W3 Buffer will read the 1D Matrix as a 2D Matrix (2X2)
            Z2_buf(Z2.data(), range<2>(1, 3)), //Z2 Buffer will read the 1D Matrix as a 2D Matrix (1X3)
            C3_buf(C3.data(), range<2>(1, 2)); //C3 Buffer will read the 1D Matrix as a 2D Matrix (1X2)

        // Submit the kernel to the queue
        Q.submit([&](handler& h) {
            //accessors access the buffers
            accessor W3{ W3_buf, h };
            accessor Z2{ Z2_buf, h };
            accessor B3{ B3_buf, h };
            accessor C3{ C3_buf, h };

            // START CODE SNIP
            h.parallel_for(range{ 1, 2 }, [=](id<2> idx) {
                //declare our constant indexes for the rows and columns
                int j = idx[0];
                int i = idx[1];
                //for loop to calculate our matrix
                for (int k = 0; k < 3; ++k) {
                    C3[j][i] += Z2[j][k] * W3[k][i]; //Matrix multiplication
                }
                C3[idx] = C3[idx] + B3[idx];//Add with biases

                C3[idx] = 1 / (1 + exp(-C3[idx])); //Applying the sigmoid activation function, 
                                                 //we obtain Z1
                });
            // END CODE SNIP
            });
    }
    //displays the output for task 3
    for (auto i : C3)
    {
        cout << i << " ";

    }

    cout << "\nResults for Task 3\n";
}