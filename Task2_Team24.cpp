#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

using namespace sycl;
using namespace std;

int main() {
    
vector<float> A = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
vector<float> B = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
vector<float> C = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
vector<float> D = { 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4 };
vector<float> ANS = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
vector<float> ANSB = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
vector<float> ANSC = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
vector<float> ANSD = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
vector<float> ANSE = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    {
        // Create buffers associated with inputs and output
        buffer<float, 2>
            A_buf(A.data(), range<2>(4, 4)),
            ANS_buf(ANS.data(), range<2>(4, 4)),


        Q.submit([&](handler& h) {
            //accessors access the buffers
            accessor A{ A_buf, h };
            accessor ANS{ ANS_buf, h };

            // START CODE SNIP
            h.parallel_for(range{ 4, 4 }, [=](id<2> idx) {
                //declare our constant indexes for the rows and columns
                int j = idx[0];
                int i = idx[1];
                //for loop to calculate our matrix
                for (int k = 0; k < 2; ++k) {
                    ANS[j][i] += A[j][k] * A[k][i]; //Matrix multiplication
                }
                
                });
            // END CODE SNIP
            });
    }
    
    //displays the output for task 1
    for (auto i : Accumulator A)
    {
        cout << i << " ";

    }

    cout << endl;
    

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
    //displays the output for task 2
    for (auto i : C2)
    {
        cout << i << " ";

    }

    cout << "\nResults for Task 2\n";
}