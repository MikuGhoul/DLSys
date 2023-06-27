#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

float * matrix_mul(const float *a, const float *b, size_t m, size_t n, size_t p) {
    float *c = new float[m*p];
    memset(c, 0, m * p * sizeof(float));
    for (int i = 0; i != m; i++) {
        for (int j = 0; j !=p; j++) {
            for (int k = 0; k != n; k++) {
                c[i*p+j] += a[i*n+k] * b[k*p+j];
            }
        }
    }
    return c;
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int i = 0; i != m/batch; i++) {
        float *batch_x = new float[batch*n];
        float *batch_y = new float[batch];
        memset(batch_x, 0, batch * n * sizeof(float));
        memset(batch_y, 0, batch * sizeof(float));
        for (int j = 0; j != batch; j++) {
            for (int k = 0; k != n; k++) {
                // be careful of `i*batch*n`
                batch_x[j*n+k] = X[i*batch*n+j*n+k];
            }
            batch_y[j] = y[i*batch+j];
        }

        // h = exp(batch_x@theta)
        // h.shape = (batch, k)
        float *h = matrix_mul(batch_x, theta, batch, n, k);
        for (int j = 0; j != batch; j++) {
            for (int l = 0; l != k; l++) {
                h[j*k+l] = float(std::exp(h[j*k+l]));
            }
        }

        // normalization: h/h.sum(1)
        float *Z = new float[batch*k];
        memset(Z, 0, batch * k * sizeof(float));
        for (int j = 0; j != batch; j++) {
            float h_sum = 0.0;
            for (int l = 0; l !=k; l++) {
                h_sum += h[j*k+l];
            }
            for (int l = 0; l !=k; l++) {
                Z[j*k+l] = h[j*k+l] / h_sum;
            }
        }

        // one hot
        float *I = new float[batch*k];
        memset(I, 0, batch * k * sizeof(float));
        for (int j = 0; j != batch; j++) {
            I[j*k+int(batch_y[j])] = 1;
        }

        // Z - I
        float *ZmI = new float[batch*k];
        memset(ZmI, 0, batch * k * sizeof(float));
        for (int j = 0; j != batch; j++) {
            for (int l = 0; l !=k; l++) {
                ZmI[j*k+l] = Z[j*k+l] - I[j*k+l];
            }
        }

        // batch_x.T
        float *batch_x_T = new float[n*batch];
        memset(batch_x_T, 0, n * batch * sizeof(float));
        for (int j = 0; j != n; j++) {
            for (int k = 0; k != batch; k++) {
                batch_x_T[j*batch+k] = batch_x[k*n+j];
            }
        }

        // batch_x.T@(Z-I)
        float *gradient_tmp = matrix_mul(batch_x_T, ZmI, n, batch, k);

        // gradients = batch_x.T@(Z-I) / batch
        float *gradients = new float[n*k];
        memset(gradients, 0, n * k * sizeof(float));
        for (int j = 0; j != n; j++) {
            for (int l = 0; l != k; l++) {
                gradients[j*k+l] = gradient_tmp[j*k+l] / batch;
            }
        }

        // theta -= lr * gradients
        for (int j = 0; j != n; j++) {
            for (int l = 0; l != k; l++) {
                theta[j*k+l] -= lr * gradients[j*k+l];
            }
        }

        delete(batch_x);
        delete(batch_y);
        delete(h);
        delete(Z);
        delete(I);
        delete(ZmI);
        delete(batch_x_T);
        delete(gradient_tmp);
        delete(gradients);
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
