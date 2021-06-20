#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <thread>



// cuda operations ------------------------------
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor/variable")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// declare the functions in .cu file
torch::Tensor ball_query_forward_cuda(const torch::Tensor node_to_point_dist,
                                   const float radius,
                                   const int K);

torch::Tensor ball_query_forward_cuda_shared_mem(const torch::Tensor node_to_point_dist,
                                              const float radius,
                                              const int K);

torch::Tensor ball_query_forward_cuda_wrapper(const torch::Tensor node_to_point_dist,
                                                          const float radius,
                                                          const int K){
    CHECK_INPUT(node_to_point_dist);

    std::cout<<"Not implemented yet."<<std::endl;

//    return ball_query_forward_cuda(node_to_point_dist.data(), radius, K);
}

torch::Tensor ball_query_forward_cuda_wrapper_shared_mem(const torch::Tensor node_to_point_dist,
                                                                    const float radius,
                                                                    const int K){
    CHECK_INPUT(node_to_point_dist);

    return ball_query_forward_cuda_shared_mem(node_to_point_dist, radius, K);
}
// cuda operations ------------------------------




PYBIND11_MODULE(ball_query, m) {
    m.def("forward_cuda", &ball_query_forward_cuda_wrapper, "CUDA code without shared memory");
    m.def("forward_cuda_shared_mem", &ball_query_forward_cuda_wrapper_shared_mem, "CUDA code with shared memory");
}
