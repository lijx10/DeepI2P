#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>



__global__ void ball_query_forward_cuda_kernel_shared_mem(const float* __restrict__ node_to_point_dist,
                                                          int* __restrict__ output_points_idx,
                                                          const float radius,
                                                          const int K,
                                                          const int B, const int M, const int N){
    int m = blockIdx.x;
    int b = threadIdx.x;

    extern __shared__ int output_points_unique_number_shared[];
    // initialize shared memory per thread in the block
    output_points_unique_number_shared[b] = 0;

    for(int n=0;n<N;++n){
        int unique_idx = output_points_unique_number_shared[b];
        if(unique_idx < K){
            if(node_to_point_dist[b*M*N+m*N+n] <= radius){
                output_points_idx[b*M*K+m*K+unique_idx] = n;
                output_points_unique_number_shared[b] += 1;
            }
        } else {
            break;
        }
    }

    // fill the un-defined points
    int unique_idx = output_points_unique_number_shared[b];
    if(unique_idx == 0){
        for(int i=0;i<K-unique_idx;++i){
            output_points_idx[b*M*K + m*K + unique_idx+i] = 0;
        }
    }
    else if(unique_idx < K){
        for(int i=0;i<K-unique_idx;++i){
            int idx_repeat = output_points_idx[b*M*K + m*K + i % unique_idx];
            output_points_idx[b*M*K + m*K + unique_idx+i] = idx_repeat;
        }
    }


}



torch::Tensor ball_query_forward_cuda_shared_mem(const torch::Tensor node_to_point_dist,
                                              const float radius,
                                              const int K){
    int B = node_to_point_dist.size(0);
    int M = node_to_point_dist.size(1);
    int N = node_to_point_dist.size(2);

    auto device_idx = node_to_point_dist.device().index();
    torch::TensorOptions options = torch::TensorOptions({torch::kCUDA, device_idx}).dtype(torch::kInt32);
    auto output_points_idx = torch::zeros({B, M, K}, options);

    ball_query_forward_cuda_kernel_shared_mem<<<M, B, B*sizeof(int)>>>(node_to_point_dist.data<float>(),
                                                                         output_points_idx.data<int>(),
                                                                         radius,
                                                                         K,
                                                                         B, M, N);
    return output_points_idx;
}
