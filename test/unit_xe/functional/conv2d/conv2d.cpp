#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include <CL/sycl.hpp>
#include "inline_pisa.hpp"
#include "validation.hpp"
#include <cutlass/pipeline/xe4_pipeline.hpp>

using namespace sycl;
using namespace cutlass::xe4;
using namespace cute;

class CONV2D_SMALL;
class CONV2D_LARGE;
class CONV2D_SMALL_WITH_PAD_WITH_STRIDE;
class CONV2D_LARGE_WITH_PAD_WITH_STRIDE;
class CONV2D_OTHER_WITH_PAD_WITH_STRIDE;
class CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE;
class CONV2D_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class CONV2D_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION;
class CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION;

template <typename T, uint32_t Dim>
inline uint32_t get_copy_size(const sycl::vec<int32_t, Dim> &coord, const sycl::vec<uint32_t, Dim> &shape, uint32_t width_2d) {
    uint32_t left_size = (shape[0] - coord[0]) * sizeof(T);
    uint32_t copy_size = left_size < width_2d ? left_size : width_2d;
    return copy_size;
}

template<typename test>
int run_test(const conv2d::problem_shape_t &problem_shape)
{
    queue q;
    auto dev = q.get_device();
    std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
    auto ctxt = q.get_context();

    // input:   (N, H, W, C)
    // kernel:  (K, R, S, C)
    // output:  (Out_N, Out_H, Out_W, Out_C)
    uint32_t N = problem_shape.get_in_batch();
    uint32_t H = problem_shape.get_in_height();
    uint32_t W = problem_shape.get_in_width();
    uint32_t C = problem_shape.get_in_channel();
    uint32_t K = problem_shape.get_kernel_num();
    uint32_t R = problem_shape.get_kernel_height();
    uint32_t S = problem_shape.get_kernel_width();
    uint32_t Out_N = problem_shape.get_out_batch();
    uint32_t Out_H = problem_shape.get_out_height();
    uint32_t Out_W = problem_shape.get_out_width();
    uint32_t Out_C = problem_shape.get_out_channel();
    uint32_t padding_top = problem_shape.get_padding_top();
    uint32_t padding_left = problem_shape.get_padding_left();
    uint32_t padding_bottom = problem_shape.get_padding_bottom();
    uint32_t padding_right = problem_shape.get_padding_right();
    uint32_t stride_h = problem_shape.get_stride_h();
    uint32_t stride_w = problem_shape.get_stride_w();
    uint32_t dilation_h = problem_shape.get_dilation_h();
    uint32_t dilation_w = problem_shape.get_dilation_w();

    using dtypeA = bf16;
    using dtypeB = bf16;
    using dtypeAcc = float;
    using dtypeC = bf16;

    constexpr uint32_t dim = 4;
    constexpr uint32_t wg_m = 128;
    constexpr uint32_t wg_n = 128;
    constexpr uint32_t wg_k = 128;
    constexpr uint32_t stage = 3;

    using Pipeline = cutlass::xe4::PipelineTmaAsync<stage>;
    using PipelineStore = cutlass::xe4::PipelineTmaAsync<1, 1>;

    constexpr mem_layout layout_a = mem_layout::row_major;
    constexpr mem_layout layout_b = mem_layout::col_major;
    constexpr bool is_col_major_a = layout_a == mem_layout::col_major;
    constexpr bool is_col_major_b = layout_b == mem_layout::col_major;

    uint32_t sizeA = C * W * H * N;
    uint32_t sizeB = C * S * R * K;
    uint32_t sizeC = Out_C * Out_W * Out_H * Out_N;

    auto A_shared = malloc_shared<dtypeA>(sizeA, q);
    std::generate_n(A_shared, sizeA, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto B_shared = malloc_shared<dtypeB>(sizeB, q);
    std::generate_n(B_shared, sizeB, [=]() { return static_cast<float>(rand()) / static_cast<float>(RAND_MAX); });

    auto C_shared = malloc_shared<dtypeC>(sizeC, q);
    std::fill_n(C_shared, sizeC, dtypeC(0));

    range<3> local_range(1, 1, 64);
    uint32_t mat_m = Out_W * Out_H * Out_N;
    uint32_t mat_n = K;
    uint32_t group_range_m = (mat_m + wg_m - 1) / wg_m;
    uint32_t group_range_n = (mat_n + wg_n - 1) / wg_n;
    range<3> group_range(1, group_range_m, group_range_n);
    std::cout << "Group range: {" << 1 << ", " << group_range_m << ", " << group_range_n << "} \n";
    nd_range<3> Range(group_range * local_range, local_range);

    // matrix info preparation
    constexpr slm_matrix_type cm_typeA = slm_matrix_type::type1;
    constexpr slm_matrix_type cm_typeB = slm_matrix_type::type1;
    constexpr slm_matrix_type cm_typeC = slm_matrix_type::type1;

    constexpr uint32_t width_2dA = wg_k * sizeof(dtypeA);
    constexpr uint32_t width_2dC = wg_n * sizeof(dtypeC);
    static_assert(wg_m % LANESIZE == 0);
    constexpr uint32_t num_inst = wg_m / LANESIZE;
    constexpr uint32_t inst_sizeA = LANESIZE * wg_k;
    constexpr uint32_t inst_sizeC = LANESIZE * wg_n;

    uint32_t repeat_c = (C + wg_k - 1) / wg_k;
    uint32_t kloop = repeat_c * S * R;

    using mat_desc_t = uint32_t;
    using abar_ptr_t = uint64_t*;
    using tdesc_ptr_t = uint64_t*;
    constexpr uint32_t cm_bytes = 1024;

    q.parallel_for<test>(Range, [=](nd_item<3> item) {
        uint32_t local_id = item.get_local_linear_id();
        uint32_t subgroup_id = local_id / 32;

        Pipeline pipeline(local_id);
        abar_ptr_t abar_cons_base = pipeline.abar_cons_base;

        PipelineStore pipeline_store(local_id);

        tdesc_ptr_t tdesc_ptrB = allocate_tdesc<0>();

        auto layoutA = make_layout(make_shape(C, W, H, N));
        auto layoutB = make_layout(make_shape(C, S, R, K));
        auto layoutC = make_layout(make_shape(Out_C, Out_W, Out_H, Out_N));
        auto A = make_tensor(A_shared, layoutA);
        auto B = make_tensor(B_shared, layoutB);
        auto C = make_tensor(C_shared, layoutC);

        auto layoutSA = make_layout(Shape<Int<wg_m>, Int<wg_k>, Int<stage>>{}, Stride<Int<wg_k>, _1, Int<wg_m * wg_k>>{});
        auto layoutSB = make_layout(Shape<Int<wg_n>, Int<wg_k>, Int<stage>>{}, Stride<Int<wg_k>, _1, Int<wg_n * wg_k>>{});
        auto layoutSAcc = make_layout(Shape<Int<wg_m>, Int<wg_n>>{}, Stride<Int<wg_n>, _1>{});
        auto layoutSC = make_layout(Shape<Int<wg_m>, Int<wg_n>>{}, Stride<Int<wg_n>, _1>{});

        constexpr uint32_t total_bytes_a = size(layoutSA) * sizeof(dtypeA);
        constexpr uint32_t total_bytes_b = size(layoutSB) * sizeof(dtypeB);
        constexpr uint32_t total_bytes_c = size(layoutSC) * sizeof(dtypeC);
        constexpr uint32_t total_bytes_acc = size(layoutSC) * sizeof(dtypeAcc);

        constexpr uint32_t slm_bytes_a = total_bytes_a / stage;
        constexpr uint32_t slm_bytes_b = total_bytes_b / stage;
        constexpr uint32_t slm_bytes_c = total_bytes_c;

        constexpr uint32_t slm_bytes = total_bytes_a + total_bytes_b + total_bytes_c + total_bytes_acc;
        auto slm_ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<uint8_t[slm_bytes]>(item.get_group());

        auto sA = make_tensor(reinterpret_cast<dtypeA*>(*slm_ptr), layoutSA);
        auto sB = make_tensor(reinterpret_cast<dtypeB*>(sA.data() + size(layoutSA)), layoutSB);
        auto sAcc = make_tensor(reinterpret_cast<dtypeAcc*>(sB.data() + size(layoutSB)), layoutSAcc);
        auto sC = make_tensor(reinterpret_cast<dtypeC*>(sAcc.data() + size(layoutSAcc)), layoutSC);

        uint32_t wg_id_x = item.get_group(2);
        uint32_t wg_id_y = item.get_group(1);

        int start_m = wg_id_y * wg_m;
        int start_n = wg_id_x * wg_n;

        item.barrier(access::fence_space::local_space);

        if(subgroup_id == 0){
            sycl::vec<uint32_t, dim> gmem_shapeB {shape<0>(B), shape<1>(B), shape<2>(B), shape<3>(B)};
            sycl::vec<uint64_t, dim - 1> gmem_strideB {stride<1>(B) * sizeof(dtypeB),
                stride<2>(B) * sizeof(dtypeB),
                stride<3>(B) * sizeof(dtypeB)};
            sycl::vec<uint32_t, dim> roi_shapeB {shape<1>(layoutSB), 1, 1, shape<0>(layoutSB)};
            sycl::vec<uint32_t, dim> elem_stride {1, 1, 1, 1};

            tensor_desc_fill_global_addr(tdesc_ptrB, B.data());
            tensor_descriptor_fill_dim_size<dim>(tdesc_ptrB, gmem_shapeB);
            tensor_descriptor_fill_dim_stride<dim>(tdesc_ptrB, gmem_strideB);
            tensor_descriptor_fill_traverse_stride<dim>(tdesc_ptrB, elem_stride);
            tensor_descriptor_fill_roitensor_size<dim>(tdesc_ptrB, roi_shapeB);
            tensor_descriptor_fill_misc<dtypeB, cm_typeB>(tdesc_ptrB);

            sycl::vec<uint32_t, dim> gmem_shapeC {shape<0>(C), shape<1>(C), shape<2>(C), shape<3>(C)};
            sycl::vec<uint32_t, dim - 1> gmem_strideC {stride<1>(C) * sizeof(dtypeC),
                stride<2>(C) * sizeof(dtypeC),
                stride<3>(C) * sizeof(dtypeC)};
            sycl::vec<uint32_t, dim> roi_shapeC {shape<1>(layoutSC), shape<1>(C), shape<2>(C), shape<3>(C)};

            sycl::vec<uint32_t, dim> gmem_shapeA {shape<0>(A), shape<1>(A), shape<2>(A), shape<3>(A)};
            sycl::vec<uint32_t, dim - 1> gmem_strideA {stride<1>(A) * sizeof(dtypeA),
                stride<2>(A) * sizeof(dtypeA),
                stride<3>(A) * sizeof(dtypeA)};
            sycl::vec<uint32_t, dim> roi_shapeA {shape<1>(layoutSA), shape<1>(C), shape<2>(C), shape<3>(C)};

            int32_t coord_offset_m_base = start_m + local_id;
            int32_t coord_table[num_inst * (dim - 1)];
            bool oob_table[num_inst];
            #pragma unroll
            for (uint32_t inst_idx = 0; inst_idx < num_inst; inst_idx++) {
                uint32_t index = inst_idx * (dim - 1);
                int32_t offset_m = coord_offset_m_base;

                coord_table[index] = offset_m % roi_shapeA[1];
                offset_m = offset_m / roi_shapeA[1];
                coord_table[index + 1] = offset_m % roi_shapeA[2];
                coord_table[index + 2] = offset_m / roi_shapeA[2];

                oob_table[inst_idx] = coord_table[index + 2] < gmem_shapeA[3];

                coord_offset_m_base += LANESIZE;
            }

            auto slm_pipe_write = cutlass::xe4::make_producer_start_state<Pipeline>();
            auto k_tile_iter = cute::make_coord_iterator(make_shape(repeat_c, S, R));
            auto k_tile_count = size(make_shape(repeat_c, S, R));
            for ( ; k_tile_count > 0; --k_tile_count) {
                uint32_t iter0 = get<0>(*k_tile_iter);
                uint32_t iter1 = get<1>(*k_tile_iter);
                uint32_t iter2 = get<2>(*k_tile_iter);

                uint32_t abar_index = slm_pipe_write.index();
                auto abar_prod = pipeline.producer_get_barrier(abar_index);

                pipeline.producer_try_wait(slm_pipe_write);

                // load input with row_copy
                int32_t gmem_coord_base0 = iter0 * wg_k;
                int32_t gmem_coord_base1 = iter1 * dilation_w;
                int32_t gmem_coord_base2 = iter2 * dilation_h;

                auto tA = sA(_, _, abar_index);
                auto slm_ptr_a = slm_space_cast(tA.data());

                #pragma unroll
                for (uint32_t inst_idx = 0; inst_idx < num_inst; inst_idx++) {
                    auto inst_slm_ptr_a = slm_ptr_a + inst_idx * inst_sizeA;
                    uint32_t index = inst_idx * (dim - 1);
                    auto coord_offset1 = coord_table[index] * stride_w - padding_left;
                    auto coord_offset2 = coord_table[index + 1] * stride_h - padding_top;
                    auto coord_offset3 = coord_table[index + 2];

                    sycl::vec<int32_t, dim> gmem_coord = {gmem_coord_base0,
                        gmem_coord_base1 + coord_offset1,
                        gmem_coord_base2 + coord_offset2,
                        coord_offset3};
                    bool is_coord_valid = oob_table[inst_idx] && (gmem_coord[1] >= 0) && (gmem_coord[1] < gmem_shapeA[1]);
                    is_coord_valid = is_coord_valid && (gmem_coord[2] >= 0) && (gmem_coord[2] < gmem_shapeA[2]);

                    uint32_t offset = gmem_coord[0] * sizeof(dtypeA) + gmem_coord[1] * gmem_strideA[0]
                        + gmem_coord[2] * gmem_strideA[1] + gmem_coord[3] * gmem_strideA[2];
                    offset = is_coord_valid ? offset : 0;

                    uint32_t copy_size = is_coord_valid ? get_copy_size<dtypeA, dim>(gmem_coord, gmem_shapeA, width_2dA) : 0;
                    // uint32_t left_size = (gmem_shapeA[0] - gmem_coord[0]) * sizeof(dtypeA);
                    // uint32_t copy_size = left_size < width_2dA ? left_size : width_2dA;
                    // copy_size = is_coord_valid ? copy_size : 0;

                    async_2d_tiled_load<cm_typeA, width_2dA>(inst_slm_ptr_a, A_shared, offset, copy_size, abar_prod);
                }

                if(local_id == 0) {
                    pipeline.producer_commit(abar_index, slm_bytes_a + slm_bytes_b);

                    // load kernel with tensor_copy
                    sycl::vec<int32_t, dim> gmem_coord = {iter0 * wg_k, iter1, iter2, start_n};
                    auto tB = sB(_, _, abar_index);
                    auto slm_ptr_b = slm_space_cast(tB.data());
                    async_tensor_load<dim>(tdesc_ptrB, slm_ptr_b, gmem_coord, abar_prod);
                }

                ++k_tile_iter;
                ++slm_pipe_write;
            }

            //store out
            PipelineStore::PipelineState slm_pipe_store_cons;
            pipeline_store.consumer_try_wait(slm_pipe_store_cons);

            if(local_id == 0) {
                pipeline_store.consumer_commit(slm_pipe_store_cons, slm_bytes_c);
            }

            auto slm_ptr_c = slm_space_cast(sC.data());

            #pragma unroll
            for (uint32_t inst_idx = 0; inst_idx < num_inst; inst_idx++) {
                auto inst_slm_ptr_c = slm_ptr_c + inst_idx * inst_sizeC;
                uint32_t index = inst_idx * (dim - 1);
                sycl::vec<int32_t, dim> gmem_coord = {start_n,
                    coord_table[index],
                    coord_table[index + 1],
                    coord_table[index + 2]};
                bool is_coord_valid = oob_table[inst_idx] && (gmem_coord[1] >= 0) && (gmem_coord[1] < gmem_shapeC[1]);
                is_coord_valid = is_coord_valid && (gmem_coord[2] >= 0) && (gmem_coord[2] < gmem_shapeC[2]);

                uint32_t offset = gmem_coord[0] * sizeof(dtypeC) + gmem_coord[1] * gmem_strideC[0]
                    + gmem_coord[2] * gmem_strideC[1] + gmem_coord[3] * gmem_strideC[2];
                offset = is_coord_valid ? offset : 0;

                uint32_t copy_size = is_coord_valid ? get_copy_size<dtypeC, dim>(gmem_coord, gmem_shapeC, width_2dC) : 0;
                // uint32_t left_size = (gmem_shapeC[0] - gmem_coord[0]) * sizeof(dtypeC);
                // uint32_t copy_size = left_size < width_2dC ? left_size : width_2dC;
                // copy_size = is_coord_valid ? copy_size : 0;

                uint32_t abar_store_cons_index = slm_pipe_store_cons.index();
                auto abar_store_cons = pipeline_store.producer_get_barrier(abar_store_cons_index);
                async_2d_tiled_store<cm_typeC, width_2dC>(inst_slm_ptr_c, C_shared, offset, copy_size, abar_store_cons);
            }

            ++slm_pipe_store_cons;
            pipeline_store.producer_try_wait(slm_pipe_store_cons);
        }
        else if (subgroup_id == 1){
            if (local_id == 32){
                mat_desc_t mat_desc_a = reinterpret_cast<uint64_t>(slm_space_cast(sA.data())) >> 9;
                mat_desc_t mat_desc_b = reinterpret_cast<uint64_t>(slm_space_cast(sB.data())) >> 9;
                mat_desc_t mat_desc_acc = reinterpret_cast<uint64_t>(slm_space_cast(sAcc.data())) >> 9;
                mat_desc_t mat_desc_c = reinterpret_cast<uint64_t>(slm_space_cast(sC.data())) >> 9;
                constexpr uint32_t cm_size_a_x = is_col_major_a ? 32: 32 / sizeof(dtypeA);
                constexpr uint32_t cm_num_a_x = is_col_major_a ? wg_m / cm_size_a_x : wg_k / cm_size_a_x;
                constexpr uint32_t cm_size_b_x = 32 / sizeof(dtypeB);
                constexpr uint32_t cm_num_b_x = is_col_major_b ? wg_k / cm_size_b_x : wg_n / cm_size_b_x;
                constexpr uint32_t cm_size_c_x = 32 / sizeof(dtypeC);
                constexpr uint32_t cm_num_c_x = wg_n / cm_size_c_x;
                constexpr uint32_t cm_size_acc_x = 32 / sizeof(dtypeAcc);
                constexpr uint32_t cm_num_acc_x = wg_n / cm_size_acc_x;
                constexpr uint32_t cm_stride_a = (cm_bytes * cm_num_a_x) >> 10;
                constexpr uint32_t cm_stride_b = (cm_bytes * cm_num_b_x) >> 10;
                constexpr uint32_t cm_stride_c = (cm_bytes * cm_num_c_x) >> 10;
                constexpr uint32_t cm_stride_acc = (cm_bytes * cm_num_acc_x) >> 10;
                mat_desc_a |= (cm_stride_a << 16);
                mat_desc_b |= (cm_stride_b << 16);
                mat_desc_c |= (cm_stride_c << 16);
                mat_desc_acc |= (cm_stride_acc << 16);

                if (kloop == 1) {
                    Pipeline::PipelineState slm_pipe_read;
                    pipeline.consumer_try_wait(slm_pipe_read);
                    async_gmma<dtypeC, dtypeA, dtypeB, wg_m, wg_n, wg_k, layout_a, layout_b>(mat_desc_c, mat_desc_a, mat_desc_b, abar_cons_base);
                    pipeline.consumer_commit(slm_pipe_read);
                } else {
                    Pipeline::PipelineState slm_pipe_read;
                    pipeline.consumer_try_wait(slm_pipe_read);
                    async_gmma<dtypeAcc, dtypeA, dtypeB, wg_m, wg_n, wg_k, layout_a, layout_b>(
                                mat_desc_acc, mat_desc_a, mat_desc_b, abar_cons_base);
                    pipeline.consumer_commit(slm_pipe_read);

                    static_assert(stage > 1);
                    for (uint32_t i = 1; i < kloop - 1; i++) {
                        ++slm_pipe_read;
                        uint32_t abar_index = slm_pipe_read.index();
                        auto abar_cons = pipeline.consumer_get_barrier(abar_index);
                        auto slm_offset_a = (abar_index * slm_bytes_a) >> 9;
                        auto slm_offset_b = (abar_index * slm_bytes_b) >> 9;
                        pipeline.consumer_try_wait(slm_pipe_read);

                        async_gmma<dtypeAcc, dtypeAcc, dtypeA, dtypeB, wg_m, wg_n, wg_k, layout_a, layout_b>(
                                    mat_desc_acc, mat_desc_acc, mat_desc_a + slm_offset_a, mat_desc_b + slm_offset_b,
                                    abar_cons);
                        pipeline.consumer_commit(slm_pipe_read);
                    }
                    {
                        auto slm_pipe_store_prod = cutlass::xe4::make_producer_start_state<PipelineStore>();
                        uint32_t abar_store_prod_index = slm_pipe_store_prod.index();
                        auto abar_store_prod = pipeline_store.producer_get_barrier(abar_store_prod_index);

                        ++slm_pipe_read;
                        uint32_t abar_index = slm_pipe_read.index();
                        auto slm_offset_a = (abar_index * slm_bytes_a) >> 9;
                        auto slm_offset_b = (abar_index * slm_bytes_b) >> 9;

                        auto abar_cons = pipeline.consumer_get_barrier(abar_index);
                        auto abar_prod = pipeline.producer_get_barrier(abar_index);

                        uint32_t phase = ((kloop - 1) / stage) & 1u;
                        pipeline.consumer_try_wait(abar_index, phase);

                        async_gmma<dtypeC, dtypeAcc, dtypeA, dtypeB, wg_m, wg_n, wg_k, layout_a, layout_b>(
                            mat_desc_c, mat_desc_acc, mat_desc_a + slm_offset_a, mat_desc_b + slm_offset_b, abar_store_prod);
                        pipeline_store.producer_commit(slm_pipe_store_prod, 1);
                    }
                }
            }
        }
    }).wait();

    uint32_t err_cnt = validate_conv2d_result_by_onednn(A_shared, B_shared, C_shared, problem_shape);

    int rtn = 0;
    if (err_cnt > 0)
    {
        std::cout << "Test Failed!" << std::endl;
        rtn = -1;
    }
    else
    {
        std::cout << "Test Pass!" << std::endl;
        rtn = 0;
    }

    return rtn;
}

int main(){
#if defined(TEST_SMALL)
    conv2d::problem_shape_t small {{80, 7, 7, 1}, {80, 3, 3, 80}, {0, 0}, {0, 0}, {1, 1}, {1, 1}};
    run_test<CONV2D_SMALL>(small);
#elif defined(TEST_LARGE)
    conv2d::problem_shape_t large {{160, 16, 16, 2}, {160, 3, 3, 224}, {0, 0}, {0, 0}, {1, 1}, {1, 1}};
    run_test<CONV2D_LARGE>(large);
#elif defined(TEST_SMALL_WITH_PAD_WITH_STRIDE)
    conv2d::problem_shape_t small_with_pad_with_stride {{80, 7, 7, 1}, {80, 3, 3, 80}, {1, 1}, {1, 1}, {2, 2}, {1, 1}};
    run_test<CONV2D_SMALL_WITH_PAD_WITH_STRIDE>(small_with_pad_with_stride);
#elif defined(TEST_LARGE_WITH_PAD_WITH_STRIDE)
    conv2d::problem_shape_t large_with_pad_with_stride {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 1}, {1, 1}, {2, 2}, {1, 1}};
    run_test<CONV2D_LARGE_WITH_PAD_WITH_STRIDE>(large_with_pad_with_stride);
#elif defined(TEST_OTHER_WITH_PAD_WITH_STRIDE)
    conv2d::problem_shape_t other_with_pad_with_stride {{160, 16, 16, 2}, {160, 5, 5, 224}, {1, 1}, {1, 1}, {3, 3}, {1, 1}};
    run_test<CONV2D_OTHER_WITH_PAD_WITH_STRIDE>(other_with_pad_with_stride);
#elif defined(TEST_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE)
    conv2d::problem_shape_t asymmetric_pad_asynmmetric_stride {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 2}, {3, 4}, {2, 3}, {1, 1}};
    run_test<CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE>(asymmetric_pad_asynmmetric_stride);
#elif defined(TEST_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION)
    conv2d::problem_shape_t large_with_pad_with_stride_with_dilation {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 1}, {1, 1}, {2, 2}, {2, 2}};
    run_test<CONV2D_LARGE_WITH_PAD_WITH_STRIDE_WITH_DILATION>(large_with_pad_with_stride_with_dilation);  // generated XeISA is for this casae
#elif defined(TEST_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION)
        conv2d::problem_shape_t other_with_pad_with_stride_with_dilation {{160, 16, 16, 2}, {160, 5, 5, 224}, {1, 1}, {1, 1}, {3, 3}, {3, 3}};
        run_test<CONV2D_OTHER_WITH_PAD_WITH_STRIDE_WITH_DILATION>(other_with_pad_with_stride_with_dilation);
#elif defined(TEST_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION)
        conv2d::problem_shape_t asymmetric_pad_asynmmetric_stride_with_dilation {{160, 16, 16, 2}, {160, 3, 3, 224}, {1, 2}, {3, 4}, {2, 3}, {2, 3}};
        run_test<CONV2D_ASYNMMETRIC_PAD_ASYNMMETRIC_STRIDE_WITH_DILATION>(asymmetric_pad_asynmmetric_stride_with_dilation);
#endif

    return 0;
}
