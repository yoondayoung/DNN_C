#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>

#include <iostream>
#include <vector>
#include <stack>
#include <utility>

#include "oneapi/dnnl/dnnl.h"

#include "example_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
void ariel_enable() { printf("Inside Ariel\n"); }
void* mlm_malloc(size_t size, int level)
{
	if(size == 0)
      {
		// printf("ZERO BYTE MALLOC\n");
		void* bt_entries[64];
		exit(-1);
	}

	printf("Performing a mlm Malloc for size %lu\n", size);

	return malloc(size);
}
#ifdef __cplusplus
}
#endif

#define BATCH 1
#define IC 3
#define CONV_IH 224
#define CONV_IW 224

#define LIN_OH 1
#define LIN_OW 1
#define LIN_OC 1000

const int ndims = 4;

static size_t product(dnnl_dim_t *arr, size_t size) {
    size_t prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= arr[i];
    return prod;
}

static void init_net_data(float *data, uint32_t dim, const dnnl_dim_t *dims) {
    if (dim == 1) {
        for (dnnl_dim_t i = 0; i < dims[0]; ++i) {
            data[i] = (float)(i % 1637);
        }
    } else if (dim == 4) {
        for (dnnl_dim_t in = 0; in < dims[0]; ++in)
            for (dnnl_dim_t ic = 0; ic < dims[1]; ++ic)
                for (dnnl_dim_t ih = 0; ih < dims[2]; ++ih)
                    for (dnnl_dim_t iw = 0; iw < dims[3]; ++iw) {
                        dnnl_dim_t indx = in * dims[1] * dims[2] * dims[3]
                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] = (float)(indx % 1637);
                    }
    }
}

typedef struct {
    int nargs;
    dnnl_exec_arg_t *args;
} args_t;

static void prepare_arg_node(args_t *node, int nargs) {
    node->args = (dnnl_exec_arg_t *)malloc(sizeof(dnnl_exec_arg_t) * nargs);
    node->nargs = nargs;
}
static void free_arg_node(args_t *node) {
    free(node->args);
}

static void set_arg(dnnl_exec_arg_t *arg, int arg_idx, dnnl_memory_t memory) {
    arg->arg = arg_idx;
    arg->memory = memory;
}

static void init_data_memory(uint32_t dim, const dnnl_dim_t *dims,
        dnnl_format_tag_t user_tag, dnnl_engine_t engine, float *data,
        dnnl_memory_t *memory) {
    dnnl_memory_desc_t user_md;
    CHECK(dnnl_memory_desc_create_with_tag(
            &user_md, dim, dims, dnnl_f32, user_tag));
    CHECK(dnnl_memory_create(memory, user_md, engine, DNNL_MEMORY_ALLOCATE));
    CHECK(dnnl_memory_desc_destroy(user_md));
    write_to_dnnl_memory(data, *memory);
}

dnnl_status_t prepare_reorder(dnnl_memory_t *user_memory, // in
        const_dnnl_memory_desc_t prim_memory_md, // in
        dnnl_engine_t prim_engine, // in: primitive's engine
        int dir_is_user_to_prim, // in: user -> prim or prim -> user
        dnnl_memory_t *prim_memory, // out: primitive's memory created
        dnnl_primitive_t *reorder, // out: reorder primitive created
        uint32_t *net_index, // primitive index in net (inc if reorder created)
        dnnl_primitive_t *net, args_t *net_args) { // net params
    const_dnnl_memory_desc_t user_memory_md;
    dnnl_memory_get_memory_desc(*user_memory, &user_memory_md);

    dnnl_engine_t user_mem_engine;
    dnnl_memory_get_engine(*user_memory, &user_mem_engine);

    if (!dnnl_memory_desc_equal(user_memory_md, prim_memory_md)) {
        CHECK(dnnl_memory_create(prim_memory, prim_memory_md, prim_engine,
                DNNL_MEMORY_ALLOCATE));

        dnnl_primitive_desc_t reorder_pd;
        if (dir_is_user_to_prim) {
            CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                    user_memory_md, user_mem_engine, prim_memory_md,
                    prim_engine, NULL));
        } else {
            CHECK(dnnl_reorder_primitive_desc_create(&reorder_pd,
                    prim_memory_md, prim_engine, user_memory_md,
                    user_mem_engine, NULL));
        }
        CHECK(dnnl_primitive_create(reorder, reorder_pd));
        CHECK(dnnl_primitive_desc_destroy(reorder_pd));

        net[*net_index] = *reorder;
        prepare_arg_node(&net_args[*net_index], 2);
        set_arg(&net_args[*net_index].args[0], DNNL_ARG_FROM,
                dir_is_user_to_prim ? *user_memory : *prim_memory);
        set_arg(&net_args[*net_index].args[1], DNNL_ARG_TO,
                dir_is_user_to_prim ? *prim_memory : *user_memory);
        (*net_index)++;
    } else {
        *prim_memory = NULL;
        *reorder = NULL;
    }

    return dnnl_success;
}

void vgg16_net(){
    dnnl_engine_t engine;
    CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));

    // build a net from file
    uint32_t n_fwd = 0;
    dnnl_primitive_t net_fwd[37];
    args_t net_fwd_args[37];

    dnnl_dims_t net_src_sizes = {BATCH, IC, CONV_IH, CONV_IW};
    dnnl_dims_t net_dst_sizes = {BATCH, LIN_OC};

    float *net_src
            = (float *)malloc(product(net_src_sizes, ndims) * sizeof(float));
    float *net_dst
            = (float *)malloc(product(net_dst_sizes, ndims) * sizeof(float));

    init_net_data(net_src, ndims, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, ndims) * sizeof(float));

    //----------------------------------------------------------------------
    //----------------- Forward Stream -------------------------------------
    // VGG16: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    // {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
    // {BATCH, OC, CONV_OH, CONV_OW}
    // strides: {CONV_STRIDE, CONV_STRIDE}, padding: {}, dilation: {0,0}
    dnnl_dims_t conv_user_src_sizes = { BATCH, IC, CONV_IH, CONV_IW };
    dnnl_dims_t conv_user_weights_sizes = {64, IC, 3, 3};
    dnnl_dims_t conv_bias_sizes = {64};
    dnnl_dims_t conv_user_dst_sizes = {BATCH, 64, 224, 224};
    dnnl_dims_t conv_strides = {1, 1};
    dnnl_dims_t conv_dilation = {0, 0};
    dnnl_dims_t conv_padding = {1, 1};

    float *conv_src = net_src;
    float *conv_weights = (float *)malloc(
            product(conv_user_weights_sizes, ndims) * sizeof(float));
    float *conv_bias
            = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));

    init_net_data(conv_weights, ndims, conv_user_weights_sizes);
    init_net_data(conv_bias, 1, conv_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv_user_src_memory, conv_user_weights_memory,
            conv_user_bias_memory;
    init_data_memory(ndims, conv_user_src_sizes, dnnl_nchw, engine, conv_src,
            &conv_user_src_memory);
    init_data_memory(ndims, conv_user_weights_sizes, dnnl_oihw, engine,
            conv_weights, &conv_user_weights_memory);
    init_data_memory(1, conv_bias_sizes, dnnl_x, engine, conv_bias,
            &conv_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv_pd;

    {
        // create data descriptors for convolution w/ no specified format
        dnnl_memory_desc_t conv_src_md, conv_weights_md, conv_bias_md,
                conv_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&conv_src_md, ndims,
                conv_user_src_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&conv_weights_md, ndims,
                conv_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(
                &conv_bias_md, 1, conv_bias_sizes, dnnl_f32, dnnl_x));
        CHECK(dnnl_memory_desc_create_with_tag(&conv_dst_md, ndims,
                conv_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv_pd, engine,
                dnnl_forward, dnnl_convolution_direct, conv_src_md,
                conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
                conv_dilation, conv_padding, conv_padding, NULL));

        CHECK(dnnl_memory_desc_destroy(conv_src_md));
        CHECK(dnnl_memory_desc_destroy(conv_weights_md));
        CHECK(dnnl_memory_desc_destroy(conv_bias_md));
        CHECK(dnnl_memory_desc_destroy(conv_dst_md));
    }

    dnnl_memory_t conv_internal_src_memory, conv_internal_weights_memory,
            conv_internal_dst_memory;

    // create memory for dst data, we don't need to reorder it to user data
    const_dnnl_memory_desc_t conv_dst_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv_internal_dst_memory, conv_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));

    // create reorder primitives between user data and convolution srcs
    // if required
    dnnl_primitive_t conv_reorder_src, conv_reorder_weights;

    const_dnnl_memory_desc_t conv_src_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_src_md, 0);
    CHECK(prepare_reorder(&conv_user_src_memory, conv_src_md, engine, 1,
            &conv_internal_src_memory, &conv_reorder_src, &n_fwd, net_fwd,
            net_fwd_args));

    const_dnnl_memory_desc_t conv_weights_md
            = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv_user_weights_memory, conv_weights_md, engine, 1,
            &conv_internal_weights_memory, &conv_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));

    dnnl_memory_t conv_src_memory = conv_internal_src_memory
            ? conv_internal_src_memory
            : conv_user_src_memory;
    dnnl_memory_t conv_weights_memory = conv_internal_weights_memory
            ? conv_internal_weights_memory
            : conv_user_weights_memory;

    // finally create a convolution primitive
    dnnl_primitive_t conv;
    CHECK(dnnl_primitive_create(&conv, conv_pd));
    net_fwd[n_fwd] = conv;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, conv_src_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv_internal_dst_memory);
    n_fwd++;

    // VGG16: ReLU(inplace=True)
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}

    float negative_slope = 0.0f;

    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const_dnnl_memory_desc_t relu_src_md = conv_dst_md;
    const_dnnl_memory_desc_t relu_dst_md = relu_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu_src_md, relu_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu_dst_memory;
    CHECK(dnnl_memory_create(
            &relu_dst_memory, relu_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu;
    CHECK(dnnl_primitive_create(&relu, relu_pd));
    net_fwd[n_fwd] = relu;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu_dst_memory);
    n_fwd++;
    
    // VGG16: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 224,224/ out:224,224
    // {BATCH, IC, CONV_IH, CONV_IW} (x) {OC, IC, 11, 11} ->
    // {BATCH, OC, CONV_OH, CONV_OW}
    // strides: {CONV_STRIDE, CONV_STRIDE}, padding: {}, dilation: {0,0}
//     dnnl_dims_t conv2_user_src_sizes = { BATCH, 64, 224, 224 };
    dnnl_dims_t conv2_user_weights_sizes = {64, 64, 3, 3};
    dnnl_dims_t conv2_bias_sizes = {64};
    dnnl_dims_t conv2_user_dst_sizes = {BATCH, 64, 224, 224};
    dnnl_dims_t conv2_strides = {1, 1};
    dnnl_dims_t conv2_dilation = {0, 0};
    dnnl_dims_t conv2_padding = {1, 1};

    float *conv2_weights = (float *)mlm_malloc(
                product(conv2_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv2_bias
            = (float *)mlm_malloc(product(conv2_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv2_weights, ndims, conv2_user_weights_sizes);
    init_net_data(conv2_bias, 1, conv2_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv2_user_weights_memory, conv2_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv2_user_weights_sizes, dnnl_oihw, engine,
            conv2_weights, &conv2_user_weights_memory);
    init_data_memory(1, conv2_bias_sizes, dnnl_x, engine, conv2_bias,
            &conv2_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv2_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv2_weights_md, conv2_bias_md, conv2_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv2_weights_md, ndims,
                    conv2_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv2_bias_md, 1, conv2_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv2_dst_md, ndims,
                    conv2_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv2_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu_dst_md,
                    conv2_weights_md, conv2_bias_md, conv2_dst_md, conv2_strides,
                    conv2_dilation, conv2_padding, conv2_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv2_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv2_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv2_dst_md));
    }

    dnnl_memory_t conv2_internal_weights_memory, conv2_internal_dst_memory;
    const_dnnl_memory_desc_t conv2_dst_md
            = dnnl_primitive_desc_query_md(conv2_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv2_internal_dst_memory, conv2_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv2_reorder_weights;
    const_dnnl_memory_desc_t conv2_weights_md
                = dnnl_primitive_desc_query_md(conv2_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv2_user_weights_memory, conv2_weights_md, engine, 1,
            &conv2_internal_weights_memory, &conv2_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv2_weights_memory = conv2_internal_weights_memory
                ? conv2_internal_weights_memory
                : conv2_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv2;
    CHECK(dnnl_primitive_create(&conv2, conv2_pd));
    net_fwd[n_fwd] = conv2;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv2_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv2_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv2_internal_dst_memory);
    n_fwd++;

    // VGG16: ReLU(inplace=True)
    // {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}

    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const_dnnl_memory_desc_t relu2_src_md = conv2_dst_md;
    const_dnnl_memory_desc_t relu2_dst_md = relu2_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu2_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu2_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu2_src_md, relu2_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu2_dst_memory;
    CHECK(dnnl_memory_create(
            &relu2_dst_memory, relu2_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu2;
    CHECK(dnnl_primitive_create(&relu2, relu2_pd));
    net_fwd[n_fwd] = relu2;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv2_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu2_dst_memory);
    n_fwd++;

     // MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) / in: 224,224/ out:112,112
     dnnl_dims_t pool_dst_sizes = {BATCH, 64, 112, 112};
     dnnl_dims_t pool_kernel = {2, 2};
     dnnl_dims_t pool_strides = {2, 2};
     dnnl_dims_t pool_padding = {0, 0};
     dnnl_dims_t pool_dilation = {0, 0};
     
     dnnl_primitive_desc_t pool_pd;
     {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool_dst_md, 4, pool_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool_pd, engine,
                dnnl_forward, dnnl_pooling_max, relu2_dst_md, pool_dst_md,
                pool_strides, pool_kernel, pool_dilation, pool_padding,
                pool_padding, NULL));
        CHECK(dnnl_memory_desc_destroy(pool_dst_md));
     }

        // create memory for workspace
     dnnl_memory_t pool_ws_memory;
     const_dnnl_memory_desc_t pool_ws_md
         = dnnl_primitive_desc_query_md(pool_pd, dnnl_query_workspace_md, 0);
     CHECK(dnnl_memory_create(
                &pool_ws_memory, pool_ws_md, engine, DNNL_MEMORY_ALLOCATE));

     // create reorder primitives between pooling dsts and user format dst
     // if required
     // dnnl_primitive_t pool_reorder_dst;
     dnnl_memory_t pool_dst_memory;
     const_dnnl_memory_desc_t pool_dst_md
        = dnnl_primitive_desc_query_md(pool_pd, dnnl_query_dst_md, 0);
        
     CHECK(dnnl_memory_create(&pool_dst_memory, pool_dst_md, engine, DNNL_MEMORY_ALLOCATE));

     // finally create a pooling primitive
     dnnl_primitive_t pool;
     CHECK(dnnl_primitive_create(&pool, pool_pd));
     net_fwd[n_fwd] = pool;
     prepare_arg_node(&net_fwd_args[n_fwd], 3);
     set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu2_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, pool_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, pool_ws_memory);
     n_fwd++;

    // Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 112,112/ out:112,112
    // 변수명: conv3_....
    dnnl_dims_t conv3_user_weights_sizes = {128, 64, 3, 3};
    dnnl_dims_t conv3_bias_sizes = {128};
    dnnl_dims_t conv3_user_dst_sizes = {BATCH, 128, 112, 112};
    dnnl_dims_t conv3_strides = {1, 1};
    dnnl_dims_t conv3_dilation = {0, 0};
    dnnl_dims_t conv3_padding = {1, 1};

    float *conv3_weights = (float *)mlm_malloc(
                product(conv3_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv3_bias
            = (float *)mlm_malloc(product(conv3_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv3_weights, ndims, conv3_user_weights_sizes);
    init_net_data(conv3_bias, 1, conv3_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv3_user_weights_memory, conv3_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv3_user_weights_sizes, dnnl_oihw, engine,
            conv3_weights, &conv3_user_weights_memory);
    init_data_memory(1, conv3_bias_sizes, dnnl_x, engine, conv3_bias,
            &conv3_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv3_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv3_weights_md, conv3_bias_md, conv3_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv3_weights_md, ndims,
                    conv3_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv3_bias_md, 1, conv3_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv3_dst_md, ndims,
                    conv3_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv3_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, pool_dst_md,
                    conv3_weights_md, conv3_bias_md, conv3_dst_md, conv3_strides,
                    conv3_dilation, conv3_padding, conv3_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv3_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv3_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv3_dst_md));
    }

    dnnl_memory_t conv3_internal_weights_memory, conv3_internal_dst_memory;
    const_dnnl_memory_desc_t conv3_dst_md
            = dnnl_primitive_desc_query_md(conv3_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv3_internal_dst_memory, conv3_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv3_reorder_weights;
    const_dnnl_memory_desc_t conv3_weights_md
                = dnnl_primitive_desc_query_md(conv3_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv3_user_weights_memory, conv3_weights_md, engine, 1,
            &conv3_internal_weights_memory, &conv3_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv3_weights_memory = conv3_internal_weights_memory
                ? conv3_internal_weights_memory
                : conv3_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv3;
    CHECK(dnnl_primitive_create(&conv3, conv3_pd));
    net_fwd[n_fwd] = conv3;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, pool_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv3_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv3_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv3_internal_dst_memory);
    n_fwd++;
    // ReLU(inplace=True)

    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const_dnnl_memory_desc_t relu3_src_md = conv3_dst_md;
    const_dnnl_memory_desc_t relu3_dst_md = relu3_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu3_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu3_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu3_src_md, relu3_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu3_dst_memory;
    CHECK(dnnl_memory_create(
            &relu3_dst_memory, relu3_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu3;
    CHECK(dnnl_primitive_create(&relu3, relu3_pd));
    net_fwd[n_fwd] = relu3;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv3_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu3_dst_memory);
    n_fwd++;


    // Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 112,112/ out:112,112
    dnnl_dims_t conv4_user_weights_sizes = {128, 128, 3, 3};
    dnnl_dims_t conv4_bias_sizes = {128};
    dnnl_dims_t conv4_user_dst_sizes = {BATCH, 128, 112, 112};
    dnnl_dims_t conv4_strides = {1, 1};
    dnnl_dims_t conv4_dilation = {0, 0};
    dnnl_dims_t conv4_padding = {1, 1};

    float *conv4_weights = (float *)mlm_malloc(
                product(conv4_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv4_bias
            = (float *)mlm_malloc(product(conv4_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv4_weights, ndims, conv4_user_weights_sizes);
    init_net_data(conv4_bias, 1, conv4_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv4_user_weights_memory, conv4_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv4_user_weights_sizes, dnnl_oihw, engine,
            conv4_weights, &conv4_user_weights_memory);
    init_data_memory(1, conv4_bias_sizes, dnnl_x, engine, conv4_bias,
            &conv4_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv4_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv4_weights_md, conv4_bias_md, conv4_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv4_weights_md, ndims,
                    conv4_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv4_bias_md, 1, conv4_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv4_dst_md, ndims,
                    conv4_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv4_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu3_dst_md,
                    conv4_weights_md, conv4_bias_md, conv4_dst_md, conv4_strides,
                    conv4_dilation, conv4_padding, conv4_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv4_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv4_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv4_dst_md));
    }

    dnnl_memory_t conv4_internal_weights_memory, conv4_internal_dst_memory;
    const_dnnl_memory_desc_t conv4_dst_md
            = dnnl_primitive_desc_query_md(conv4_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv4_internal_dst_memory, conv4_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv4_reorder_weights;
    const_dnnl_memory_desc_t conv4_weights_md
                = dnnl_primitive_desc_query_md(conv4_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv4_user_weights_memory, conv4_weights_md, engine, 1,
            &conv4_internal_weights_memory, &conv4_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv4_weights_memory = conv4_internal_weights_memory
                ? conv4_internal_weights_memory
                : conv4_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv4;
    CHECK(dnnl_primitive_create(&conv4, conv4_pd));
    net_fwd[n_fwd] = conv4;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu3_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv4_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv4_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv4_internal_dst_memory);
    n_fwd++;
    // ReLU(inplace=True)

    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const_dnnl_memory_desc_t relu4_src_md = conv4_dst_md;
    const_dnnl_memory_desc_t relu4_dst_md = relu4_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu4_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu4_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu4_src_md, relu4_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu4_dst_memory;
    CHECK(dnnl_memory_create(
            &relu4_dst_memory, relu4_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu4;
    CHECK(dnnl_primitive_create(&relu4, relu4_pd));
    net_fwd[n_fwd] = relu4;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv4_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu4_dst_memory);
    n_fwd++;

    // MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) / in: 112,112/ out:56, 56
     dnnl_dims_t pool2_dst_sizes = {BATCH, 128, 56, 56};
     dnnl_dims_t pool2_kernel = {2, 2};
     dnnl_dims_t pool2_strides = {2, 2};
     dnnl_dims_t pool2_padding = {0, 0};
     dnnl_dims_t pool2_dilation = {0, 0};
   
     dnnl_primitive_desc_t pool2_pd;
     {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool2_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool2_dst_md, 4, pool2_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool2_pd, engine,
                dnnl_forward, dnnl_pooling_max, relu4_dst_md, pool2_dst_md,
                pool2_strides, pool2_kernel, pool2_dilation, pool2_padding,
                pool2_padding, NULL));
        CHECK(dnnl_memory_desc_destroy(pool2_dst_md));
     }

        // create memory for workspace
     dnnl_memory_t pool2_ws_memory;
     const_dnnl_memory_desc_t pool2_ws_md
         = dnnl_primitive_desc_query_md(pool2_pd, dnnl_query_workspace_md, 0);
     CHECK(dnnl_memory_create(
                &pool2_ws_memory, pool2_ws_md, engine, DNNL_MEMORY_ALLOCATE));

     // create reorder primitives between pooling dsts and user format dst
     // if required
     // dnnl_primitive_t pool_reorder_dst;
     dnnl_memory_t pool2_dst_memory;
     const_dnnl_memory_desc_t pool2_dst_md
        = dnnl_primitive_desc_query_md(pool2_pd, dnnl_query_dst_md, 0);
        
     CHECK(dnnl_memory_create(&pool2_dst_memory, pool2_dst_md, engine, DNNL_MEMORY_ALLOCATE));

     // finally create a pooling primitive
     dnnl_primitive_t pool2;
     CHECK(dnnl_primitive_create(&pool2, pool2_pd));
     net_fwd[n_fwd] = pool2;
     prepare_arg_node(&net_fwd_args[n_fwd], 3);
     set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu4_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, pool2_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, pool2_ws_memory);
     n_fwd++;

    // Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 56,56/ out:56, 56
    dnnl_dims_t conv5_user_weights_sizes = {256, 128, 3, 3};
    dnnl_dims_t conv5_bias_sizes = {256};
    dnnl_dims_t conv5_user_dst_sizes = {BATCH, 256, 56, 56};
    dnnl_dims_t conv5_strides = {1, 1};
    dnnl_dims_t conv5_dilation = {0, 0};
    dnnl_dims_t conv5_padding = {1, 1};

    float *conv5_weights = (float *)mlm_malloc(
                product(conv5_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv5_bias
            = (float *)mlm_malloc(product(conv5_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv5_weights, ndims, conv5_user_weights_sizes);
    init_net_data(conv5_bias, 1, conv5_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv5_user_weights_memory, conv5_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv5_user_weights_sizes, dnnl_oihw, engine,
            conv5_weights, &conv5_user_weights_memory);
    init_data_memory(1, conv5_bias_sizes, dnnl_x, engine, conv5_bias,
            &conv5_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv5_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv5_weights_md, conv5_bias_md, conv5_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv5_weights_md, ndims,
                    conv5_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv5_bias_md, 1, conv5_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv5_dst_md, ndims,
                    conv5_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv5_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, pool2_dst_md,
                    conv5_weights_md, conv5_bias_md, conv5_dst_md, conv5_strides,
                    conv5_dilation, conv5_padding, conv5_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv5_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv5_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv5_dst_md));
    }

    dnnl_memory_t conv5_internal_weights_memory, conv5_internal_dst_memory;
    const_dnnl_memory_desc_t conv5_dst_md
            = dnnl_primitive_desc_query_md(conv5_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv5_internal_dst_memory, conv5_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv5_reorder_weights;
    const_dnnl_memory_desc_t conv5_weights_md
                = dnnl_primitive_desc_query_md(conv5_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv5_user_weights_memory, conv5_weights_md, engine, 1,
            &conv5_internal_weights_memory, &conv5_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv5_weights_memory = conv5_internal_weights_memory
                ? conv5_internal_weights_memory
                : conv5_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv5;
    CHECK(dnnl_primitive_create(&conv5, conv5_pd));
    net_fwd[n_fwd] = conv5;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, pool2_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv5_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv5_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv5_internal_dst_memory);
    n_fwd++;
    // ReLU(inplace=True)
    
    // keep memory format of source same as the format of convolution
    // output in order to avoid reorder
    const_dnnl_memory_desc_t relu5_src_md = conv5_dst_md;
    const_dnnl_memory_desc_t relu5_dst_md = relu5_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu5_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu5_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu5_src_md, relu5_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu5_dst_memory;
    CHECK(dnnl_memory_create(
            &relu5_dst_memory, relu5_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu5;
    CHECK(dnnl_primitive_create(&relu5, relu5_pd));
    net_fwd[n_fwd] = relu5;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv5_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu5_dst_memory);
    n_fwd++;

    // Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 56,56/ out:56, 56
    dnnl_dims_t conv6_user_weights_sizes = {256, 256, 3, 3};
    dnnl_dims_t conv6_bias_sizes = {256};
    dnnl_dims_t conv6_user_dst_sizes = {BATCH, 256, 56, 56};
    dnnl_dims_t conv6_strides = {1, 1};
    dnnl_dims_t conv6_dilation = {0, 0};
    dnnl_dims_t conv6_padding = {1, 1};

    float *conv6_weights = (float *)mlm_malloc(
                product(conv6_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv6_bias
            = (float *)mlm_malloc(product(conv6_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv6_weights, ndims, conv6_user_weights_sizes);
    init_net_data(conv6_bias, 1, conv6_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv6_user_weights_memory, conv6_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv6_user_weights_sizes, dnnl_oihw, engine,
            conv6_weights, &conv6_user_weights_memory);
    init_data_memory(1, conv6_bias_sizes, dnnl_x, engine, conv6_bias,
            &conv6_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv6_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv6_weights_md, conv6_bias_md, conv6_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv6_weights_md, ndims,
                    conv6_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv6_bias_md, 1, conv6_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv6_dst_md, ndims,
                    conv6_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv6_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu5_dst_md,
                    conv6_weights_md, conv6_bias_md, conv6_dst_md, conv6_strides,
                    conv6_dilation, conv6_padding, conv6_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv6_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv6_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv6_dst_md));
    }

    dnnl_memory_t conv6_internal_weights_memory, conv6_internal_dst_memory;
    const_dnnl_memory_desc_t conv6_dst_md
            = dnnl_primitive_desc_query_md(conv6_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv6_internal_dst_memory, conv6_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv6_reorder_weights;
    const_dnnl_memory_desc_t conv6_weights_md
                = dnnl_primitive_desc_query_md(conv6_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv6_user_weights_memory, conv6_weights_md, engine, 1,
            &conv6_internal_weights_memory, &conv6_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv6_weights_memory = conv6_internal_weights_memory
                ? conv6_internal_weights_memory
                : conv6_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv6;
    CHECK(dnnl_primitive_create(&conv6, conv6_pd));
    net_fwd[n_fwd] = conv6;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu5_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv6_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv6_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv6_internal_dst_memory);
    n_fwd++;
    // ReLU(inplace=True)

    const_dnnl_memory_desc_t relu6_src_md = conv6_dst_md;
    const_dnnl_memory_desc_t relu6_dst_md = relu6_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu6_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu6_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu6_src_md, relu6_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu6_dst_memory;
    CHECK(dnnl_memory_create(
            &relu6_dst_memory, relu6_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu6;
    CHECK(dnnl_primitive_create(&relu6, relu6_pd));
    net_fwd[n_fwd] = relu6;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv6_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu6_dst_memory);
    n_fwd++;

    
    // Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 56,56/ out:56, 56
    dnnl_dims_t conv7_user_weights_sizes = {256, 256, 3, 3};
    dnnl_dims_t conv7_bias_sizes = {256};
    dnnl_dims_t conv7_user_dst_sizes = {BATCH, 256, 56, 56};
    dnnl_dims_t conv7_strides = {1, 1};
    dnnl_dims_t conv7_dilation = {0, 0};
    dnnl_dims_t conv7_padding = {1, 1};

    float *conv7_weights = (float *)mlm_malloc(
                product(conv7_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv7_bias
            = (float *)mlm_malloc(product(conv7_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv7_weights, ndims, conv7_user_weights_sizes);
    init_net_data(conv7_bias, 1, conv7_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv7_user_weights_memory, conv7_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv7_user_weights_sizes, dnnl_oihw, engine,
            conv7_weights, &conv7_user_weights_memory);
    init_data_memory(1, conv7_bias_sizes, dnnl_x, engine, conv7_bias,
            &conv7_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv7_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv7_weights_md, conv7_bias_md, conv7_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv7_weights_md, ndims,
                    conv7_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv7_bias_md, 1, conv7_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv7_dst_md, ndims,
                    conv7_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv7_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu6_dst_md,
                    conv7_weights_md, conv7_bias_md, conv7_dst_md, conv7_strides,
                    conv7_dilation, conv7_padding, conv7_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv7_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv7_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv7_dst_md));
    }

    dnnl_memory_t conv7_internal_weights_memory, conv7_internal_dst_memory;
    const_dnnl_memory_desc_t conv7_dst_md
            = dnnl_primitive_desc_query_md(conv7_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv7_internal_dst_memory, conv7_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv7_reorder_weights;
    const_dnnl_memory_desc_t conv7_weights_md
                = dnnl_primitive_desc_query_md(conv7_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv7_user_weights_memory, conv7_weights_md, engine, 1,
            &conv7_internal_weights_memory, &conv7_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv7_weights_memory = conv7_internal_weights_memory
                ? conv7_internal_weights_memory
                : conv7_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv7;
    CHECK(dnnl_primitive_create(&conv7, conv7_pd));
    net_fwd[n_fwd] = conv7;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu6_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv7_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv7_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv7_internal_dst_memory);
    n_fwd++;
    
    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu7_src_md = conv7_dst_md;
    const_dnnl_memory_desc_t relu7_dst_md = relu7_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu7_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu7_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu7_src_md, relu7_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu7_dst_memory;
    CHECK(dnnl_memory_create(
            &relu7_dst_memory, relu7_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu7;
    CHECK(dnnl_primitive_create(&relu7, relu7_pd));
    net_fwd[n_fwd] = relu7;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv7_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu7_dst_memory);
    n_fwd++;
    
    // MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) / in: 56,56/ out:28, 28
     dnnl_dims_t pool3_dst_sizes = {BATCH, 256, 28, 28};
     dnnl_dims_t pool3_kernel = {2, 2};
     dnnl_dims_t pool3_strides = {2, 2};
     dnnl_dims_t pool3_padding = {0, 0};
     dnnl_dims_t pool3_dilation = {0, 0};
   
     dnnl_primitive_desc_t pool3_pd;
     {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool3_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool3_dst_md, 4, pool3_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool3_pd, engine,
                dnnl_forward, dnnl_pooling_max, relu7_dst_md, pool3_dst_md,
                pool3_strides, pool3_kernel, pool3_dilation, pool3_padding,
                pool3_padding, NULL));
        CHECK(dnnl_memory_desc_destroy(pool3_dst_md));
     }

        // create memory for workspace
     dnnl_memory_t pool3_ws_memory;
     const_dnnl_memory_desc_t pool3_ws_md
         = dnnl_primitive_desc_query_md(pool3_pd, dnnl_query_workspace_md, 0);
     CHECK(dnnl_memory_create(
                &pool3_ws_memory, pool3_ws_md, engine, DNNL_MEMORY_ALLOCATE));

     // create reorder primitives between pooling dsts and user format dst
     // if required
     // dnnl_primitive_t pool_reorder_dst;
     dnnl_memory_t pool3_dst_memory;
     const_dnnl_memory_desc_t pool3_dst_md
        = dnnl_primitive_desc_query_md(pool3_pd, dnnl_query_dst_md, 0);
        
     CHECK(dnnl_memory_create(&pool3_dst_memory, pool3_dst_md, engine, DNNL_MEMORY_ALLOCATE));

     // finally create a pooling primitive
     dnnl_primitive_t pool3;
     CHECK(dnnl_primitive_create(&pool3, pool3_pd));
     net_fwd[n_fwd] = pool3;
     prepare_arg_node(&net_fwd_args[n_fwd], 3);
     set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu7_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, pool3_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, pool3_ws_memory);
     n_fwd++;
   
    // Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 28, 28/ out:28, 28
    dnnl_dims_t conv8_user_weights_sizes = {512, 256, 3, 3};
    dnnl_dims_t conv8_bias_sizes = {512};
    dnnl_dims_t conv8_user_dst_sizes = {BATCH, 512, 28, 28};
    dnnl_dims_t conv8_strides = {1, 1};
    dnnl_dims_t conv8_dilation = {0, 0};
    dnnl_dims_t conv8_padding = {1, 1};

    float *conv8_weights = (float *)mlm_malloc(
                product(conv8_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv8_bias
            = (float *)mlm_malloc(product(conv8_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv8_weights, ndims, conv8_user_weights_sizes);
    init_net_data(conv8_bias, 1, conv8_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv8_user_weights_memory, conv8_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv8_user_weights_sizes, dnnl_oihw, engine,
            conv8_weights, &conv8_user_weights_memory);
    init_data_memory(1, conv8_bias_sizes, dnnl_x, engine, conv8_bias,
            &conv8_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv8_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv8_weights_md, conv8_bias_md, conv8_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv8_weights_md, ndims,
                    conv8_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv8_bias_md, 1, conv8_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv8_dst_md, ndims,
                    conv8_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv8_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, pool3_dst_md,
                    conv8_weights_md, conv8_bias_md, conv8_dst_md, conv8_strides,
                    conv8_dilation, conv8_padding, conv8_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv8_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv8_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv8_dst_md));
    }

    dnnl_memory_t conv8_internal_weights_memory, conv8_internal_dst_memory;
    const_dnnl_memory_desc_t conv8_dst_md
            = dnnl_primitive_desc_query_md(conv8_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv8_internal_dst_memory, conv8_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv8_reorder_weights;
    const_dnnl_memory_desc_t conv8_weights_md
                = dnnl_primitive_desc_query_md(conv8_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv8_user_weights_memory, conv8_weights_md, engine, 1,
            &conv8_internal_weights_memory, &conv8_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv8_weights_memory = conv8_internal_weights_memory
                ? conv8_internal_weights_memory
                : conv8_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv8;
    CHECK(dnnl_primitive_create(&conv8, conv8_pd));
    net_fwd[n_fwd] = conv8;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, pool3_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv8_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv8_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv8_internal_dst_memory);
    n_fwd++;
    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu8_src_md = conv8_dst_md;
    const_dnnl_memory_desc_t relu8_dst_md = relu8_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu8_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu8_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu8_src_md, relu8_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu8_dst_memory;
    CHECK(dnnl_memory_create(
            &relu8_dst_memory, relu8_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu8;
    CHECK(dnnl_primitive_create(&relu8, relu8_pd));
    net_fwd[n_fwd] = relu8;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv8_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu8_dst_memory);
    n_fwd++;

    // Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 28, 28/ out:28, 28
    dnnl_dims_t conv9_user_weights_sizes = {512, 512, 3, 3};
    dnnl_dims_t conv9_bias_sizes = {512};
    dnnl_dims_t conv9_user_dst_sizes = {BATCH, 512, 28, 28};
    dnnl_dims_t conv9_strides = {1, 1};
    dnnl_dims_t conv9_dilation = {0, 0};
    dnnl_dims_t conv9_padding = {1, 1};

    float *conv9_weights = (float *)mlm_malloc(
                product(conv9_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv9_bias
            = (float *)mlm_malloc(product(conv9_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv9_weights, ndims, conv9_user_weights_sizes);
    init_net_data(conv9_bias, 1, conv9_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv9_user_weights_memory, conv9_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv9_user_weights_sizes, dnnl_oihw, engine,
            conv9_weights, &conv9_user_weights_memory);
    init_data_memory(1, conv9_bias_sizes, dnnl_x, engine, conv9_bias,
            &conv9_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv9_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv9_weights_md, conv9_bias_md, conv9_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv9_weights_md, ndims,
                    conv9_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv9_bias_md, 1, conv9_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv9_dst_md, ndims,
                    conv9_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv9_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu8_dst_md,
                    conv9_weights_md, conv9_bias_md, conv9_dst_md, conv9_strides,
                    conv9_dilation, conv9_padding, conv9_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv9_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv9_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv9_dst_md));
    }

    dnnl_memory_t conv9_internal_weights_memory, conv9_internal_dst_memory;
    const_dnnl_memory_desc_t conv9_dst_md
            = dnnl_primitive_desc_query_md(conv9_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv9_internal_dst_memory, conv9_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv9_reorder_weights;
    const_dnnl_memory_desc_t conv9_weights_md
                = dnnl_primitive_desc_query_md(conv9_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv9_user_weights_memory, conv9_weights_md, engine, 1,
            &conv9_internal_weights_memory, &conv9_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv9_weights_memory = conv9_internal_weights_memory
                ? conv9_internal_weights_memory
                : conv9_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv9;
    CHECK(dnnl_primitive_create(&conv9, conv9_pd));
    net_fwd[n_fwd] = conv9;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu8_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv9_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv9_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv9_internal_dst_memory);
    n_fwd++;
    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu9_src_md = conv9_dst_md;
    const_dnnl_memory_desc_t relu9_dst_md = relu9_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu9_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu9_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu9_src_md, relu9_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu9_dst_memory;
    CHECK(dnnl_memory_create(
            &relu9_dst_memory, relu9_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu9;
    CHECK(dnnl_primitive_create(&relu9, relu9_pd));
    net_fwd[n_fwd] = relu9;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv9_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu9_dst_memory);
    n_fwd++;
    
    // Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 28, 28/ out:28, 28
    dnnl_dims_t conv10_user_weights_sizes = {512, 512, 3, 3};
    dnnl_dims_t conv10_bias_sizes = {512};
    dnnl_dims_t conv10_user_dst_sizes = {BATCH, 512, 28, 28};
    dnnl_dims_t conv10_strides = {1, 1};
    dnnl_dims_t conv10_dilation = {0, 0};
    dnnl_dims_t conv10_padding = {1, 1};

    float *conv10_weights = (float *)mlm_malloc(
                product(conv10_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv10_bias
            = (float *)mlm_malloc(product(conv10_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv10_weights, ndims, conv10_user_weights_sizes);
    init_net_data(conv10_bias, 1, conv10_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv10_user_weights_memory, conv10_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv10_user_weights_sizes, dnnl_oihw, engine,
            conv10_weights, &conv10_user_weights_memory);
    init_data_memory(1, conv10_bias_sizes, dnnl_x, engine, conv10_bias,
            &conv10_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv10_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv10_weights_md, conv10_bias_md, conv10_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv10_weights_md, ndims,
                    conv10_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv10_bias_md, 1, conv10_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv10_dst_md, ndims,
                    conv10_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv10_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu9_dst_md,
                    conv10_weights_md, conv10_bias_md, conv10_dst_md, conv10_strides,
                    conv10_dilation, conv10_padding, conv10_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv10_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv10_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv10_dst_md));
    }

    dnnl_memory_t conv10_internal_weights_memory, conv10_internal_dst_memory;
    const_dnnl_memory_desc_t conv10_dst_md
            = dnnl_primitive_desc_query_md(conv10_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv10_internal_dst_memory, conv10_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv10_reorder_weights;
    const_dnnl_memory_desc_t conv10_weights_md
                = dnnl_primitive_desc_query_md(conv10_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv10_user_weights_memory, conv10_weights_md, engine, 1,
            &conv10_internal_weights_memory, &conv10_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv10_weights_memory = conv10_internal_weights_memory
                ? conv10_internal_weights_memory
                : conv10_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv10;
    CHECK(dnnl_primitive_create(&conv10, conv10_pd));
    net_fwd[n_fwd] = conv10;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu9_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv10_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv10_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv10_internal_dst_memory);
    n_fwd++;

    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu10_src_md = conv10_dst_md;
    const_dnnl_memory_desc_t relu10_dst_md = relu10_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu10_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu10_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu10_src_md, relu10_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu10_dst_memory;
    CHECK(dnnl_memory_create(
            &relu10_dst_memory, relu10_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu10;
    CHECK(dnnl_primitive_create(&relu10, relu10_pd));
    net_fwd[n_fwd] = relu10;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv10_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu10_dst_memory);
    n_fwd++;
    
    // MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) / in: 28, 28/ out:14, 14
     dnnl_dims_t pool4_dst_sizes = {BATCH, 512, 14, 14};
     dnnl_dims_t pool4_kernel = {2, 2};
     dnnl_dims_t pool4_strides = {2, 2};
     dnnl_dims_t pool4_padding = {0, 0};
     dnnl_dims_t pool4_dilation = {0, 0};
   
     dnnl_primitive_desc_t pool4_pd;
     {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool4_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool4_dst_md, 4, pool4_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool4_pd, engine,
                dnnl_forward, dnnl_pooling_max, relu10_dst_md, pool4_dst_md,
                pool4_strides, pool4_kernel, pool4_dilation, pool4_padding,
                pool4_padding, NULL));
        CHECK(dnnl_memory_desc_destroy(pool4_dst_md));
     }

        // create memory for workspace
     dnnl_memory_t pool4_ws_memory;
     const_dnnl_memory_desc_t pool4_ws_md
         = dnnl_primitive_desc_query_md(pool4_pd, dnnl_query_workspace_md, 0);
     CHECK(dnnl_memory_create(
                &pool4_ws_memory, pool4_ws_md, engine, DNNL_MEMORY_ALLOCATE));

     // create reorder primitives between pooling dsts and user format dst
     // if required
     // dnnl_primitive_t pool_reorder_dst;
     dnnl_memory_t pool4_dst_memory;
     const_dnnl_memory_desc_t pool4_dst_md
        = dnnl_primitive_desc_query_md(pool4_pd, dnnl_query_dst_md, 0);
        
     CHECK(dnnl_memory_create(&pool4_dst_memory, pool4_dst_md, engine, DNNL_MEMORY_ALLOCATE));

     // finally create a pooling primitive
     dnnl_primitive_t pool4;
     CHECK(dnnl_primitive_create(&pool4, pool4_pd));
     net_fwd[n_fwd] = pool4;
     prepare_arg_node(&net_fwd_args[n_fwd], 3);
     set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu10_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, pool4_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, pool4_ws_memory);
     n_fwd++;

    // Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 14, 14/ out:14, 14
    dnnl_dims_t conv11_user_weights_sizes = {512, 512, 3, 3};
    dnnl_dims_t conv11_bias_sizes = {512};
    dnnl_dims_t conv11_user_dst_sizes = {BATCH, 512, 14, 14};
    dnnl_dims_t conv11_strides = {1, 1};
    dnnl_dims_t conv11_dilation = {0, 0};
    dnnl_dims_t conv11_padding = {1, 1};

    float *conv11_weights = (float *)mlm_malloc(
                product(conv11_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv11_bias
            = (float *)mlm_malloc(product(conv11_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv11_weights, ndims, conv11_user_weights_sizes);
    init_net_data(conv11_bias, 1, conv11_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv11_user_weights_memory, conv11_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv11_user_weights_sizes, dnnl_oihw, engine,
            conv11_weights, &conv11_user_weights_memory);
    init_data_memory(1, conv11_bias_sizes, dnnl_x, engine, conv11_bias,
            &conv11_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv11_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv11_weights_md, conv11_bias_md, conv11_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv11_weights_md, ndims,
                    conv11_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv11_bias_md, 1, conv11_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv11_dst_md, ndims,
                    conv11_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv11_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, pool4_dst_md,
                    conv11_weights_md, conv11_bias_md, conv11_dst_md, conv11_strides,
                    conv11_dilation, conv11_padding, conv11_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv11_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv11_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv11_dst_md));
    }

    dnnl_memory_t conv11_internal_weights_memory, conv11_internal_dst_memory;
    const_dnnl_memory_desc_t conv11_dst_md
            = dnnl_primitive_desc_query_md(conv11_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv11_internal_dst_memory, conv11_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv11_reorder_weights;
    const_dnnl_memory_desc_t conv11_weights_md
                = dnnl_primitive_desc_query_md(conv11_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv11_user_weights_memory, conv11_weights_md, engine, 1,
            &conv11_internal_weights_memory, &conv11_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv11_weights_memory = conv11_internal_weights_memory
                ? conv11_internal_weights_memory
                : conv11_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv11;
    CHECK(dnnl_primitive_create(&conv11, conv11_pd));
    net_fwd[n_fwd] = conv11;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, pool4_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv11_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv11_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv11_internal_dst_memory);
    n_fwd++;

    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu11_src_md = conv11_dst_md;
    const_dnnl_memory_desc_t relu11_dst_md = relu11_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu11_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu11_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu11_src_md, relu11_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu11_dst_memory;
    CHECK(dnnl_memory_create(
            &relu11_dst_memory, relu11_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu11;
    CHECK(dnnl_primitive_create(&relu11, relu11_pd));
    net_fwd[n_fwd] = relu11;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv11_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu11_dst_memory);
    n_fwd++;
    
    // Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 14, 14/ out:14, 14
    dnnl_dims_t conv12_user_weights_sizes = {512, 512, 3, 3};
    dnnl_dims_t conv12_bias_sizes = {512};
    dnnl_dims_t conv12_user_dst_sizes = {BATCH, 512, 14, 14};
    dnnl_dims_t conv12_strides = {1, 1};
    dnnl_dims_t conv12_dilation = {0, 0};
    dnnl_dims_t conv12_padding = {1, 1};

    float *conv12_weights = (float *)mlm_malloc(
                product(conv12_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv12_bias
            = (float *)mlm_malloc(product(conv12_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv12_weights, ndims, conv12_user_weights_sizes);
    init_net_data(conv12_bias, 1, conv12_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv12_user_weights_memory, conv12_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv12_user_weights_sizes, dnnl_oihw, engine,
            conv12_weights, &conv12_user_weights_memory);
    init_data_memory(1, conv12_bias_sizes, dnnl_x, engine, conv12_bias,
            &conv12_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv12_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv12_weights_md, conv12_bias_md, conv12_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv12_weights_md, ndims,
                    conv12_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv12_bias_md, 1, conv12_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv12_dst_md, ndims,
                    conv12_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv12_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu11_dst_md,
                    conv12_weights_md, conv12_bias_md, conv12_dst_md, conv12_strides,
                    conv12_dilation, conv12_padding, conv12_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv12_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv12_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv12_dst_md));
    }

    dnnl_memory_t conv12_internal_weights_memory, conv12_internal_dst_memory;
    const_dnnl_memory_desc_t conv12_dst_md
            = dnnl_primitive_desc_query_md(conv12_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv12_internal_dst_memory, conv12_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv12_reorder_weights;
    const_dnnl_memory_desc_t conv12_weights_md
                = dnnl_primitive_desc_query_md(conv12_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv12_user_weights_memory, conv12_weights_md, engine, 1,
            &conv12_internal_weights_memory, &conv12_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv12_weights_memory = conv12_internal_weights_memory
                ? conv12_internal_weights_memory
                : conv12_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv12;
    CHECK(dnnl_primitive_create(&conv12, conv12_pd));
    net_fwd[n_fwd] = conv12;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu11_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv12_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv12_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv12_internal_dst_memory);
    n_fwd++;

    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu12_src_md = conv12_dst_md;
    const_dnnl_memory_desc_t relu12_dst_md = relu12_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu12_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu12_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu12_src_md, relu12_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu12_dst_memory;
    CHECK(dnnl_memory_create(
            &relu12_dst_memory, relu12_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu12;
    CHECK(dnnl_primitive_create(&relu12, relu12_pd));
    net_fwd[n_fwd] = relu12;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv12_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu12_dst_memory);
    n_fwd++;
    
    // Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) / in: 14, 14/ out:14, 14
    dnnl_dims_t conv13_user_weights_sizes = {512, 512, 3, 3};
    dnnl_dims_t conv13_bias_sizes = {512};
    dnnl_dims_t conv13_user_dst_sizes = {BATCH, 512, 14, 14};
    dnnl_dims_t conv13_strides = {1, 1};
    dnnl_dims_t conv13_dilation = {0, 0};
    dnnl_dims_t conv13_padding = {1, 1};

    float *conv13_weights = (float *)mlm_malloc(
                product(conv13_user_weights_sizes, ndims) * sizeof(float),1);
    float *conv13_bias
            = (float *)mlm_malloc(product(conv13_bias_sizes, 1) * sizeof(float),1);
    
    init_net_data(conv13_weights, ndims, conv13_user_weights_sizes);
    init_net_data(conv13_bias, 1, conv13_bias_sizes);

    // create memory for user data
    dnnl_memory_t conv13_user_weights_memory, conv13_user_bias_memory;
    
    // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
    init_data_memory(ndims, conv13_user_weights_sizes, dnnl_oihw, engine,
            conv13_weights, &conv13_user_weights_memory);
    init_data_memory(1, conv13_bias_sizes, dnnl_x, engine, conv13_bias,
            &conv13_user_bias_memory);
    
    // create a convolution
    dnnl_primitive_desc_t conv13_pd;
    {
            // create data descriptors for convolution w/ no specified format
            dnnl_memory_desc_t conv13_weights_md, conv13_bias_md, conv13_dst_md;
           
            CHECK(dnnl_memory_desc_create_with_tag(&conv13_weights_md, ndims,
                    conv13_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &conv13_bias_md, 1, conv13_bias_sizes, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(&conv13_dst_md, ndims,
                    conv13_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
            
            CHECK(dnnl_convolution_forward_primitive_desc_create(&conv13_pd, engine,
                    dnnl_forward, dnnl_convolution_direct, relu12_dst_md,
                    conv13_weights_md, conv13_bias_md, conv13_dst_md, conv13_strides,
                    conv13_dilation, conv13_padding, conv13_padding, NULL));
            
            CHECK(dnnl_memory_desc_destroy(conv13_weights_md));
            CHECK(dnnl_memory_desc_destroy(conv13_bias_md));
            CHECK(dnnl_memory_desc_destroy(conv13_dst_md));
    }

    dnnl_memory_t conv13_internal_weights_memory, conv13_internal_dst_memory;
    const_dnnl_memory_desc_t conv13_dst_md
            = dnnl_primitive_desc_query_md(conv13_pd, dnnl_query_dst_md, 0);
    CHECK(dnnl_memory_create(&conv13_internal_dst_memory, conv13_dst_md, engine,
            DNNL_MEMORY_ALLOCATE));
    
    dnnl_primitive_t conv13_reorder_weights;
    const_dnnl_memory_desc_t conv13_weights_md
                = dnnl_primitive_desc_query_md(conv13_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&conv13_user_weights_memory, conv13_weights_md, engine, 1,
            &conv13_internal_weights_memory, &conv13_reorder_weights, &n_fwd,
            net_fwd, net_fwd_args));
    dnnl_memory_t conv13_weights_memory = conv13_internal_weights_memory
                ? conv13_internal_weights_memory
                : conv13_user_weights_memory;
    
    // finally create a convolution primitive
    dnnl_primitive_t conv13;
    CHECK(dnnl_primitive_create(&conv13, conv13_pd));
    net_fwd[n_fwd] = conv13;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu12_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS,
            conv13_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, conv13_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST,
            conv13_internal_dst_memory);
    n_fwd++;
    
    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu13_src_md = conv13_dst_md;
    const_dnnl_memory_desc_t relu13_dst_md = relu13_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu13_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu13_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu13_src_md, relu13_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu13_dst_memory;
    CHECK(dnnl_memory_create(
            &relu13_dst_memory, relu13_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu13;
    CHECK(dnnl_primitive_create(&relu13, relu13_pd));
    net_fwd[n_fwd] = relu13;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC,
            conv13_internal_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu13_dst_memory);
    n_fwd++;

    // MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) / in: 14, 14/ out:7, 7
     dnnl_dims_t pool5_dst_sizes = {BATCH, 512, 7, 7};
     dnnl_dims_t pool5_kernel = {2, 2};
     dnnl_dims_t pool5_strides = {2, 2};
     dnnl_dims_t pool5_padding = {0, 0};
     dnnl_dims_t pool5_dilation = {0, 0};
   
     dnnl_primitive_desc_t pool5_pd;
     {
        // create pooling src memory descriptor using dst descriptor
        //  from previous primitive

        // create descriptors for dst pooling data
        dnnl_memory_desc_t pool5_dst_md;
        CHECK(dnnl_memory_desc_create_with_tag(&pool5_dst_md, 4, pool5_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));

        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool5_pd, engine,
                dnnl_forward, dnnl_pooling_max, relu13_dst_md, pool5_dst_md,
                pool5_strides, pool5_kernel, pool5_dilation, pool5_padding,
                pool5_padding, NULL));
        CHECK(dnnl_memory_desc_destroy(pool5_dst_md));
     }

        // create memory for workspace
     dnnl_memory_t pool5_ws_memory;
     const_dnnl_memory_desc_t pool5_ws_md
         = dnnl_primitive_desc_query_md(pool5_pd, dnnl_query_workspace_md, 0);
     CHECK(dnnl_memory_create(
                &pool5_ws_memory, pool5_ws_md, engine, DNNL_MEMORY_ALLOCATE));

     // create reorder primitives between pooling dsts and user format dst
     // if required
     // dnnl_primitive_t pool_reorder_dst;
     dnnl_memory_t pool5_dst_memory;
     const_dnnl_memory_desc_t pool5_dst_md
        = dnnl_primitive_desc_query_md(pool5_pd, dnnl_query_dst_md, 0);
        
     CHECK(dnnl_memory_create(&pool5_dst_memory, pool5_dst_md, engine, DNNL_MEMORY_ALLOCATE));

     // finally create a pooling primitive
     dnnl_primitive_t pool5;
     CHECK(dnnl_primitive_create(&pool5, pool5_pd));
     net_fwd[n_fwd] = pool5;
     prepare_arg_node(&net_fwd_args[n_fwd], 3);
     set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu13_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, pool5_dst_memory);
     set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_WORKSPACE, pool5_ws_memory);
     n_fwd++;

    // Linear(in_features=25088, out_features=4096, bias=True)
    // weight dim {OC, IC, IH, IW}
    dnnl_dims_t fc_weights_sizes = {4096, 512, 7, 7};
    // bias dim {OC}
    dnnl_dims_t fc_bias_sizes = {4096};
    // dst dim {N, OC}
    dnnl_dims_t fc_dst_sizes = {BATCH, 4096};
    // allocate src, weight, bias, dst
    float *fc_weights = (float*)mlm_malloc(product(fc_weights_sizes, ndims) * sizeof(float), 1);
    float *fc_bias = (float*)mlm_malloc(product(fc_bias_sizes, 1)*sizeof(float), 1);
    // initialiaze weight, bias
    init_net_data(fc_weights, 4, fc_weights_sizes);
    init_net_data(fc_bias, 1, fc_bias_sizes);

    // create memory for user data
    dnnl_memory_t fc_user_weights_memory, fc_user_bias_memory;
    init_data_memory(ndims, fc_weights_sizes, dnnl_oihw, engine,
        fc_weights, &fc_user_weights_memory);
    init_data_memory(1, fc_bias_sizes, dnnl_x, engine, fc_bias,
        &fc_user_bias_memory);

    // create memory descriptor -> src_md, bias_md, dst_md, weights
    dnnl_primitive_desc_t fc_pd;
    {
        dnnl_memory_desc_t fc_weights_md, fc_bias_md, fc_dst_md;
        dnnl_memory_desc_t fc_src_md;
        CHECK(dnnl_memory_desc_create_with_tag(&fc_dst_md, 2, fc_dst_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&fc_weights_md, ndims, fc_weights_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&fc_bias_md, 1, fc_bias_sizes, dnnl_f32, dnnl_x));
                
        // dnnl_memory_desc_reshape(&fc_src_md, pool5_dst_md, 4, fc_src_sizes);
        CHECK(dnnl_inner_product_forward_primitive_desc_create(&fc_pd, engine, dnnl_forward, pool5_dst_md, fc_weights_md, fc_bias_md, fc_dst_md, NULL));
        CHECK(dnnl_memory_desc_destroy(fc_weights_md));
        CHECK(dnnl_memory_desc_destroy(fc_bias_md));
        CHECK(dnnl_memory_desc_destroy(fc_dst_md));
    }
        
    // create reorder primitives between pooling dsts and user format dst if required
    dnnl_primitive_t fc_reorder_weights;
    dnnl_memory_t fc_dst_memory, fc_internal_weights_memory;

    // create inner product primitive descriptor
    const_dnnl_memory_desc_t fc_dst_md = dnnl_primitive_desc_query_md(fc_pd, dnnl_query_dst_md, 0);
        
    CHECK(dnnl_memory_create(&fc_dst_memory, fc_dst_md, engine, DNNL_MEMORY_ALLOCATE));
    const_dnnl_memory_desc_t fc_weights_md = dnnl_primitive_desc_query_md(fc_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&fc_user_weights_memory, fc_weights_md, engine, 1,
                &fc_internal_weights_memory, &fc_reorder_weights, &n_fwd,
                net_fwd, net_fwd_args));
    dnnl_memory_t fc_weights_memory = fc_internal_weights_memory
                ? fc_internal_weights_memory
                : fc_user_weights_memory;

    // finally create a pooling primitive
    dnnl_primitive_t fc;
    CHECK(dnnl_primitive_create(&fc, fc_pd));
    net_fwd[n_fwd] = fc;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    // set args (src, weights, vias, dst)
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, pool5_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS, fc_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, fc_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST, fc_dst_memory);
    n_fwd++;
    
    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu14_src_md = fc_dst_md;
    const_dnnl_memory_desc_t relu14_dst_md = relu14_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu14_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu14_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu14_src_md, relu14_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu14_dst_memory;
    CHECK(dnnl_memory_create(
            &relu14_dst_memory, relu14_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu14;
    CHECK(dnnl_primitive_create(&relu14, relu14_pd));
    net_fwd[n_fwd] = relu14;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, fc_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu14_dst_memory);
    n_fwd++;

    // Linear(in_features=4096, out_features=4096, bias=True)
    // src dim {N, IC, IH, IW}
    dnnl_dims_t fc2_src_sizes = {BATCH, 4096, 1, 1};
    // weight dim {OC, IC, IH, IW}
    dnnl_dims_t fc2_weights_sizes = {4096, 4096, 1, 1};
    // bias dim {OC}
    dnnl_dims_t fc2_bias_sizes = {4096};
    // dst dim {N, OC}
    dnnl_dims_t fc2_dst_sizes = {BATCH, 4096};
    // allocate src, weight, bias, dst
    float *fc2_weights = (float*)mlm_malloc(product(fc2_weights_sizes, ndims) * sizeof(float), 1);
    float *fc2_bias = (float*)mlm_malloc(product(fc2_bias_sizes, 1)*sizeof(float), 1);
    // initialiaze weight, bias
    init_net_data(fc2_weights, 4, fc2_weights_sizes);
    init_net_data(fc2_bias, 1, fc2_bias_sizes);

    // create memory for user data
    dnnl_memory_t fc2_user_weights_memory, fc2_user_bias_memory;
    init_data_memory(ndims, fc2_weights_sizes, dnnl_oihw, engine,
        fc2_weights, &fc2_user_weights_memory);
    init_data_memory(1, fc2_bias_sizes, dnnl_x, engine, fc2_bias,
        &fc2_user_bias_memory);

    // create memory descriptor -> src_md, bias_md, dst_md, weights
    dnnl_primitive_desc_t fc2_pd;
    {
        dnnl_memory_desc_t fc2_weights_md, fc2_bias_md, fc2_dst_md;
        dnnl_memory_desc_t fc2_src_md;
        CHECK(dnnl_memory_desc_create_with_tag(&fc2_dst_md, 2, fc2_dst_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&fc2_weights_md, ndims, fc2_weights_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&fc2_bias_md, 1, fc2_bias_sizes, dnnl_f32, dnnl_x));
                
        dnnl_memory_desc_reshape(&fc2_src_md, relu14_dst_md, 4, fc2_src_sizes);
        CHECK(dnnl_inner_product_forward_primitive_desc_create(&fc2_pd, engine, dnnl_forward, fc2_src_md, fc2_weights_md, fc2_bias_md, fc2_dst_md, NULL));
        CHECK(dnnl_memory_desc_destroy(fc2_weights_md));
        CHECK(dnnl_memory_desc_destroy(fc2_bias_md));
        CHECK(dnnl_memory_desc_destroy(fc2_dst_md));
    }
        
    // create reorder primitives between pooling dsts and user format dst if required
    dnnl_primitive_t fc2_reorder_weights;
    dnnl_memory_t fc2_dst_memory, fc2_internal_weights_memory;

    // create inner product primitive descriptor
    const_dnnl_memory_desc_t fc2_dst_md = dnnl_primitive_desc_query_md(fc2_pd, dnnl_query_dst_md, 0);
        
    CHECK(dnnl_memory_create(&fc2_dst_memory, fc2_dst_md, engine, DNNL_MEMORY_ALLOCATE));
    const_dnnl_memory_desc_t fc2_weights_md = dnnl_primitive_desc_query_md(fc2_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&fc2_user_weights_memory, fc2_weights_md, engine, 1,
                &fc2_internal_weights_memory, &fc2_reorder_weights, &n_fwd,
                net_fwd, net_fwd_args));
    dnnl_memory_t fc2_weights_memory = fc2_internal_weights_memory
                ? fc2_internal_weights_memory
                : fc2_user_weights_memory;

    // finally create a pooling primitive
    dnnl_primitive_t fc2;
    CHECK(dnnl_primitive_create(&fc2, fc2_pd));
    net_fwd[n_fwd] = fc2;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    // set args (src, weights, vias, dst)
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu14_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS, fc2_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, fc2_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST, fc2_dst_memory);
    n_fwd++;

    // ReLU(inplace=True)
    const_dnnl_memory_desc_t relu15_src_md = fc2_dst_md;
    const_dnnl_memory_desc_t relu15_dst_md = relu15_src_md;

    // create a relu primitive descriptor
    dnnl_primitive_desc_t relu15_pd;
    CHECK(dnnl_eltwise_forward_primitive_desc_create(&relu15_pd, engine,
            dnnl_forward, dnnl_eltwise_relu, relu15_src_md, relu15_dst_md,
            negative_slope, 0, NULL));

    // create relu dst memory
    dnnl_memory_t relu15_dst_memory;
    CHECK(dnnl_memory_create(
            &relu15_dst_memory, relu15_dst_md, engine, DNNL_MEMORY_ALLOCATE));

    // finally create a relu primitive
    dnnl_primitive_t relu15;
    CHECK(dnnl_primitive_create(&relu15, relu15_pd));
    net_fwd[n_fwd] = relu15;
    prepare_arg_node(&net_fwd_args[n_fwd], 2);
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, fc2_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_DST, relu15_dst_memory);
    n_fwd++;
    
    // Linear(in_features=4096, out_features=1000, bias=True)
    // src dim {N, IC, IH, IW}
    dnnl_dims_t fc3_src_sizes = {BATCH, 4096, 1, 1};
    // weight dim {OC, IC, IH, IW}
    dnnl_dims_t fc3_weights_sizes = {1000, 4096, 1, 1};
    // bias dim {OC}
    dnnl_dims_t fc3_bias_sizes = {1000};
    // dst dim {N, OC}
    dnnl_dims_t fc3_dst_sizes = {BATCH, 1000};
    // allocate src, weight, bias, dst
    float *fc3_weights = (float*)mlm_malloc(product(fc3_weights_sizes, ndims) * sizeof(float), 1);
    float *fc3_bias = (float*)mlm_malloc(product(fc3_bias_sizes, 1)*sizeof(float), 1);
    // initialiaze weight, bias
    init_net_data(fc3_weights, 4, fc3_weights_sizes);
    init_net_data(fc3_bias, 1, fc3_bias_sizes);

    // create memory for user data
    dnnl_memory_t fc3_user_weights_memory, fc3_user_bias_memory, fc3_user_dst_memory;
    init_data_memory(ndims, fc3_weights_sizes, dnnl_oihw, engine,
        fc3_weights, &fc3_user_weights_memory);
    init_data_memory(1, fc3_bias_sizes, dnnl_x, engine, fc3_bias,
        &fc3_user_bias_memory);
    // last fc layer
    init_data_memory(2, fc3_dst_sizes, dnnl_nc, engine, net_dst, &fc3_user_dst_memory);

    // create memory descriptor -> src_md, bias_md, dst_md, weights
    dnnl_primitive_desc_t fc3_pd;
    {
        dnnl_memory_desc_t fc3_weights_md, fc3_bias_md, fc3_dst_md;
        dnnl_memory_desc_t fc3_src_md;
        CHECK(dnnl_memory_desc_create_with_tag(&fc3_dst_md, 2, fc3_dst_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&fc3_weights_md, ndims, fc3_weights_sizes, dnnl_f32, dnnl_format_tag_any));
        CHECK(dnnl_memory_desc_create_with_tag(&fc3_bias_md, 1, fc3_bias_sizes, dnnl_f32, dnnl_x));
                
        dnnl_memory_desc_reshape(&fc3_src_md, relu15_dst_md, 4, fc3_src_sizes);
        CHECK(dnnl_inner_product_forward_primitive_desc_create(&fc3_pd, engine, dnnl_forward, fc3_src_md, fc3_weights_md, fc3_bias_md, fc3_dst_md, NULL));
        CHECK(dnnl_memory_desc_destroy(fc3_weights_md));
        CHECK(dnnl_memory_desc_destroy(fc3_bias_md));
        CHECK(dnnl_memory_desc_destroy(fc3_dst_md));
    }
        
    // create reorder primitives between pooling dsts and user format dst if required
    dnnl_primitive_t fc3_reorder_dst, fc3_reorder_weights;
    dnnl_memory_t fc3_internal_dst_memory, fc3_dst_memory, fc3_internal_weights_memory;

    // create inner product primitive descriptor
    const_dnnl_memory_desc_t fc3_dst_md = dnnl_primitive_desc_query_md(fc3_pd, dnnl_query_dst_md, 0);

    // last fc layer
    n_fwd += 1;
    CHECK(prepare_reorder(&fc3_user_dst_memory, fc3_dst_md, engine, 0, &fc3_internal_dst_memory, &fc3_reorder_dst, &n_fwd, net_fwd, net_fwd_args));
    n_fwd -= fc3_reorder_dst ? 2 : 1;
    fc3_dst_memory = fc3_internal_dst_memory
                ? fc3_internal_dst_memory
                : fc3_user_dst_memory;
    
    const_dnnl_memory_desc_t fc3_weights_md = dnnl_primitive_desc_query_md(fc3_pd, dnnl_query_weights_md, 0);
    CHECK(prepare_reorder(&fc3_user_weights_memory, fc3_weights_md, engine, 1,
                &fc3_internal_weights_memory, &fc3_reorder_weights, &n_fwd,
                net_fwd, net_fwd_args));
    dnnl_memory_t fc3_weights_memory = fc3_internal_weights_memory
                ? fc3_internal_weights_memory
                : fc3_user_weights_memory;

    // finally create a pooling primitive
    dnnl_primitive_t fc3;
    CHECK(dnnl_primitive_create(&fc3, fc3_pd));
    net_fwd[n_fwd] = fc3;
    prepare_arg_node(&net_fwd_args[n_fwd], 4);
    // set args (src, weights, vias, dst)
    set_arg(&net_fwd_args[n_fwd].args[0], DNNL_ARG_SRC, relu15_dst_memory);
    set_arg(&net_fwd_args[n_fwd].args[1], DNNL_ARG_WEIGHTS, fc3_weights_memory);
    set_arg(&net_fwd_args[n_fwd].args[2], DNNL_ARG_BIAS, fc3_user_bias_memory);
    set_arg(&net_fwd_args[n_fwd].args[3], DNNL_ARG_DST, fc3_dst_memory);
    n_fwd++;

    // last fc layer
    if(fc3_reorder_dst) n_fwd += 1;
        
    // execute code
    dnnl_stream_t stream;
    std::cout <<"Total " <<  n_fwd << " layers" <<  std::endl; 
    CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));
    for (uint32_t i = 0; i < n_fwd; ++i) {
        CHECK(dnnl_primitive_execute(
                net_fwd[i], stream, net_fwd_args[i].nargs, net_fwd_args[i].args));
        printf("%d executed\n", i);
    }

    CHECK(dnnl_stream_wait(stream));
    std::cout << "inference completed" << std::endl;  
    dnnl_stream_destroy(stream);
    // clean-up
//     for (uint32_t i = 0; i < n_fwd; ++i)
//         free_arg_node(&net_fwd_args[i]);
    
//     CHECK(dnnl_primitive_desc_destroy(conv_pd));
//     CHECK(dnnl_primitive_desc_destroy(relu_pd));
//     CHECK(dnnl_primitive_desc_destroy(pool_pd));
//     CHECK(dnnl_primitive_desc_destroy(fc_pd));

}

int main(int argc, char **argv) {
    vgg16_net();
    printf("Example passed on CPU.\n");
    return 0;
}