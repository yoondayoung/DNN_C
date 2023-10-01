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
// void* mlm_malloc(size_t size, int level)
// {
// 	if(size == 0)
//       {
// 		// printf("ZERO BYTE MALLOC\n");
// 		void* bt_entries[64];
// 		exit(-1);
// 	}

// 	// printf("Performing a mlm Malloc for size %lu\n", size);

// 	return malloc(size);
// }
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
// Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
// ReLU(inplace=False)
// MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
// AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
// Linear(in_features, out_features, bias=True, device=None, dtype=None)
// BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
// GELU : 인자 없음
// LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None)
enum LayerType { Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, BatchNorm2d, GELU, LayerNorm };
enum LayerOrder { First, Internal, Last };

struct prv_layer{
        dnnl_primitive_desc_t fwd_hint;
        const_dnnl_memory_desc_t dst_md;
        dnnl_memory_t dst_memory;
        const_dnnl_memory_desc_t src_md;
        dnnl_memory_t src_memory;
        dnnl_memory_t ws_memory;
};

struct layer_info{
        LayerOrder layer_order;
        int in_h;
        int in_w;
        int in_c;
        int out_h;
        int out_w;
        int out_c;
        int k;
        int s;
        int p;
        int d;
        bool b;
        prv_layer player; // 이전 레이어 정보
        prv_layer clayer; // 현재 레이어의 실행 결과 정보 (fwd->bwd)
};

std::vector <std::string> token_ret(std::string ori, std::string d){
    std::vector <std::string> ret;
    std::vector <int> de;

    de.push_back(-1);
    for(int i=0; i< ori.size(); i++){
        if(d.find(ori[i]) == -1) continue;
        de.push_back(i);
    }
    de.push_back((int)ori.size());

    for(int i=1; i<(int)de.size(); i++){
        int s = de[i-1] + 1;
        int e = de[i] - 1;
        if(s > e) continue;

        ret.push_back(ori.substr(s, e-s+1));
    }
    
    return ret;
}

layer_info parsing_cmd_args(int num, std::string model_info){
    layer_info result;
    
    std::vector <std::string> model = token_ret(model_info, "(,)= ");
    switch(num){
        case Conv2d:
            result.in_c = stoi(model.at(0));
            result.out_c = stoi(model.at(1));
            result.k = stoi(model.at(3));
            result.s = stoi(model.at(6));
            if(model.size() > 10){
                result.p = stoi(model.at(9));
            }
            else result.p = 0;
            result.d = 0;
            break;
        case ReLU:
            //do nothing.
            break;
        case MaxPool2d:
            result.k = stoi(model.at(1));
            result.s = stoi(model.at(3));
            result.p = stoi(model.at(5));
            result.d = stoi(model.at(7))-1;
            break;

        case AdaptiveAvgPool2d:
            // result.out_h = stoi(model.at(1));
            // result.out_w = stoi(model.at(2));
            // result.k = 3;
            // result.p = 1;
            // result.s = 1;
            break;
        
        case Linear:
            //TODO
            //bias 에 대한 정보는 어떻게 처리할 건지 정해주세요.
            result.in_c = stoi(model.at(1));
            result.out_c = stoi(model.at(3));
            if(std::string(model.at(5))=="True\n")
                result.b = true;
            else result.b = false;
            break;
        
        case BatchNorm2d:
            // do nothing
            break;
        
        case LayerNorm:
            break;
    }
    return result;
}

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

static void init_scale_data(float *data, uint32_t dim, const dnnl_dim_t *dims) {
    if (dim == 1) {
        for (dnnl_dim_t i = 0; i < dims[0]; ++i) {
            data[i] = std::sin(i* 2.f);
        }
    } else if (dim == 4) {
        for (dnnl_dim_t in = 0; in < dims[0]; ++in)
            for (dnnl_dim_t ic = 0; ic < dims[1]; ++ic)
                for (dnnl_dim_t ih = 0; ih < dims[2]; ++ih)
                    for (dnnl_dim_t iw = 0; iw < dims[3]; ++iw) {
                        dnnl_dim_t indx = in * dims[1] * dims[2] * dims[3]
                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] = std::sin(indx * 2.f);
                    }
    }
}

static void init_shift_data(float *data, uint32_t dim, const dnnl_dim_t *dims) {
    if (dim == 1) {
        for (dnnl_dim_t i = 0; i < dims[0]; ++i) {
            data[i] = std::tan(float(i));
        }
    } else if (dim == 4) {
        for (dnnl_dim_t in = 0; in < dims[0]; ++in)
            for (dnnl_dim_t ic = 0; ic < dims[1]; ++ic)
                for (dnnl_dim_t ih = 0; ih < dims[2]; ++ih)
                    for (dnnl_dim_t iw = 0; iw < dims[3]; ++iw) {
                        dnnl_dim_t indx = in * dims[1] * dims[2] * dims[3]
                                + ic * dims[2] * dims[3] + ih * dims[3] + iw;
                        data[indx] =std::tan(float(indx));
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

// in_h, in_w, in_c: input height, weight, channel
// out_h, out_w, out_c: output height, weight, channel
// k: kernel_size, s: stride size, p: padding, d: dilation
prv_layer create_conv_layer(dnnl_engine_t engine, uint32_t *n_fwd, dnnl_primitive_t *net_fwd, args_t *net_fwd_args, layer_info layer_i, float *net_src){
        // std::cout << "create conv layer: in c" << layer_i.in_c << ",in h " << layer_i.in_h << ",in w " << layer_i.in_w << ",b " << layer_i.b << ",d " << layer_i.d << ",p "  << layer_i.p << ",k "  << layer_i.k << ",s " << layer_i.s << ",out c " << layer_i.out_c << ", out h" << layer_i.out_h << ", out w" << layer_i.out_w << std::endl;
        dnnl_dims_t conv_user_src_sizes = { BATCH, layer_i.in_c, layer_i.in_h, layer_i.in_w };
        dnnl_dims_t conv_user_weights_sizes = {layer_i.out_c, layer_i.in_c, layer_i.k, layer_i.k};
        dnnl_dims_t conv_bias_sizes = {layer_i.out_c};
        dnnl_dims_t conv_user_dst_sizes = {BATCH, layer_i.out_c, layer_i.out_h, layer_i.out_w};
        dnnl_dims_t conv_strides = {layer_i.s, layer_i.s};
        dnnl_dims_t conv_dilation = {layer_i.d, layer_i.d};
        dnnl_dims_t conv_padding = {layer_i.p, layer_i.p};
        // std::cout << "=============create conv layer===============" << std:: endl;
        // std::cout << "src size:" << BATCH*layer_i.in_c*layer_i.in_h*layer_i.in_w*8 << "dst size:" << BATCH*layer_i.out_c*layer_i.out_h*layer_i.out_w*8 << std::endl;
        // std::cout << "weight size:" << layer_i.out_c*layer_i.in_c*layer_i.k*layer_i.k*8 << std::endl;
        // std::cout << "total: " << BATCH*layer_i.in_c*layer_i.in_h*layer_i.in_w*8+BATCH*layer_i.out_c*layer_i.out_h*layer_i.out_w*8+layer_i.out_c*layer_i.in_c*layer_i.k*layer_i.k*8 << std::endl;
        float *conv_src;
        if(layer_i.layer_order==First) conv_src = net_src;
        // else -> conv_src 사용 x

        float *conv_weights = (float *)malloc(
                product(conv_user_weights_sizes, ndims) * sizeof(float));
        float *conv_bias
                = (float *)malloc(product(conv_bias_sizes, 1) * sizeof(float));

        init_net_data(conv_weights, ndims, conv_user_weights_sizes);
        init_net_data(conv_bias, 1, conv_bias_sizes);

        // create memory for user data
        dnnl_memory_t conv_user_src_memory, conv_user_weights_memory,
                conv_user_bias_memory;
        if(layer_i.layer_order==First)    
                init_data_memory(ndims, conv_user_src_sizes, dnnl_nchw, engine, conv_src,
                        &conv_user_src_memory);
        // else -> 할 필요 없음 (conv_user_src_memory에 메모리 쓰기)
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
                if(layer_i.layer_order==First)
                        CHECK(dnnl_memory_desc_create_with_tag(&conv_src_md, ndims,
                                conv_user_src_sizes, dnnl_f32, dnnl_format_tag_any));
                CHECK(dnnl_memory_desc_create_with_tag(&conv_weights_md, ndims,
                        conv_user_weights_sizes, dnnl_f32, dnnl_format_tag_any));
                CHECK(dnnl_memory_desc_create_with_tag(
                        &conv_bias_md, 1, conv_bias_sizes, dnnl_f32, dnnl_x));
                CHECK(dnnl_memory_desc_create_with_tag(&conv_dst_md, ndims,
                        conv_user_dst_sizes, dnnl_f32, dnnl_format_tag_any));
                if(layer_i.layer_order==First)
                        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv_pd, engine,
                                dnnl_forward, dnnl_convolution_direct, conv_src_md,
                                conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
                                conv_dilation, conv_padding, conv_padding, NULL));
                else
                        CHECK(dnnl_convolution_forward_primitive_desc_create(&conv_pd, engine,
                                dnnl_forward, dnnl_convolution_direct, layer_i.player.dst_md,
                                conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
                                conv_dilation, conv_padding, conv_padding, NULL));
                if(layer_i.layer_order==First) CHECK(dnnl_memory_desc_destroy(conv_src_md));
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

        // 만약 이게 첫번째 layer라면 conv_src_md를 만들어야 함, 아니라면 전에거 갖다쓰기
        const_dnnl_memory_desc_t conv_src_md;
        if (layer_i.layer_order==First){
                conv_src_md = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_src_md, 0);
        }
        else {
                conv_src_md = layer_i.player.dst_md;
        }

        if(layer_i.layer_order==First)
                CHECK(prepare_reorder(&conv_user_src_memory, conv_src_md, engine, 1,
                        &conv_internal_src_memory, &conv_reorder_src, n_fwd, net_fwd,
                        net_fwd_args));

        const_dnnl_memory_desc_t conv_weights_md
                = dnnl_primitive_desc_query_md(conv_pd, dnnl_query_weights_md, 0);
        CHECK(prepare_reorder(&conv_user_weights_memory, conv_weights_md, engine, 1,
                &conv_internal_weights_memory, &conv_reorder_weights, n_fwd,
                net_fwd, net_fwd_args));

        dnnl_memory_t conv_src_memory;
        if(layer_i.layer_order==First) {
                conv_src_memory = conv_internal_src_memory
                ? conv_internal_src_memory
                : conv_user_src_memory;
        }
        else conv_src_memory = layer_i.player.dst_memory;
        dnnl_memory_t conv_weights_memory = conv_internal_weights_memory
                ? conv_internal_weights_memory
                : conv_user_weights_memory;

        // finally create a convolution primitive
        dnnl_primitive_t conv;
        CHECK(dnnl_primitive_create(&conv, conv_pd));
        // printf("cur n_fwd: %d\n", *n_fwd);
        net_fwd[*n_fwd] = conv;
        prepare_arg_node(&net_fwd_args[*n_fwd], 4);
        set_arg(&net_fwd_args[*n_fwd].args[0], DNNL_ARG_SRC, conv_src_memory);
        set_arg(&net_fwd_args[*n_fwd].args[1], DNNL_ARG_WEIGHTS,
                conv_weights_memory);
        set_arg(&net_fwd_args[*n_fwd].args[2], DNNL_ARG_BIAS, conv_user_bias_memory);
        set_arg(&net_fwd_args[*n_fwd].args[3], DNNL_ARG_DST,
                conv_internal_dst_memory);
        *n_fwd++;

        prv_layer return_val = { .fwd_hint = conv_pd, .dst_md = conv_dst_md, .dst_memory = conv_internal_dst_memory, .src_md = conv_src_md, .src_memory = conv_src_memory };
        return return_val;
}

// basic relu
// {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, CONV_OH, CONV_OW}
prv_layer create_relu_layer(dnnl_engine_t engine, uint32_t *n_fwd, dnnl_primitive_t *net_fwd, args_t *net_fwd_args, layer_info layer_i){
        // std::cout << "create relu layer: in c" << layer_i.in_c << ",in h " << layer_i.in_h << ",in w " << layer_i.in_w <<  ",out c " << layer_i.out_c << ", out h" << layer_i.out_h << ", out w" << layer_i.out_w << std::endl;
        float negative_slope = 0.0f;

        // keep memory format of source same as the format of convolution
        // output in order to avoid reorder
        const_dnnl_memory_desc_t relu_src_md = layer_i.player.dst_md;
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
        net_fwd[*n_fwd] = relu;
        prepare_arg_node(&net_fwd_args[*n_fwd], 2);
        set_arg(&net_fwd_args[*n_fwd].args[0], DNNL_ARG_SRC,
                layer_i.player.dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[1], DNNL_ARG_DST, relu_dst_memory);
        *n_fwd++;
        prv_layer return_val = { .fwd_hint = relu_pd, .dst_md = relu_dst_md, .dst_memory = relu_dst_memory, .src_md = relu_src_md, .src_memory = layer_i.player.dst_memory };
        return return_val;
}

// {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
// kernel: {k, k}
// strides: {s, s}
// dilation: {d, d}
prv_layer create_maxpool_layer(dnnl_engine_t engine, uint32_t *n_fwd, dnnl_primitive_t *net_fwd, args_t *net_fwd_args, layer_info layer_i){
        // std::cout << "create max pool layer: in c" << layer_i.in_c << ",in h " << layer_i.in_h << ",in w " << layer_i.in_w <<  ",out c " << layer_i.out_c << ", out h" << layer_i.out_h << ", out w" << layer_i.out_w << std::endl;
        dnnl_dims_t pool_dst_sizes = {BATCH, layer_i.out_c, layer_i.out_h, layer_i.out_w};
        dnnl_dims_t pool_kernel = {layer_i.k, layer_i.k};
        dnnl_dims_t pool_strides = {layer_i.s, layer_i.s};
        dnnl_dims_t pool_padding = {layer_i.p, layer_i.p};
        dnnl_dims_t pool_dilation = {layer_i.d, layer_i.d};
        // std::cout << "=============create conv layer===============" << std:: endl;
        // std::cout << "src size:" << BATCH*layer_i.in_c*layer_i.in_h*layer_i.in_w*8 << "dst size:" << BATCH*layer_i.out_c*layer_i.out_h*layer_i.out_w*8 << std::endl;
        // std::cout << "weight size:" << layer_i.out_c*layer_i.in_c*layer_i.k*layer_i.k*8 << std::endl;
        // std::cout << "total: " << BATCH*layer_i.in_c*layer_i.in_h*layer_i.in_w*8+BATCH*layer_i.out_c*layer_i.out_h*layer_i.out_w*8+layer_i.out_c*layer_i.in_c*layer_i.k*layer_i.k*8 << std::endl;

        // create a pooling primitive descriptor
        dnnl_primitive_desc_t pool_pd;

        {
                // create pooling src memory descriptor using dst descriptor
                //  from previous primitive
                const_dnnl_memory_desc_t pool_src_md = layer_i.player.dst_md;

                // create descriptors for dst pooling data
                dnnl_memory_desc_t pool_dst_md;
                CHECK(dnnl_memory_desc_create_with_tag(&pool_dst_md, 4, pool_dst_sizes,
                        dnnl_f32, dnnl_format_tag_any));

                CHECK(dnnl_pooling_forward_primitive_desc_create(&pool_pd, engine,
                        dnnl_forward, dnnl_pooling_max, pool_src_md, pool_dst_md,
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
        net_fwd[*n_fwd] = pool;
        prepare_arg_node(&net_fwd_args[*n_fwd], 3);
        set_arg(&net_fwd_args[*n_fwd].args[0], DNNL_ARG_SRC, layer_i.player.dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[1], DNNL_ARG_DST, pool_dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[2], DNNL_ARG_WORKSPACE, pool_ws_memory);
        *n_fwd++;
        prv_layer return_val = { .fwd_hint = pool_pd, .dst_md = pool_dst_md, .dst_memory = pool_dst_memory, .src_md = layer_i.player.dst_md, .src_memory = layer_i.player.dst_memory, .ws_memory = pool_ws_memory };
        return return_val;
}

// {BATCH, OC, CONV_OH, CONV_OW} -> {BATCH, OC, POOL_OH, POOL_OW}
// kernel: {k, k}
// strides: {s, s}
// dilation: {d, d}
prv_layer create_avgpool_layer(dnnl_engine_t engine, uint32_t *n_fwd, dnnl_primitive_t *net_fwd, args_t *net_fwd_args, layer_info layer_i){
        // std::cout << "create avgpool layer: in c" << layer_i.in_c << ",in h " << layer_i.in_h << ",in w " << layer_i.in_w << ",b " << layer_i.b << ",d " << layer_i.d << ",p "  << layer_i.p << ",k "  << layer_i.k << ",s " << layer_i.s << ",out c " << layer_i.out_c << ", out h" << layer_i.out_h << ", out w" << layer_i.out_w << std::endl;
        dnnl_dims_t pool_dst_sizes = {BATCH, layer_i.out_c, layer_i.out_h, layer_i.out_w};
        dnnl_dims_t pool_kernel = {layer_i.k, layer_i.k};
        dnnl_dims_t pool_strides = {layer_i.s, layer_i.s};
        dnnl_dims_t pool_padding = {layer_i.p, layer_i.p};
        dnnl_dims_t pool_dilation = {0, 0};

        // create a pooling primitive descriptor
        dnnl_primitive_desc_t pool_pd;

        {
                // create pooling src memory descriptor using dst descriptor
                //  from previous primitive
                const_dnnl_memory_desc_t pool_src_md = layer_i.player.dst_md;

                // create descriptors for dst pooling data
                dnnl_memory_desc_t pool_dst_md;
                CHECK(dnnl_memory_desc_create_with_tag(&pool_dst_md, 4, pool_dst_sizes,
                        dnnl_f32, dnnl_format_tag_any));
                if(layer_i.p > 0){
                        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool_pd, engine,
                        dnnl_forward, dnnl_pooling_avg_include_padding, pool_src_md, pool_dst_md,
                        pool_strides, pool_kernel, pool_dilation, pool_padding,
                        pool_padding, NULL));
                }
                else {
                        CHECK(dnnl_pooling_forward_primitive_desc_create(&pool_pd, engine,
                        dnnl_forward, dnnl_pooling_avg_exclude_padding, pool_src_md, pool_dst_md,
                        pool_strides, pool_kernel, pool_dilation, pool_padding,
                        pool_padding, NULL));
                }
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
        net_fwd[*n_fwd] = pool;
        prepare_arg_node(&net_fwd_args[*n_fwd], 3);
        set_arg(&net_fwd_args[*n_fwd].args[0], DNNL_ARG_SRC, layer_i.player.dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[1], DNNL_ARG_DST, pool_dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[2], DNNL_ARG_WORKSPACE, pool_ws_memory);
        *n_fwd++;
        prv_layer return_val = { .fwd_hint = pool_pd, .dst_md = pool_dst_md, .dst_memory = pool_dst_memory, .src_md = layer_i.player.dst_md, .src_memory = layer_i.player.dst_memory, .ws_memory=pool_ws_memory };
        return return_val;
}

prv_layer create_fc_layer(dnnl_engine_t engine, uint32_t *n_fwd, dnnl_primitive_t *net_fwd, args_t *net_fwd_args,  layer_info layer_i, float *net_dst){
        // std::cout << "create fc layer: in c" << layer_i.in_c << ",in h " << layer_i.in_h << ",in w " << layer_i.in_w <<  ",out c " << layer_i.out_c << ", out h" << layer_i.out_h << ", out w" << layer_i.out_w << std::endl;
        // src dim {N, IC, IH, IW}
        dnnl_dims_t fc_src_sizes = {BATCH, layer_i.in_c, layer_i.in_h, layer_i.in_w};
        // weight dim {OC, IC, IH, IW}
        dnnl_dims_t fc_weights_sizes = {layer_i.out_c, layer_i.in_c, layer_i.in_h, layer_i.in_w};
        // bias dim {OC}
        dnnl_dims_t fc_bias_sizes = {layer_i.out_c};
        // dst dim {N, OC}
        dnnl_dims_t fc_dst_sizes = {BATCH, layer_i.out_c};
        // allocate src, weight, bias, dst
        float *fc_weights = (float*)malloc(product(fc_weights_sizes, ndims) * sizeof(float));
        float *fc_bias = (float*)malloc(product(fc_bias_sizes, 1)*sizeof(float));
        // initialiaze weight, bias
        init_net_data(fc_weights, ndims, fc_weights_sizes);
        init_net_data(fc_bias, 1, fc_bias_sizes);

        // create memory for user data
        dnnl_memory_t fc_user_weights_memory, fc_user_bias_memory, fc_user_dst_memory;
        init_data_memory(ndims, fc_weights_sizes, dnnl_oihw, engine,
                fc_weights, &fc_user_weights_memory);
        init_data_memory(1, fc_bias_sizes, dnnl_x, engine, fc_bias,
                &fc_user_bias_memory);
        if (layer_i.layer_order==Last){
                init_data_memory(2, fc_dst_sizes, dnnl_nc, engine, net_dst, &fc_user_dst_memory);
        }

        // create memory descriptor -> src_md, bias_md, dst_md, weights
        dnnl_primitive_desc_t fc_pd;
        {
                dnnl_memory_desc_t fc_weights_md, fc_bias_md, fc_dst_md;
                dnnl_memory_desc_t fc_src_md;

                CHECK(dnnl_memory_desc_create_with_tag(&fc_dst_md, 2, fc_dst_sizes,
                dnnl_f32, dnnl_format_tag_any));
                CHECK(dnnl_memory_desc_create_with_tag(&fc_weights_md, ndims,
                        fc_weights_sizes, dnnl_f32, dnnl_format_tag_any));
                CHECK(dnnl_memory_desc_create_with_tag(
                        &fc_bias_md, 1, fc_bias_sizes, dnnl_f32, dnnl_x));
                
                dnnl_memory_desc_reshape(&fc_src_md, layer_i.player.dst_md, 4, fc_src_sizes);
                CHECK(dnnl_inner_product_forward_primitive_desc_create(&fc_pd, engine,
                                dnnl_forward, fc_src_md, fc_weights_md, fc_bias_md, fc_dst_md, NULL));

                CHECK(dnnl_memory_desc_destroy(fc_weights_md));
                CHECK(dnnl_memory_desc_destroy(fc_bias_md));
                CHECK(dnnl_memory_desc_destroy(fc_dst_md));
        }
        
        // create memory for dst data

        // create reorder primitives between pooling dsts and user format dst if required
        dnnl_primitive_t fc_reorder_dst, fc_reorder_weights;
        dnnl_memory_t fc_internal_dst_memory, fc_dst_memory, fc_internal_weights_memory;

        // create inner product primitive descriptor
        const_dnnl_memory_desc_t fc_dst_md = dnnl_primitive_desc_query_md(fc_pd, dnnl_query_dst_md, 0);
        if (layer_i.layer_order==Last){
                *n_fwd += 1;
                CHECK(prepare_reorder(&fc_user_dst_memory, fc_dst_md, engine, 0, &fc_internal_dst_memory, &fc_reorder_dst, n_fwd, net_fwd, net_fwd_args));
                *n_fwd -= fc_reorder_dst ? 2 : 1;
                fc_dst_memory = fc_internal_dst_memory
                        ? fc_internal_dst_memory
                        : fc_user_dst_memory;
        }
        else{
                CHECK(dnnl_memory_create(&fc_dst_memory, fc_dst_md, engine, DNNL_MEMORY_ALLOCATE));
        }

        const_dnnl_memory_desc_t fc_weights_md
                        = dnnl_primitive_desc_query_md(fc_pd, dnnl_query_weights_md, 0);
        CHECK(prepare_reorder(&fc_user_weights_memory, fc_weights_md, engine, 1,
                &fc_internal_weights_memory, &fc_reorder_weights, n_fwd,
                net_fwd, net_fwd_args));
        dnnl_memory_t fc_weights_memory = fc_internal_weights_memory
                ? fc_internal_weights_memory
                : fc_user_weights_memory;

        // finally create a pooling primitive
        dnnl_primitive_t fc;
        CHECK(dnnl_primitive_create(&fc, fc_pd));
        net_fwd[*n_fwd] = fc;
        prepare_arg_node(&net_fwd_args[*n_fwd], 4);
        // set args (src, weights, vias, dst)
        set_arg(&net_fwd_args[*n_fwd].args[0], DNNL_ARG_SRC, layer_i.player.dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[1], DNNL_ARG_WEIGHTS,
                fc_weights_memory);
        set_arg(&net_fwd_args[*n_fwd].args[2], DNNL_ARG_BIAS, fc_user_bias_memory);
        set_arg(&net_fwd_args[*n_fwd].args[3], DNNL_ARG_DST,
                fc_dst_memory);
        *n_fwd++;
        if(layer_i.layer_order==Last && fc_reorder_dst) *n_fwd += 1;
        
        prv_layer return_val = { .fwd_hint = fc_pd, .dst_md = fc_dst_md, .dst_memory = fc_dst_memory, .src_md = layer_i.player.dst_md, .src_memory = layer_i.player.dst_memory };
        return return_val;
}

prv_layer create_bn_layer(dnnl_engine_t engine, uint32_t *n_fwd, dnnl_primitive_t *net_fwd, args_t *net_fwd_args,  layer_info layer_i){
        // std::cout << "create bn layer: in c" << layer_i.in_c << ",in h " << layer_i.in_h << ",in w " << layer_i.in_w <<  ",out c " << layer_i.out_c << ", out h" << layer_i.out_h << ", out w" << layer_i.out_w << std::endl;
        // src dim {N, IC, IH, IW}
        dnnl_dims_t bn_src_sizes = {BATCH, layer_i.in_c, layer_i.in_h, layer_i.in_w};
        dnnl_dims_t scaleshift_dims = {layer_i.in_c};

        // allocate src, weight, bias, dst
        float *scales = (float*)malloc(product(scaleshift_dims, 1) * sizeof(float));
        float *shifts = (float*)malloc(product(scaleshift_dims, 1)*sizeof(float));
        // initialiaze weight, bias
        init_scale_data(scales, 1, scaleshift_dims);
        init_shift_data(shifts, 1, scaleshift_dims);

        // Create src and scale/shift memory descriptors and memory objects.
        dnnl_memory_t scales_memory, shifts_memory;
        init_data_memory(1, scaleshift_dims, dnnl_x, engine, scales, &scales_memory);
        init_data_memory(1, scaleshift_dims, dnnl_x, engine, shifts, &shifts_memory);

        // create memory descriptor -> src_md, bias_md, dst_md, weights
        dnnl_primitive_desc_t bn_pd;
        {
            dnnl_memory_desc_t bn_scales_md, bn_shifts_md, bn_dst_md;
            
            CHECK(dnnl_memory_desc_create_with_tag(&bn_dst_md, 4, bn_src_sizes,
            dnnl_f32, dnnl_format_tag_any));
            CHECK(dnnl_memory_desc_create_with_tag(&bn_scales_md, 1,
                    scaleshift_dims, dnnl_f32, dnnl_x));
            CHECK(dnnl_memory_desc_create_with_tag(
                    &bn_shifts_md, 1, scaleshift_dims, dnnl_f32, dnnl_x));
            
            // dnnl_memory_desc_reshape(&fc_src_md, layer_i.player.dst_md, 4, fc_src_sizes);
            CHECK(dnnl_batch_normalization_forward_primitive_desc_create(&bn_pd, engine,
                            dnnl_forward, layer_i.player.dst_md, bn_dst_md, 1.e-05f, dnnl_use_scale | dnnl_use_shift
                | dnnl_fuse_norm_relu, NULL));
            CHECK(dnnl_memory_desc_destroy(bn_dst_md));
            CHECK(dnnl_memory_desc_destroy(bn_scales_md));
            CHECK(dnnl_memory_desc_destroy(bn_shifts_md));
        }
       
        // create memory for dst data
        dnnl_memory_t bn_dst_memory;
        const_dnnl_memory_desc_t bn_dst_md = dnnl_primitive_desc_query_md(bn_pd, dnnl_query_dst_md, 0);
        CHECK(dnnl_memory_create(&bn_dst_memory, bn_dst_md, engine, DNNL_MEMORY_ALLOCATE));
        
        dnnl_memory_t mean_memory, variance_memory, workspace_memory;
        const_dnnl_memory_desc_t mean_md, variance_md, workspace_md;
        
        mean_md = dnnl_primitive_desc_query_md(bn_pd, dnnl_query_dst_md, 1);
        variance_md = dnnl_primitive_desc_query_md(bn_pd, dnnl_query_dst_md, 2);
        workspace_md = dnnl_primitive_desc_query_md(bn_pd, dnnl_query_workspace_md, 0);
        CHECK(dnnl_memory_create(&mean_memory, mean_md, engine, DNNL_MEMORY_ALLOCATE));
        CHECK(dnnl_memory_create(&variance_memory, variance_md, engine, DNNL_MEMORY_ALLOCATE));
        CHECK(dnnl_memory_create(&workspace_memory, workspace_md, engine, DNNL_MEMORY_ALLOCATE));

        // finally create a pooling primitive
        dnnl_primitive_t bn;
        CHECK(dnnl_primitive_create(&bn, bn_pd));
        net_fwd[*n_fwd] = bn;
        prepare_arg_node(&net_fwd_args[*n_fwd], 7);
        // set args (src, weights, vias, dst)
        set_arg(&net_fwd_args[*n_fwd].args[0], DNNL_ARG_SRC, layer_i.player.dst_memory);
        set_arg(&net_fwd_args[*n_fwd].args[1], DNNL_ARG_MEAN, mean_memory);
        set_arg(&net_fwd_args[*n_fwd].args[2], DNNL_ARG_VARIANCE, variance_memory);
        set_arg(&net_fwd_args[*n_fwd].args[3], DNNL_ARG_SCALE, scales_memory);
        set_arg(&net_fwd_args[*n_fwd].args[4], DNNL_ARG_SHIFT, shifts_memory);
        set_arg(&net_fwd_args[*n_fwd].args[5], DNNL_ARG_WORKSPACE, workspace_memory);
        set_arg(&net_fwd_args[*n_fwd].args[6], DNNL_ARG_DST, layer_i.player.dst_memory);
        *n_fwd++;
        
        prv_layer return_val = { .fwd_hint = bn_pd, .dst_md = bn_dst_md, .dst_memory = bn_dst_memory, .src_md = layer_i.player.dst_md, .src_memory = layer_i.player.dst_memory };
        return return_val;
}

void execute_model(FILE* model_file) {
    char line[300];

    dnnl_engine_t engine;
    CHECK(dnnl_engine_create(&engine, dnnl_cpu, 0));

    // build a net from file
    uint32_t n_fwd = 0;
    fgets(line, sizeof(line), model_file);
    uint32_t n_mod = atoi(line);
    dnnl_primitive_t net_fwd[n_mod];
    args_t net_fwd_args[n_mod];

    dnnl_dims_t net_src_sizes = {BATCH, IC, CONV_IH, CONV_IW};
    dnnl_dims_t net_dst_sizes = {BATCH, LIN_OC};

    float *net_src
            = (float *)malloc(product(net_src_sizes, ndims) * sizeof(float));
    float *net_dst
            = (float *)malloc(product(net_dst_sizes, ndims) * sizeof(float));

    init_net_data(net_src, ndims, net_src_sizes);
    memset(net_dst, 0, product(net_dst_sizes, ndims) * sizeof(float));

    // info for each layer
    int cur_cmd;
    
    layer_info layer_args, back_layer_args;
    prv_layer tmp_layer_md;
    int next_ic, next_ih, next_iw;
    next_ic = IC; next_ih = CONV_IH; next_iw = CONV_IW;
    // forward stream
    for(int mod_i=0; mod_i<n_mod; mod_i++){
        // get the layer command
        fgets(line, sizeof(line), model_file);
        cur_cmd = atoi(line);
        // printf("cur cmd: %d\n", cur_cmd);
        // get the command args
        fgets(line, sizeof(line), model_file);
        
        if (cur_cmd==3){
        //     std::cout << "adaptive skip" << std::endl;
            continue;
        }
        layer_args = parsing_cmd_args(cur_cmd, line);
        layer_args.in_c = next_ic;
        layer_args.in_h = next_ih;
        layer_args.in_w = next_iw;
        if (mod_i==0){
                layer_args.layer_order = First;
        }
        else if (mod_i==n_mod-1){
                layer_args.layer_order = Last;
        }
        else{
                layer_args.layer_order = Internal;
        }
        
        if (mod_i>0) layer_args.player = tmp_layer_md;
        
        switch(cur_cmd){
            case Conv2d:
                // out_h, out_w 차원 계산
                layer_args.out_h = (int)floor((layer_args.in_h-(layer_args.d+1)*(layer_args.k-1)+2*layer_args.p-1)/layer_args.s+1);
                layer_args.out_w = (int)floor((layer_args.in_w-(layer_args.d+1)*(layer_args.k-1)+2*layer_args.p-1)/layer_args.s+1);
                tmp_layer_md = create_conv_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args, net_src);
                break;
            case ReLU:
                // out_h, out_w 차원 계산
                layer_args.out_h = layer_args.in_h;
                layer_args.out_w = layer_args.in_w;
                layer_args.out_c = layer_args.in_c;
                tmp_layer_md = create_relu_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args);
                break;
            case MaxPool2d:
                // out_h, out_w 차원 계산
                layer_args.out_h = (int)floor((layer_args.in_h-(layer_args.d+1)*(layer_args.k-1)+2*layer_args.p-1)/layer_args.s+1);
                layer_args.out_w = (int)floor((layer_args.in_w-(layer_args.d+1)*(layer_args.k-1)+2*layer_args.p-1)/layer_args.s+1);
                layer_args.out_c = layer_args.in_c;
                tmp_layer_md = create_maxpool_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args);
                break;
            case AdaptiveAvgPool2d:
                // // out_h, out_w 차원 계산
                // // layer_args.out_h = (int)floor((layer_args.in_h-layer_args.k+2*layer_args.p-1)/layer_args.s)+1;
                // // layer_args.out_w = (int)floor((layer_args.in_w-layer_args.k+2*layer_args.p-1)/layer_args.s)+1;
                // layer_args.out_c = layer_args.in_c;
                // tmp_layer_md = create_avgpool_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args);
                break;
            case Linear:
                layer_args.out_h = 1;
                layer_args.out_w = 1;
                tmp_layer_md = create_fc_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args, net_dst);
                break;
            case BatchNorm2d:
                // create_bn_layer 구현하기
                layer_args.out_h = layer_args.in_h;
                layer_args.out_w = layer_args.in_w;
                layer_args.out_c = layer_args.in_c;
                tmp_layer_md = create_bn_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args);
                break;
            case GELU:
                // create_bn_layer 구현하기
                layer_args.out_h = layer_args.in_h;
                layer_args.out_w = layer_args.in_w;
                layer_args.out_c = layer_args.in_c;
                tmp_layer_md = create_relu_layer(engine, &n_fwd, net_fwd, net_fwd_args, layer_args);
                break;
            case LayerNorm:
                break;
            default:
                break;
        }
        next_ic = layer_args.out_c; next_ih = layer_args.out_h; next_iw = layer_args.out_w;
        layer_args.clayer = tmp_layer_md;
        // backward stack에 넣기
    }
    
    dnnl_memory_t net_user_dst_memory = tmp_layer_md.dst_memory;
    
    dnnl_stream_t stream;
    CHECK(dnnl_stream_create(&stream, engine, dnnl_stream_default_flags));
//     ariel_enable();
    for (uint32_t i = 0; i < 9; ++i){
        CHECK(dnnl_primitive_execute(net_fwd[i], stream,
            net_fwd_args[i].nargs, net_fwd_args[i].args));
    }
    
    CHECK(dnnl_stream_wait(stream));
//     std::cout << "inference completed" << std::endl;  
//     dnnl_stream_destroy(stream);

     // clean up nets
//     for (uint32_t i = 0; i < n_fwd; ++i)
//         free_arg_node(&net_fwd_args[i]);
// //     for (uint32_t i = 0; i < n_bwd; ++i)
// //         free_arg_node(&net_bwd_args[i]);

//     free(net_src);
//     free(net_dst);
    
//     free(net_diff_dst);

//     dnnl_engine_destroy(engine);
}

int main(int argc, char **argv) {
    FILE* model_file = fopen("/home/ydy/oneDNN/torch2oneDNN/make_models/check_googlenet_model.txt", "r");
    ariel_enable();
    execute_model(model_file);
//     printf("Example passed on CPU.\n");
//     fclose(model_file);
    return 0;
}