//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#include "air/base/container.h"
#include "air/base/st.h"
#include "air/base/visitor.h"
#include "air/core/handler.h"
#include "air/core/null_handler.h"
#include "nn/core/handler.h"
#include "nn/core/opcode.h"
#include "nn/vector/core_handler.h"
#include "nn/vector/tensor2vector_handler.h"
#include "nn/vector/vector_gen.h"
#include "nn/vector/vector_opcode.h"

using namespace air::base;
using namespace nn::core;
using namespace nn::vector;

int main() {
  bool ret = air::core::Register_core();
  AIR_ASSERT_MSG(ret, "Register core domain failed");
  ret = nn::core::Register_nn();
  AIR_ASSERT_MSG(ret, "Register nn domain failed");
  ret = nn::vector::Register_vector_domain();
  AIR_ASSERT_MSG(ret, "Register vector domain failed");

  // check opcodes
  bool op_is_valid =
      META_INFO::Valid_operator(VECTOR_DOMAIN::ID, VECTOR_OPCODE::MUL);
  std::cout << "op_is_valid MUL: " << op_is_valid << std::endl;
  op_is_valid =
      META_INFO::Valid_operator(VECTOR_DOMAIN::ID, VECTOR_OPCODE::ROLL);
  std::cout << "op_is_valid ROLL: " << op_is_valid << std::endl;

  op_is_valid =
      META_INFO::Valid_operator(nn::core::NN, nn::core::OPCODE::MAX_POOL);
  std::cout << "op_is_valid max pool: " << op_is_valid << std::endl;

  GLOB_SCOPE* glob = GLOB_SCOPE::Get();

  SPOS spos = glob->Unknown_simple_spos();
  // name of avg pool function
  STR_PTR name_str = glob->New_str("avg_pool_func");
  // avg pool function
  FUNC_PTR avg_pool_func = glob->New_func(name_str, spos);
  avg_pool_func->Set_parent(glob->Comp_env_id());
  // signature of avg pool function
  SIGNATURE_TYPE_PTR sig = glob->New_sig_type();
  // return type of avg pool function
  TYPE_PTR func_rtype = glob->Prim_type(PRIMITIVE_TYPE::VOID);
  glob->New_ret_param(func_rtype, sig);
  // parameter x of function avg pool
  TYPE_PTR base_type_idx = glob->Prim_type(PRIMITIVE_TYPE::FLOAT_32);
  std::vector<int64_t> shape{1, 1, 4, 4};
  TYPE_PTR             array_type =
      New_array_type(glob, "float", base_type_idx, shape, spos);

  STR_PTR var_str = glob->New_str("input");
  glob->New_param(var_str, array_type, sig, spos);
  sig->Set_complete();
  glob->New_entry_point(sig, avg_pool_func, name_str, spos);
  // set define before create a new scope
  FUNC_SCOPE* func_scope = &glob->New_func_scope(avg_pool_func);
  CONTAINER*  cntr       = &func_scope->Container();
  STMT_PTR    ent_stmt   = cntr->New_func_entry(spos);

  // array x;
  ADDR_DATUM_PTR var_x = func_scope->Formal(0);

  STR_PTR        z_str = glob->New_str("result");
  ADDR_DATUM_PTR var_z = func_scope->New_var(array_type, z_str, spos);

  // z = nn::core:MAX_POOL(x)
  NODE_PTR nnx_node   = cntr->New_ld(var_x, spos);
  NODE_PTR nnadd_node = cntr->New_una_arith(
      air::base::OPCODE(nn::core::NN, nn::core::OPCODE::MAX_POOL), nnx_node,
      spos);

  int64_t kernel_shape = 2;
  nnadd_node->Set_attr("kernel_shape", &kernel_shape, 1);

  int64_t stride = 2;
  nnadd_node->Set_attr("strides", &stride, 1);

  int64_t pad = 0;
  nnadd_node->Set_attr("pads", &pad, 1);

  STMT_PTR  nnstmt = cntr->New_st(nnadd_node, var_z, spos);
  STMT_LIST sl     = cntr->Stmt_list();
  sl.Append(nnstmt);
  // return z;
  NODE_PTR z_node   = cntr->New_ld(var_z, spos);
  STMT_PTR ret_stmt = cntr->New_retv(z_node, spos);
  sl.Append(ret_stmt);

  std::cout << "tensor IR:" << std::endl;
  func_scope->Print();

  // tensor->vector
  nn::vector::VECTOR_CTX    ctx;
  nn::vector::VECTOR_CONFIG config;
  GLOB_SCOPE* new_glob = nn::vector::Vector_driver(glob, ctx, nullptr, config);
  std::cout << "vector IR:" << std::endl;
  new_glob->Open_func_scope(func_scope->Id()).Print();

  return 0;
}
