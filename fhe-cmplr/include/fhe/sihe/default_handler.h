//-*-c++-*-
//=============================================================================
//
// Copyright (c) XXXX-XXXX
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=============================================================================

#ifndef FHE_SIHE_DEFAULT_HANDLER_H
#define FHE_SIHE_DEFAULT_HANDLER_H

#include "air/base/container.h"
#include "air/base/opcode_gen.h"
#include "fhe/sihe/opcode_def.inc"

namespace fhe {
namespace sihe {

//! @brief Default handler which always call visitor Context's Handle_node
//! and Handle_block to handle nodes
class DEFAULT_HANDLER {
public:
  // null handler implementation for each OPCODE
#define SIHE_OPCODE(NAME, name, cat, kid_num, fld_num, prop) \
  OPCODE_DEFAULT_HANDLER_GEN(NAME, name, cat, kid_num, fld_num, prop)
#include "opcode_def.inc"
#undef SIHE_OPCODE
};

}  // namespace sihe
}  // namespace fhe

#endif  // FHE_SIHE_DEFAULT_HANDLER_H
