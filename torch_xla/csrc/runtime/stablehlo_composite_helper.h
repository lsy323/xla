#ifndef STABLEHLO_COMPOSITE_HELPER_H_
#define STABLEHLO_COMPOSITE_HELPER_H_
#include <utility>

#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Pass/Pass.h"

namespace torch_xla {
namespace runtime {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreatePrepareTorchXLABoundariesPass();

}  // namespace runtime
}  // namespace torch_xla

#endif