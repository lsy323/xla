#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/OperationSupport.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeUtilities.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/TypeID.h"
#include "single_include/nlohmann/json.hpp"
#include "single_include/nlohmann/json_fwd.hpp"
#include "stablehlo/dialect/StablehloOps.h"

//===----------------------------------------------------------------------===//
// The PrepareTorchXLABoundaries Pass.
//
namespace torch_xla {
namespace runtime {

namespace {

using nlohmann::json;

struct BoundaryMetadata {
  std::string name;
  int64_t id;
  int64_t pos;
  bool is_input;
  std::unordered_map<std::string, json> attrs;

  std::string boundary_id() const { return absl::StrCat(name, "__", id); }

  auto uid() const { return std::forward_as_tuple(name, id, pos, is_input); }

  bool operator==(const BoundaryMetadata& other) const {
    return uid() == other.uid();
  }
  bool operator<(const BoundaryMetadata& other) const {
    return uid() < other.uid();
  }

  static std::unique_ptr<BoundaryMetadata> Parse(llvm::StringRef str) {
    auto j = json::parse(str, /*cb=*/nullptr, /*allow_exceptions=*/false);
    return Build(j);
  }

 private:
  static std::unique_ptr<BoundaryMetadata> Build(
      const nlohmann::basic_json<>& j) {
    BoundaryMetadata metadata;
#define FIELD(key, value_type)                           \
  if (auto kv = j.find(#key); kv != j.end()) {           \
    if (kv.value().type() != value_type) return nullptr; \
    kv.value().get_to(metadata.key);                     \
  } else {                                               \
    return nullptr;                                      \
  }

    FIELD(name, json::value_t::string);
    FIELD(id, json::value_t::number_unsigned);
    FIELD(pos, json::value_t::number_unsigned);
    FIELD(is_input, json::value_t::boolean);
#undef FIELD

    if (auto kv = j.find("attr"); kv != j.end() && kv.value().is_object()) {
      auto& attrs_j = kv.value();
      for (auto attr_j = attrs_j.begin(); attr_j != attrs_j.end(); ++attr_j) {
        metadata.attrs.insert({attr_j.key(), attr_j.value()});
      }
    }
    return std::make_unique<BoundaryMetadata>(std::move(metadata));
  }
};

class PrepareTorchXLABoundariesPass
    : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit PrepareTorchXLABoundariesPass()
      : mlir::OperationPass<mlir::ModuleOp>::OperationPass(
            mlir::TypeID::get<PrepareTorchXLABoundariesPass>()) {}

  ~PrepareTorchXLABoundariesPass() override = default;

  void runOnOperation() override {
    BuildStableHLOCompositeOps();
    EraseXlaMarkTensorOps();
  }

  mlir::StringRef getName() const override {
    return llvm::getTypeName<PrepareTorchXLABoundariesPass>();
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<PrepareTorchXLABoundariesPass>(*this);
  }

 private:
  llvm::DenseMap<const mlir::Operation*, size_t> BuildOperationsLineNumberMap(
      mlir::func::FuncOp func) const {
    llvm::DenseMap<const mlir::Operation*, size_t> op_line_num;
    for (const auto& op : llvm::enumerate(func.getOps())) {
      op_line_num[&op.value()] = op.index();
    }
    return op_line_num;
  }

  void BuildStableHLOCompositeOps() {
    auto module_op = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> raw_funcs(
        module_op.getOps<mlir::func::FuncOp>());
    for (auto func : raw_funcs) {
      llvm::DenseMap<const mlir::Operation*, size_t> op_line_num =
          BuildOperationsLineNumberMap(func);
      for (auto& op : func.getOps()) {
        BuildStableHLOCompositeOp(&op, op_line_num);
      }
    }
  }

  bool IsXlaMarkTensorOp(mlir::Operation* op) {
    if (op == nullptr) {
      return false;
    }
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return false;
    }
    if (op->getName().getStringRef() != "stablehlo.custom_call") {
      return false;
    }
    auto target_name =
        op->getAttr("call_target_name").dyn_cast<mlir::StringAttr>();
    if (target_name == nullptr || target_name.str() != "xla_mark_tensor") {
      return false;
    }
    return true;
  }

  void EraseXlaMarkTensorOps() {
    auto module_op = getOperation();
    for (auto func : module_op.getOps<mlir::func::FuncOp>()) {
      llvm::SmallVector<mlir::Operation*> raw_ops;
      for (mlir::Operation& op : func.getOps()) {
        raw_ops.push_back(&op);
      }

      for (mlir::Operation* mark_tensor : raw_ops) {
        if (!IsXlaMarkTensorOp(mark_tensor)) {
          continue;
        }
        mlir::Value original_value = mark_tensor->getOperand(0);

        llvm::SmallVector<std::tuple<mlir::Operation*, size_t>> uses;
        for (mlir::OpOperand& use : mark_tensor->getResult(0).getUses()) {
          uses.push_back({use.getOwner(), use.getOperandNumber()});
        }

        for (auto [user, operand_number] : uses) {
          user->setOperand(operand_number, original_value);
        }
        mark_tensor->erase();
      }
    }
  }

  std::unique_ptr<BoundaryMetadata> GetBoundaryMetadata(mlir::Operation* op) {
    if (!IsXlaMarkTensorOp(op)) {
      return nullptr;
    }
    auto backend_config =
        op->getAttr("backend_config").dyn_cast<mlir::StringAttr>();
    if (backend_config == nullptr) {
      return nullptr;
    }
    return BoundaryMetadata::Parse(backend_config);
  }

  void BuildStableHLOCompositeOp(
      mlir::Operation* op,
      const llvm::DenseMap<const mlir::Operation*, size_t>& op_line_num) {
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    std::unique_ptr<BoundaryMetadata> metadata = GetBoundaryMetadata(op);
    if (metadata == nullptr || metadata->is_input) {
      return;
    }
    const auto& output_metadata = *metadata;

    llvm::SetVector<mlir::Operation*> scope_ops_setvec;
    llvm::SetVector<std::pair<mlir::Value, int64_t>> arg_pos_setvec;
    llvm::SmallVector<mlir::Operation*> processing({op});

    while (!processing.empty()) {
      mlir::Operation* curr_op = processing.back();
      processing.pop_back();
      if (scope_ops_setvec.contains(curr_op)) {
        continue;
      }

      if (auto curr_metadata_ptr = GetBoundaryMetadata(curr_op);
          curr_metadata_ptr != nullptr) {
        const auto& curr_metadata = *curr_metadata_ptr;
        if (curr_metadata.is_input &&
            curr_metadata.boundary_id() == output_metadata.boundary_id()) {
          arg_pos_setvec.insert({curr_op->getResult(0).dyn_cast<mlir::Value>(),
                                 curr_metadata.pos});
          continue;
        }
      }

      scope_ops_setvec.insert(curr_op);
      for (mlir::Value value : curr_op->getOperands()) {
        mlir::Operation* def_op = value.getDefiningOp();
        if (def_op == nullptr) {
          // Global args
          arg_pos_setvec.insert({value, std::numeric_limits<int64_t>::max()});
        } else if (def_op->getName().getStringRef() == "stablehlo.constant") {
          scope_ops_setvec.insert(def_op);
        } else {
          processing.push_back(def_op);
        }
      }
    }

    auto scope_ops = scope_ops_setvec.takeVector();
    for (auto& op : scope_ops) {
      if (!op_line_num.contains(op)) {
        LOG(ERROR) << "!!!! Op line number not found";
        return;
      }
    }
    std::sort(scope_ops.begin(), scope_ops.end(),
              [&op_line_num](const auto& a, const auto& b) {
                return op_line_num.at(a) < op_line_num.at(b);
              });

    auto arg_pos_pairs = arg_pos_setvec.takeVector();
    std::stable_sort(
        arg_pos_pairs.begin(), arg_pos_pairs.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    llvm::SmallVector<mlir::Value> args;
    args.reserve(arg_pos_pairs.size());
    for (auto& [arg, unused] : arg_pos_pairs) {
      args.push_back(arg);
    }

    LOG(ERROR) << "-- ARGS:: " << args.size();
    LOG(ERROR) << "-- SCOPE_OPS:: " << scope_ops.size();
    for (auto scope_op : scope_ops) {
      LOG(ERROR) << "---- " << std::string(scope_op->getName().getStringRef());
    }

    llvm::SmallVector<mlir::Location> arg_locs;
    llvm::SmallVector<mlir::Type> arg_types,
        result_types(op->getResultTypes().begin(), op->getResultTypes().end());
    for (auto& arg : args) {
      arg_types.push_back(arg.getType());
      arg_locs.push_back(arg.getLoc());
    }

    mlir::ModuleOp module_op = getOperation();

    auto func_type = mlir::FunctionType::get(context, arg_types, result_types);
    auto func = builder.create<mlir::func::FuncOp>(
        module_op.getLoc(),
        absl::StrCat(output_metadata.boundary_id(), ".impl"), func_type);
    mlir::IRMapping mapping;
    builder.createBlock(&func.getBody(), func.begin(), arg_types, arg_locs);
    for (const auto& arg : llvm::enumerate(args)) {
      mapping.map(arg.value(), func.getArgument(arg.index()));
    }
    for (mlir::Operation* original_op : scope_ops) {
      mlir::Operation* cloned_op = builder.clone(*original_op, mapping);
      mapping.map(original_op, cloned_op);
    }
    builder.create<mlir::func::ReturnOp>(func.getBody().getLoc(),
                                         mapping.lookup(op)->getResults());

    // Adds the new function to symbol table.
    mlir::SymbolTable symbol_table(module_op);
    func.setPrivate();
    symbol_table.insert(func);

    // Replaces scope ops with call op to the new function
    builder.setInsertionPointAfter(op);
    llvm::SmallVector<mlir::NamedAttribute> call_attrs;
    call_attrs.push_back({
        builder.getStringAttr("call_target_name"),
        builder.getStringAttr("stablehlo.composite"),
    });
    call_attrs.push_back({builder.getStringAttr("called_computations"),
                          builder.getArrayAttr(mlir::FlatSymbolRefAttr::get(
                              builder.getContext(), func.getSymName()))});

    // Add boundary attributes to the new function.
    llvm::SmallVector<mlir::NamedAttribute> backend_config_attrs;
    for (auto& [key, j] : output_metadata.attrs) {
      switch (j.type()) {
        case json::value_t::number_integer:
        case json::value_t::number_unsigned:
          backend_config_attrs.push_back(
              {builder.getStringAttr(key),
               builder.getI64IntegerAttr(j.template get<int64_t>())});
          break;
        case json::value_t::number_float:
          backend_config_attrs.push_back(
              {builder.getStringAttr(key),
               builder.getI64IntegerAttr(j.template get<float>())});
          break;
        case json::value_t::boolean:
          backend_config_attrs.push_back(
              {builder.getStringAttr(key),
               builder.getI64IntegerAttr(j.template get<bool>())});
          break;
        default:
          // Ignored unrecognizable attr json
          break;
      }
    }
    call_attrs.push_back(
        {builder.getStringAttr("composite.backend_config"),
         builder.getDictionaryAttr(std::vector<mlir::NamedAttribute>{
             {builder.getStringAttr("attributes"),
              builder.getDictionaryAttr(backend_config_attrs)},
             {builder.getStringAttr("name"),
              builder.getStringAttr(output_metadata.name)},
         })});

    mlir::Operation* call_op = builder.create<mlir::stablehlo::CustomCallOp>(
        op->getLoc(), func.getFunctionType().getResults(), args, call_attrs);

    // Updates all users of this op's result(s) to use the results(s) of func
    // call.
    for (size_t i = 0; i < op->getNumResults(); ++i) {
      mlir::OpResult result = op->getResult(i);
      mlir::OpResult new_result = call_op->getResult(i);
      for (mlir::OpOperand& use : result.getUses()) {
        use.getOwner()->setOperand(use.getOperandNumber(), new_result);
      }
    }

    // The unused scope_ops can be eliminated with cse and canonicalize.
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreatePrepareTorchXLABoundariesPass() {
  return std::make_unique<PrepareTorchXLABoundariesPass>();
}

}  // namespace runtime
}  // namespace torch_xla
