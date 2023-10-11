import copy
import dataclasses
from dataclasses import dataclass
import json
import torch
from torch.fx import subgraph_rewriter
from torch.fx import Graph, GraphModule
import torch_xla
from torch_xla.core import xla_model as xm
from typing import List, Tuple, Dict, Any, Callable, Union, Optional

__all__ = ["mark_pattern"]


@dataclass
class PortTag:
  name: str  # Identify Patttern
  pos: int  # Arg/return position
  id: int  # Patten instance id
  is_input: bool = True  # If the tagged tensor is input/output
  attr: Dict = None  # Attribute of the pattern, only output has attr field


class TagSerializer(json.JSONEncoder):

  def default(self, obj):
    if dataclasses.is_dataclass(obj):
      return dataclasses.asdict(obj)
    return super().default(obj)


def tag_input(x, i, tag_name, total_input):
  if tag_name not in tag_input.counter:
    tag_input.counter[tag_name] = 0
  tag_count = tag_input.counter[tag_name]
  match_id = int(tag_count / total_input)
  print("tag_input name: {}, input pos: {}, match_id: {}".format(
      tag_name, i, match_id))
  torch_xla._XLAC._xla_add_tag(
      x,
      json.dumps(
          PortTag(tag_name, i, match_id, is_input=True), cls=TagSerializer))
  tag_input.counter[tag_name] += 1
  return x


tag_input.counter = dict()


def select_output(outputs, pos):
  return outputs[pos]


def tag_output(x, pos, tag_name, total_output, pattern_attrs=None):
  if tag_name not in tag_output.counter:
    tag_output.counter[tag_name] = 0
  if pattern_attrs is None:
    pattern_attrs = {}

  tag_count = tag_output.counter[tag_name]
  match_id = int(tag_count / total_output)
  print("tag_output name: {}, output pos {}, match_id: {}, attr: {}".format(
      tag_name, pos, match_id, pattern_attrs))
  torch_xla._XLAC._xla_add_tag(
      x,
      json.dumps(
          PortTag(tag_name, pos, match_id, is_input=False, attr=pattern_attrs),
          cls=TagSerializer,
      ),
  )
  tag_output.counter[tag_name] += 1
  return x


tag_output.counter = dict()


def get_pattern_node(pattern_name, pattern, args, pattern_attrs=None):
  pattern_ep = torch.export.export(pattern, args)
  n_inputs = len(pattern_ep.graph_signature.user_inputs)
  n_outputs = len(pattern_ep.graph_signature.user_outputs)
  print("pattern has {} inputs, {} outputs.".format(n_inputs, n_outputs))

  if pattern_attrs is None:
    pattern_attrs = {}

  new_g = Graph()
  tagged_placeholders = []
  n_tensor_inputs = sum(map(torch.is_tensor, args))
  i = 0
  for arg in args:
    if torch.is_tensor(arg):
      placeholder = new_g.placeholder("input_{}".format(i))
      tagged_placeholders.append(
          new_g.call_function(tag_input,
                              (placeholder, i, pattern_name, n_tensor_inputs)))
      i += 1
    else:
      tagged_placeholders.append(arg)

  if isinstance(pattern, torch.nn.Module):
    node_tagged = new_g.call_module("pattern")
  else:
    node_tagged = new_g.call_function(pattern, tuple(tagged_placeholders))

  output_nodes = []
  if n_outputs > 1:
    for pos in range(n_outputs):
      output_nodes.append(
          new_g.call_function(select_output, (node_tagged, pos)))
  else:
    output_nodes = [node_tagged]

  tagged_output_nodes = []
  for pos, output in enumerate(output_nodes):
    node_tagged_out = new_g.call_function(
        tag_output, (output, pos, pattern_name, n_outputs, pattern_attrs))
    tagged_output_nodes.append(node_tagged_out)

  node_out = new_g.output(tuple(tagged_output_nodes))
  replace_gm = GraphModule(dict(), new_g)
  return replace_gm


@dataclass
class ScalarAttrTracker:
  attr_name: str
  pattern_arg_pos: int
  transform: Callable[Any, Any] = lambda x: x
  inverse_transform: Callable[Any, Any] = lambda x: x
  source_targets: List[Tuple[Any,
                             Any]] = dataclasses.field(default_factory=list)

  def _is_equal(self, x: Any, y: Any):
    if type(x) != type(y):
      return False
    if type(x) in [int, str]:
      return x == y
    if isinstance(x, float):
      rel_tol = 1e-07
      abs_tol = 0.0
      return abs(x - y) <= max(rel_tol * max(abs(x), abs(y)), abs_tol)
    if isinstance(x, list):
      if len(x) != len(y):
        return False
      return all([self._is_equal(a, b) for a, b in zip(x, y)])

    raise Exception(f"Cannot compare type: {type(x)}")

  def track(self, *sources):
    for source in sources:
      target = self.transform(source)
      if not self._is_equal(self.inverse_transform(target), source):
        raise Exception(
            f"Invalid transform/inverse_transform for {self.attr_name}")
      self.source_targets.append([source, target])
    return self


@dataclass
class ScalarAttrLoc:
  tracker: ScalarAttrTracker
  node_name: str
  pos: int


def extract_and_replace_scalar_from_matched_pattern(
    pattern, matches: List[subgraph_rewriter.ReplacedPatterns],
    loc: ScalarAttrLoc):
  val = None
  for match in matches:
    for k, v in match.nodes_map.items():
      if k.name == loc.node_name:
        # print(str(v.args[loc.pos]))
        if loc.pos is not None:
          val = v.args[loc.pos]
        # TODO Handel kwarg
    assert val is not None
    pattern_arg_val = loc.tracker.inverse_transform(val)
    for n in match.replacements:
      if n.op == "call_function" and n.target == pattern:
        n.update_arg(loc.tracker.pattern_arg_pos, pattern_arg_val)
      if n.op == "call_function" and n.target == tag_output:
        attr_arg_idx = 4  # TODO: move to kwarg of the 'tag_ouptut'
        attr_dict = dict(n.args[attr_arg_idx])
        attr_dict[loc.tracker.attr_name] = pattern_arg_val
        n.update_arg(4, attr_dict)


def find_scalar_attr_loc(pattern, pattern_args, tracker: ScalarAttrTracker):
  scalar_loc_intersections = None
  for source, target in tracker.source_targets:
    track_args = list(pattern_args)
    track_args[tracker.pattern_arg_pos] = source
    ep = torch.export.export(pattern, tuple(track_args))

    scalar_locs = set()
    nodes = ep.graph_module.graph.nodes
    for n in nodes:
      for arg_pos, arg in enumerate(n.args):
        if type(arg) == type(target) and arg == target:
          scalar_locs.add((n.name, arg_pos))

    if scalar_loc_intersections is None:
      scalar_loc_intersections = scalar_locs
    else:
      scalar_loc_intersections = scalar_loc_intersections & scalar_locs

    if not scalar_loc_intersections:
      break

  if not scalar_loc_intersections:
    return None
  # Choose any occurrence as the attr provider
  node_name, arg_pos = scalar_loc_intersections.pop()
  return ScalarAttrLoc(tracker, node_name, arg_pos)


def eliminate_dangling_arg(graph: Graph):
  nodes_to_erase = []
  for n in graph.nodes:
    if n.op == "placeholder" and len(n.users) == 0:
      nodes_to_erase.append(n)
  for n in nodes_to_erase:
    graph.erase_node(n)


def mark_pattern(
    pattern_name: str,
    exported_ep: GraphModule,
    pattern: Union[Callable, GraphModule, torch.nn.Module],
    # Limit the pattern to not have kwargs
    pattern_args: Tuple,
    pattern_attrs: Optional[Dict[str, Any]] = None,
    scalar_attr_trackers: Optional[List[Union[ScalarAttrLoc,
                                              ScalarAttrTracker]]] = None,
):
  print("check whole graph")
  exported_ep.graph_module.graph.print_tabular()

  pattern_args = tuple(pattern_args)

  if isinstance(pattern, GraphModule):
    pattern_ep = pattern
  else:
    # pattern_ep = torch.export.export(pattern, pattern_args, pattern_kwargs)
    # FIXME: torch.export will generate a dangling input if there is constant
    pattern_ep = torch.export.export(pattern, pattern_args)
  # Build pattern replacement
  replace_pattern_gm = get_pattern_node(pattern_name, pattern, pattern_args,
                                        pattern_attrs)
  print("check replacement gm")
  replace_pattern_gm.graph.print_tabular()
  print("check pattern gm")
  pattern_ep.graph_module.graph.print_tabular()
  # Eliminate placeholder for const, which is dangling, and trgerring assertion in matching
  eliminate_dangling_arg(pattern_ep.graph_module.graph)
  matches = subgraph_rewriter.replace_pattern_with_filters(
      exported_ep.graph_module,
      pattern_ep.graph_module,
      replace_pattern_gm,
      ignore_literals=True,
  )
  print("check matches")
  print(matches)

  if scalar_attr_trackers:
    for tracker in scalar_attr_trackers:
      if isinstance(tracker, ScalarAttrLoc):
        loc = tracker
      else:
        loc = find_scalar_attr_loc(pattern, pattern_args, tracker)

      assert loc is not None
      print(loc)
      extract_and_replace_scalar_from_matched_pattern(pattern, matches, loc)

  exported_ep.graph_module.graph.print_tabular()
  return exported_ep
