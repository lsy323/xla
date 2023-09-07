import dataclasses
from dataclasses import dataclass
import json
import torch
from torch.fx import subgraph_rewriter
from torch.fx import Graph, GraphModule
import torch_xla
from torch_xla.core import xla_model as xm
from typing import List, Tuple, Dict, Any, Callable, Union

__all__ = ['mark_pattern']

@dataclass
class PortTag:
    name: str  # Identify Patttern
    pos: int  # Arg/return position
    id: int  # Patten instance id
    is_input: bool = True


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
    print(
        "tag_input name: {}, input pos: {}, match_id: {}".format(tag_name, i, match_id)
    )
    torch_xla._XLAC._xla_add_tag(
        x, json.dumps(PortTag(tag_name, i, match_id, is_input=True), cls=TagSerializer)
    )
    tag_input.counter[tag_name] += 1
    return x


tag_input.counter = dict()

def select_output(outputs, pos):
    return outputs[pos]

def tag_output(x, pos, tag_name, total_output):
    if tag_name not in tag_output.counter:
        tag_output.counter[tag_name] = 0
    tag_count = tag_output.counter[tag_name]
    match_id = int(tag_count / total_output)
    print(
        "tag_output name: {}, output pos {}, match_id: {}".format(tag_name, pos, match_id)
    )
    torch_xla._XLAC._xla_add_tag(
        x, json.dumps(PortTag(tag_name, pos, match_id, is_input=False), cls=TagSerializer)
    )
    tag_output.counter[tag_name] += 1
    return x


tag_output.counter = dict()


def get_pattern_node(pattern, args):
    pattern_ep = torch.export.export(pattern, args)
    n_inputs = len(pattern_ep.graph_signature.user_inputs)
    n_outputs = len(pattern_ep.graph_signature.user_outputs)
    print("pattern has {} inputs, {} outputs.".format(n_inputs, n_outputs))

    new_g = Graph()
    placeholders = []
    for i in range(n_inputs):
        placeholders.append(new_g.placeholder("input_{}".format(i)))

    tagged_placeholders = []
    for i in range(n_inputs):
        tagged_placeholders.append(
            new_g.call_function(
                tag_input, (placeholders[i], i, pattern.__name__, n_inputs)
                # tag_input, (placeholders[i], i, "pattern", n_inputs)
            )
        )

    node_tagged = new_g.call_function(pattern, tuple(tagged_placeholders))

    output_nodes = []
    if n_outputs > 1:
        for pos in range(n_outputs):
            output_nodes.append(new_g.call_function(select_output,(node_tagged, pos)))
    else:
        output_nodes = [node_tagged]     

    tagged_output_nodes = []
    for pos, output in enumerate(output_nodes):
        node_tagged_out = new_g.call_function(
            tag_output, (output, pos, pattern.__name__, n_outputs)
            # tag_output, (output, pos, "pattern", n_outputs)
        )
        tagged_output_nodes.append(node_tagged_out)

    node_out = new_g.output(tuple(tagged_output_nodes))
    replace_gm = GraphModule(dict(), new_g)
    return replace_gm


def mark_pattern(
    exported_ep: GraphModule, pattern: Union[Callable, GraphModule], pattern_args: Tuple
):
    if isinstance(pattern, GraphModule):
        pattern_ep = pattern
    else:
        pattern_ep = torch.export.export(pattern, pattern_args)
    # Build pattern replacement
    replace_pattern_gm = get_pattern_node(pattern, pattern_args)
    print("check replacement gm")
    replace_pattern_gm.graph.print_tabular()
    print("check pattern gm")
    pattern_ep.graph_module.graph.print_tabular()
    matches = subgraph_rewriter.replace_pattern(
        exported_ep.graph_module, pattern_ep.graph_module, replace_pattern_gm
    )
    return exported_ep
