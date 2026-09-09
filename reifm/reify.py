"""Reification of a knowledge graph into a fixed-vocabulary graph.

Every fact (h, r, t) of the original multi-relational graph becomes a *fact
node* linked to its subject/object entities and to a *relation-type node*.
The resulting graph uses a fixed vocabulary of 6 meta-relations whatever the
source dataset, which is what makes any GNN trained on it dataset-agnostic:

    fact --HAS_SUBJECT--> entity_h     entity_h --SUBJECT_OF--> fact
    fact --HAS_OBJECT --> entity_t     entity_t --OBJECT_OF --> fact
    fact --HAS_TYPE   --> rel_type_r   rel_type_r --HAS_INSTANCE--> fact

Node layout (single homogeneous id space):
    [0, num_entities)                              entity nodes
    [num_entities, num_entities + num_facts)       fact nodes
    [num_entities + num_facts, ... + num_rel_types) relation-type nodes

Edge layout: 6 contiguous blocks of num_facts edges, one per meta-relation in
the order above; edge i of every block belongs to fact i.
"""

import torch
from torch_geometric.data import Data

NUM_META_RELATIONS = 6
HAS_SUBJECT, SUBJECT_OF, HAS_OBJECT, OBJECT_OF, HAS_TYPE, HAS_INSTANCE = range(6)

# node kinds
ENTITY, FACT, REL_TYPE = 0, 1, 2


def reify(edge_index: torch.Tensor, edge_type: torch.Tensor, num_nodes: int,
          num_relations: int) -> Data:
    """Build the reified graph from a (edge_index, edge_type) multigraph.

    `edge_index`/`edge_type` must be the *fact* edges only (no target/query
    edges), so that no test fact leaks into the message-passing graph.
    """
    num_facts = edge_index.size(1)
    fact_off = num_nodes
    rel_off = num_nodes + num_facts

    heads = edge_index[0]
    tails = edge_index[1]
    facts = torch.arange(num_facts) + fact_off
    rels = edge_type + rel_off

    src = torch.cat([facts, heads, facts, tails, facts, rels])
    dst = torch.cat([heads, facts, tails, facts, rels, facts])
    etype = torch.cat([
        torch.full((num_facts,), HAS_SUBJECT),
        torch.full((num_facts,), SUBJECT_OF),
        torch.full((num_facts,), HAS_OBJECT),
        torch.full((num_facts,), OBJECT_OF),
        torch.full((num_facts,), HAS_TYPE),
        torch.full((num_facts,), HAS_INSTANCE),
    ])

    total_nodes = num_nodes + num_facts + num_relations
    node_kind = torch.cat([
        torch.full((num_nodes,), ENTITY),
        torch.full((num_facts,), FACT),
        torch.full((num_relations,), REL_TYPE),
    ])

    return Data(
        edge_index=torch.stack([src, dst]),
        edge_type=etype,
        num_nodes=total_nodes,
        node_kind=node_kind,
        num_entities=num_nodes,
        num_facts=num_facts,
        num_rel_types=num_relations,
        rel_offset=rel_off,
        fact_offset=fact_off,
        # original triple per fact node
        fact_head=edge_index[0],
        fact_tail=edge_index[1],
        fact_rel=edge_type,
    )
