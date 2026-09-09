// Rust neighbour-sampler for the reified graph — a fast drop-in for the Python
// BFS in reifm/sampling.py. The Python forward/backward is the bottleneck on
// CPU, so this barely helps the Mac; it matters on GPU (L4), where the forward
// is cheap and the sampler would otherwise starve the device. Same design as
// relational-transformer's `rustler`.
//
// Graphs are registered once (CSR + node_kind) in a process-global registry;
// fork workers inherit it copy-on-write. Per batch only the small query arrays
// cross the FFI boundary. Output mirrors sample_batch() exactly (compaction +
// `kept` when train_only_reached).

use std::collections::HashMap;
use std::sync::Mutex;

use once_cell::sync::Lazy;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

const ENTITY: i64 = 0;

struct Graph {
    rowptr: Vec<i64>,
    col: Vec<i64>,
    etype: Vec<i64>,
    node_kind: Vec<i64>,
    num_entities: i64,
    rel_offset: i64,
}

static REGISTRY: Lazy<Mutex<HashMap<i64, Graph>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[pyfunction]
fn register_graph(
    gidx: i64,
    rowptr: Vec<i64>,
    col: Vec<i64>,
    etype: Vec<i64>,
    node_kind: Vec<i64>,
    num_entities: i64,
    rel_offset: i64,
) {
    REGISTRY.lock().unwrap().insert(
        gidx,
        Graph { rowptr, col, etype, node_kind, num_entities, rel_offset },
    );
}

// BFS from `seeds` along outgoing CSR edges, fan-out capped, excluding
// `drop_node`. Returns (ordered unique nodes, edges as (src,dst,etype)).
fn sample_one(
    g: &Graph,
    seeds: &[i64],
    num_hops: usize,
    fanout: usize,
    drop_node: i64,
    max_nodes: usize,
    rng: &mut Pcg64Mcg,
) -> (Vec<i64>, Vec<(i64, i64, i64)>) {
    let mut nodes: Vec<i64> = Vec::new();
    let mut seen: HashMap<i64, i64> = HashMap::new(); // global id -> local idx
    for &s in seeds {
        if !seen.contains_key(&s) {
            seen.insert(s, nodes.len() as i64);
            nodes.push(s);
        }
    }
    let mut edges: Vec<(i64, i64, i64)> = Vec::new();
    let mut frontier: Vec<i64> = nodes.clone();
    for _ in 0..num_hops {
        let mut next: Vec<i64> = Vec::new();
        for &u in &frontier {
            if u == drop_node {
                continue;
            }
            let start = g.rowptr[u as usize];
            let end = g.rowptr[(u + 1) as usize];
            let deg = (end - start) as usize;
            if deg == 0 {
                continue;
            }
            // pick up to `fanout` neighbour offsets in [start,end)
            let picks: Vec<i64> = if deg <= fanout {
                (start..end).collect()
            } else {
                // rejection-sample `fanout` distinct offsets (fanout << deg),
                // avoiding an O(deg) permutation on million-degree hubs
                let mut chosen: Vec<i64> = Vec::with_capacity(fanout);
                while chosen.len() < fanout {
                    let off = start + rng.gen_range(0..deg as i64);
                    if !chosen.contains(&off) {
                        chosen.push(off);
                    }
                }
                chosen
            };
            for off in picks {
                let v = g.col[off as usize];
                if v == drop_node {
                    continue;
                }
                if seen.contains_key(&v) {
                    edges.push((u, v, g.etype[off as usize]));
                } else if max_nodes == 0 || nodes.len() < max_nodes {
                    // budget left: admit the new node and its edge
                    edges.push((u, v, g.etype[off as usize]));
                    seen.insert(v, nodes.len() as i64);
                    nodes.push(v);
                    next.push(v);
                }
                // else: node budget exhausted — drop edges into unseen nodes,
                // bounding worst-case subgraph size (and GPU memory) regardless
                // of hops/fanout, while keeping the already-built subgraph.
            }
        }
        frontier = next;
    }
    (nodes, edges)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_batch(
    py: Python<'_>,
    gidx: i64,
    anchors: Vec<i64>,
    rels: Vec<i64>,
    golds: Vec<i64>,
    drops: Vec<i64>, // raw fact idx, -1 = none
    num_hops: usize,
    fanout: usize,
    train_only_reached: bool,
    seed: u64,
    max_nodes: usize,
) -> PyResult<Py<PyDict>> {
    let reg = REGISTRY.lock().unwrap();
    let g = reg.get(&gidx).expect("graph not registered");
    let mut rng = Pcg64Mcg::new(seed as u128 | 0xa02bdbf7bb3c0a7);

    let mut x_kind = Vec::new();
    let (mut esrc, mut edst, mut etype) = (Vec::new(), Vec::new(), Vec::new());
    let mut batch = Vec::new();
    let (mut seed_ent, mut seed_rel) = (Vec::new(), Vec::new());
    let (mut cand_local, mut cand_query, mut cand_global) = (Vec::new(), Vec::new(), Vec::new());
    let mut gold_pos = Vec::new();
    let mut kept = Vec::new();
    let b_total = anchors.len();
    let mut offset: i64 = 0;

    for b in 0..b_total {
        let anchor = anchors[b];
        let rel_node = rels[b] + g.rel_offset;
        let drop_node = if drops[b] >= 0 { g.num_entities + drops[b] } else { -1 };
        let (nodes, edges) = sample_one(g, &[anchor, rel_node], num_hops, fanout, drop_node, max_nodes, &mut rng);
        // local index map
        let mut g2l: HashMap<i64, i64> = HashMap::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            g2l.insert(n, i as i64);
        }
        // candidate entities
        let mut ents: Vec<i64> = nodes
            .iter()
            .enumerate()
            .filter(|(_, &n)| g.node_kind[n as usize] == ENTITY)
            .map(|(i, _)| i as i64)
            .collect();
        let mut gp: i64 = -1;
        if let Some(&gl) = g2l.get(&golds[b]) {
            if !ents.contains(&gl) {
                ents.push(gl);
            }
            gp = ents.iter().position(|&e| e == gl).unwrap() as i64;
        }
        if train_only_reached && gp < 0 {
            continue;
        }
        let bc = kept.len() as i64;
        kept.push(b as i64);
        for &n in &nodes {
            x_kind.push(g.node_kind[n as usize]);
            batch.push(bc);
        }
        for (s, d, et) in &edges {
            esrc.push(g2l[s] + offset);
            edst.push(g2l[d] + offset);
            etype.push(*et);
        }
        seed_ent.push(g2l[&anchor] + offset);
        seed_rel.push(g2l[&rel_node] + offset);
        for &e in &ents {
            cand_local.push(e + offset);
            cand_query.push(bc);
            cand_global.push(nodes[e as usize]);
        }
        gold_pos.push(gp);
        offset += nodes.len() as i64;
    }
    let b = kept.len();
    let d = PyDict::new_bound(py);
    d.set_item("x_kind", x_kind)?;
    d.set_item("edge_src", esrc)?;
    d.set_item("edge_dst", edst)?;
    d.set_item("edge_type", etype)?;
    d.set_item("batch", batch)?;
    d.set_item("seed_ent_local", seed_ent)?;
    d.set_item("seed_rel_local", seed_rel)?;
    d.set_item("cand_local", cand_local)?;
    d.set_item("cand_query", cand_query)?;
    d.set_item("cand_global", cand_global)?;
    d.set_item("gold_pos", gold_pos)?;
    d.set_item("kept", kept)?;
    d.set_item("B", b)?;
    d.set_item("B_total", b_total)?;
    Ok(d.into())
}

// ===========================================================================
// On-the-fly reification: keep only the COMPACT original multigraph and
// generate the reified ego-subgraph lazily during BFS (no materialised reified
// graph). Reified id layout matches reify.py / onthefly.py exactly:
//   entity e : e            in [0, E)
//   fact   f : E + f        in [E, E+F)
//   rel    r : E + F + r     in [E+F, E+F+R)
// Reified meta-relation etypes (reify.py): HAS_SUBJECT=0 SUBJECT_OF=1
//   HAS_OBJECT=2 OBJECT_OF=3 HAS_TYPE=4 HAS_INSTANCE=5.  Node kinds: ENTITY=0
//   FACT=1 REL_TYPE=2.  Memory: ~6F ints vs ~30F for the materialised path.
// ===========================================================================
const FACT: i64 = 1;
const REL_TYPE: i64 = 2;
const HAS_SUBJECT: i64 = 0;
const SUBJECT_OF: i64 = 1;
const HAS_OBJECT: i64 = 2;
const OBJECT_OF: i64 = 3;
const HAS_TYPE: i64 = 4;
const HAS_INSTANCE: i64 = 5;

struct OrigGraph {
    head_rowptr: Vec<i64>,
    head_col: Vec<i64>, // entity -> incident fact ids (as head)
    tail_rowptr: Vec<i64>,
    tail_col: Vec<i64>, // entity -> incident fact ids (as tail)
    rel_rowptr: Vec<i64>,
    rel_col: Vec<i64>, // relation -> instance fact ids
    fact_head: Vec<i64>,
    fact_tail: Vec<i64>,
    fact_type: Vec<i64>,
    num_entities: i64,
    num_facts: i64,
    rel_offset: i64, // = E + F
}

static OTF_REGISTRY: Lazy<Mutex<HashMap<i64, OrigGraph>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

impl OrigGraph {
    #[inline]
    fn kind(&self, u: i64) -> i64 {
        if u < self.num_entities {
            ENTITY
        } else if u < self.rel_offset {
            FACT
        } else {
            REL_TYPE
        }
    }

    #[inline]
    fn degree(&self, u: i64) -> i64 {
        if u < self.num_entities {
            let e = u as usize;
            (self.head_rowptr[e + 1] - self.head_rowptr[e])
                + (self.tail_rowptr[e + 1] - self.tail_rowptr[e])
        } else if u < self.rel_offset {
            3
        } else {
            let r = (u - self.rel_offset) as usize;
            self.rel_rowptr[r + 1] - self.rel_rowptr[r]
        }
    }

    // i-th outgoing reified neighbour of u as (neighbour_global, etype).
    #[inline]
    fn neighbor_at(&self, u: i64, i: i64) -> (i64, i64) {
        let e_off = self.num_entities;
        if u < e_off {
            let e = u as usize;
            let hd = self.head_rowptr[e + 1] - self.head_rowptr[e];
            if i < hd {
                let f = self.head_col[(self.head_rowptr[e] + i) as usize];
                (e_off + f, SUBJECT_OF)
            } else {
                let j = i - hd;
                let f = self.tail_col[(self.tail_rowptr[e] + j) as usize];
                (e_off + f, OBJECT_OF)
            }
        } else if u < self.rel_offset {
            let f = (u - e_off) as usize;
            match i {
                0 => (self.fact_head[f], HAS_SUBJECT),
                1 => (self.fact_tail[f], HAS_OBJECT),
                _ => (self.rel_offset + self.fact_type[f], HAS_TYPE),
            }
        } else {
            let r = (u - self.rel_offset) as usize;
            let f = self.rel_col[(self.rel_rowptr[r] + i) as usize];
            (e_off + f, HAS_INSTANCE)
        }
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn register_graph_original(
    gidx: i64,
    head_rowptr: Vec<i64>,
    head_col: Vec<i64>,
    tail_rowptr: Vec<i64>,
    tail_col: Vec<i64>,
    rel_rowptr: Vec<i64>,
    rel_col: Vec<i64>,
    fact_head: Vec<i64>,
    fact_tail: Vec<i64>,
    fact_type: Vec<i64>,
    num_entities: i64,
    num_facts: i64,
) {
    OTF_REGISTRY.lock().unwrap().insert(
        gidx,
        OrigGraph {
            head_rowptr,
            head_col,
            tail_rowptr,
            tail_col,
            rel_rowptr,
            rel_col,
            fact_head,
            fact_tail,
            fact_type,
            num_entities,
            num_facts,
            rel_offset: num_entities + num_facts,
        },
    );
}

// Lazy-reified BFS: identical contract to sample_one() but neighbours are
// generated on the fly from the compact original adjacency.
fn sample_one_otf(
    g: &OrigGraph,
    seeds: &[i64],
    num_hops: usize,
    fanout: usize,
    drop_node: i64,
    max_nodes: usize,
    rng: &mut Pcg64Mcg,
) -> (Vec<i64>, Vec<(i64, i64, i64)>) {
    let mut nodes: Vec<i64> = Vec::new();
    let mut seen: HashMap<i64, i64> = HashMap::new();
    for &s in seeds {
        if !seen.contains_key(&s) {
            seen.insert(s, nodes.len() as i64);
            nodes.push(s);
        }
    }
    let mut edges: Vec<(i64, i64, i64)> = Vec::new();
    let mut frontier: Vec<i64> = nodes.clone();
    for _ in 0..num_hops {
        let mut next: Vec<i64> = Vec::new();
        for &u in &frontier {
            if u == drop_node {
                continue;
            }
            let deg = g.degree(u) as usize;
            if deg == 0 {
                continue;
            }
            // pick up to `fanout` distinct neighbour indices in [0,deg)
            let picks: Vec<i64> = if deg <= fanout {
                (0..deg as i64).collect()
            } else {
                let mut chosen: Vec<i64> = Vec::with_capacity(fanout);
                while chosen.len() < fanout {
                    let off = rng.gen_range(0..deg as i64);
                    if !chosen.contains(&off) {
                        chosen.push(off);
                    }
                }
                chosen
            };
            for idx in picks {
                let (v, et) = g.neighbor_at(u, idx);
                if v == drop_node {
                    continue;
                }
                if seen.contains_key(&v) {
                    edges.push((u, v, et));
                } else if max_nodes == 0 || nodes.len() < max_nodes {
                    edges.push((u, v, et));
                    seen.insert(v, nodes.len() as i64);
                    nodes.push(v);
                    next.push(v);
                }
            }
        }
        frontier = next;
    }
    (nodes, edges)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_batch_original(
    py: Python<'_>,
    gidx: i64,
    anchors: Vec<i64>,
    rels: Vec<i64>,
    golds: Vec<i64>,
    drops: Vec<i64>,
    num_hops: usize,
    fanout: usize,
    train_only_reached: bool,
    seed: u64,
    max_nodes: usize,
) -> PyResult<Py<PyDict>> {
    let reg = OTF_REGISTRY.lock().unwrap();
    let g = reg.get(&gidx).expect("original graph not registered");
    let mut rng = Pcg64Mcg::new(seed as u128 | 0xa02bdbf7bb3c0a7);

    let mut x_kind = Vec::new();
    let (mut esrc, mut edst, mut etype) = (Vec::new(), Vec::new(), Vec::new());
    let mut batch = Vec::new();
    let (mut seed_ent, mut seed_rel) = (Vec::new(), Vec::new());
    let (mut cand_local, mut cand_query, mut cand_global) = (Vec::new(), Vec::new(), Vec::new());
    let mut gold_pos = Vec::new();
    let mut kept = Vec::new();
    let b_total = anchors.len();
    let mut offset: i64 = 0;

    for b in 0..b_total {
        let anchor = anchors[b];
        let rel_node = rels[b] + g.rel_offset;
        let drop_node = if drops[b] >= 0 { g.num_entities + drops[b] } else { -1 };
        let (nodes, edges) =
            sample_one_otf(g, &[anchor, rel_node], num_hops, fanout, drop_node, max_nodes, &mut rng);
        let mut g2l: HashMap<i64, i64> = HashMap::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            g2l.insert(n, i as i64);
        }
        let mut ents: Vec<i64> = nodes
            .iter()
            .enumerate()
            .filter(|(_, &n)| n < g.num_entities) // ENTITY kind
            .map(|(i, _)| i as i64)
            .collect();
        let mut gp: i64 = -1;
        if let Some(&gl) = g2l.get(&golds[b]) {
            if !ents.contains(&gl) {
                ents.push(gl);
            }
            gp = ents.iter().position(|&e| e == gl).unwrap() as i64;
        }
        if train_only_reached && gp < 0 {
            continue;
        }
        let bc = kept.len() as i64;
        kept.push(b as i64);
        for &n in &nodes {
            x_kind.push(g.kind(n));
            batch.push(bc);
        }
        for (s, d, et) in &edges {
            esrc.push(g2l[s] + offset);
            edst.push(g2l[d] + offset);
            etype.push(*et);
        }
        seed_ent.push(g2l[&anchor] + offset);
        seed_rel.push(g2l[&rel_node] + offset);
        for &e in &ents {
            cand_local.push(e + offset);
            cand_query.push(bc);
            cand_global.push(nodes[e as usize]);
        }
        gold_pos.push(gp);
        offset += nodes.len() as i64;
    }
    let b = kept.len();
    let d = PyDict::new_bound(py);
    d.set_item("x_kind", x_kind)?;
    d.set_item("edge_src", esrc)?;
    d.set_item("edge_dst", edst)?;
    d.set_item("edge_type", etype)?;
    d.set_item("batch", batch)?;
    d.set_item("seed_ent_local", seed_ent)?;
    d.set_item("seed_rel_local", seed_rel)?;
    d.set_item("cand_local", cand_local)?;
    d.set_item("cand_query", cand_query)?;
    d.set_item("cand_global", cand_global)?;
    d.set_item("gold_pos", gold_pos)?;
    d.set_item("kept", kept)?;
    d.set_item("B", b)?;
    d.set_item("B_total", b_total)?;
    Ok(d.into())
}

// Entity-task sampler (RelBench): BFS from a SINGLE seed entity (no relation
// seed, no gold/candidates), returning the per-node global ids so the caller can
// attach semantic features. Uses the on-the-fly reified BFS — the fast drop-in
// for rdb_task.sample_entity_batch's Python loop (the multi-DB bottleneck).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sample_entity_batch_original(
    py: Python<'_>,
    gidx: i64,
    anchors: Vec<i64>,
    num_hops: usize,
    fanout: usize,
    seed: u64,
    max_nodes: usize,
) -> PyResult<Py<PyDict>> {
    let reg = OTF_REGISTRY.lock().unwrap();
    let g = reg.get(&gidx).expect("original graph not registered");
    let mut rng = Pcg64Mcg::new(seed as u128 | 0xa02bdbf7bb3c0a7);

    let mut x_kind = Vec::new();
    let (mut esrc, mut edst, mut etype) = (Vec::new(), Vec::new(), Vec::new());
    let mut batch = Vec::new();
    let mut seed_ent = Vec::new();
    let mut node_global = Vec::new();
    let mut offset: i64 = 0;

    for (b, &anchor) in anchors.iter().enumerate() {
        let (nodes, edges) =
            sample_one_otf(g, &[anchor], num_hops, fanout, -1, max_nodes, &mut rng);
        let mut g2l: HashMap<i64, i64> = HashMap::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            g2l.insert(n, i as i64);
        }
        for &n in &nodes {
            x_kind.push(g.kind(n));
            batch.push(b as i64);
            node_global.push(n);
        }
        for (s, d, et) in &edges {
            esrc.push(g2l[s] + offset);
            edst.push(g2l[d] + offset);
            etype.push(*et);
        }
        seed_ent.push(g2l[&anchor] + offset);
        offset += nodes.len() as i64;
    }
    let d = PyDict::new_bound(py);
    d.set_item("x_kind", x_kind)?;
    d.set_item("edge_src", esrc)?;
    d.set_item("edge_dst", edst)?;
    d.set_item("edge_type", etype)?;
    d.set_item("batch", batch)?;
    d.set_item("seed_ent_local", seed_ent)?;
    d.set_item("node_global", node_global)?;
    Ok(d.into())
}

#[pymodule]
fn reifm_sampler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_graph, m)?)?;
    m.add_function(wrap_pyfunction!(sample_batch, m)?)?;
    m.add_function(wrap_pyfunction!(register_graph_original, m)?)?;
    m.add_function(wrap_pyfunction!(sample_batch_original, m)?)?;
    m.add_function(wrap_pyfunction!(sample_entity_batch_original, m)?)?;
    Ok(())
}
