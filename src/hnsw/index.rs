use crate::ivf::{squared_l2_distance, EmbeddingDim, Embeddings};
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub(crate) struct HnswIndex {
    dim: EmbeddingDim,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    entry_point: u32,
    max_level: u32,
    node_levels: Vec<u32>,
    layer_links: Vec<Vec<Vec<u32>>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct HnswBuildConfig {
    pub(crate) m: usize,
    pub(crate) ef_construction: usize,
    pub(crate) ef_search: usize,
    pub(crate) seed: u64,
}

#[derive(Serialize, Deserialize)]
struct HnswIndexPayload {
    dim: u32,
    m: u32,
    ef_construction: u32,
    ef_search: u32,
    entry_point: u32,
    max_level: u32,
    node_levels: Vec<u32>,
    layer_links: Vec<Vec<Vec<u32>>>,
}

impl HnswIndex {
    pub(crate) fn dim(&self) -> usize {
        self.dim.as_usize()
    }

    pub(crate) fn ef_search(&self) -> usize {
        self.ef_search
    }

    pub(crate) fn candidate_rows(&self, query: &[f32], embeddings: &[f32], ef_search: usize) -> Vec<u32> {
        let node_count = self.node_count();
        let embedding_count_required = match node_count.checked_mul(self.dim()) {
            Some(required) => required,
            None => return Vec::new(),
        };
        if embeddings.len() < embedding_count_required {
            return Vec::new();
        }
        if query.len() != self.dim() {
            return Vec::new();
        }
        if node_count == 0 {
            return Vec::new();
        }

        let ef_search = ef_search.max(1).min(self.node_count());
        let mut entry = self.entry_point as usize;
        for layer in (1..=self.max_level as usize).rev() {
            entry = self.greedy_search_layer(query, embeddings, layer, entry);
        }
        self.beam_search_layer(query, embeddings, 0, entry, ef_search)
    }

    pub(crate) fn to_json_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let payload = HnswIndexPayload {
            dim: self.dim.as_u32(),
            m: self.m as u32,
            ef_construction: self.ef_construction as u32,
            ef_search: self.ef_search as u32,
            entry_point: self.entry_point,
            max_level: self.max_level,
            node_levels: self.node_levels.clone(),
            layer_links: self.layer_links.clone(),
        };
        Ok(serde_json::to_vec(&payload)?)
    }

    pub(crate) fn from_json_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let payload: HnswIndexPayload = serde_json::from_slice(bytes)?;
        let dim = EmbeddingDim::new(payload.dim as usize)?;
        let m = payload.m as usize;
        let ef_construction = payload.ef_construction as usize;
        let ef_search = payload.ef_search as usize;
        if m == 0 {
            return Err("HNSW index M must be > 0".into());
        }
        if ef_construction == 0 {
            return Err("HNSW index ef_construction must be > 0".into());
        }
        if ef_search == 0 {
            return Err("HNSW index ef_search must be > 0".into());
        }
        if payload.layer_links.len() != payload.max_level as usize + 1 {
            return Err("HNSW index layer count mismatch".into());
        }
        let node_count = payload.node_levels.len();
        let entry_point = payload.entry_point as usize;
        if entry_point >= node_count {
            return Err("HNSW index entry point is out of bounds".into());
        }
        if payload.layer_links.iter().any(|layer| layer.len() != node_count) {
            return Err("HNSW index layer length mismatch".into());
        }
        for (idx, level) in payload.node_levels.iter().enumerate() {
            if (*level as usize) > payload.max_level as usize {
                return Err(format!(
                    "HNSW node {} has level {} above max {}",
                    idx,
                    level,
                    payload.max_level
                )
                .into());
            }
        }
        for layer in 0..=payload.max_level as usize {
            for node in 0..node_count {
                for &neighbor in &payload.layer_links[layer][node] {
                    if neighbor as usize >= node_count {
                        return Err("HNSW index contains out-of-bounds neighbor".into());
                    }
                    if neighbor == node as u32 {
                        return Err("HNSW index contains self-loop".into());
                    }
                }
            }
        }
        Ok(Self {
            dim,
            m,
            ef_construction,
            ef_search,
            entry_point: payload.entry_point,
            max_level: payload.max_level,
            node_levels: payload.node_levels,
            layer_links: payload.layer_links,
        })
    }

    fn node_count(&self) -> usize {
        self.node_levels.len()
    }

    fn is_in_layer(&self, node: usize, layer: usize) -> bool {
        self.node_levels[node] as usize >= layer
    }

    fn vector_of<'a>(&self, embeddings: &'a [f32], node: usize) -> &'a [f32] {
        let start = node * self.dim();
        &embeddings[start..start + self.dim()]
    }

    fn node_distance(&self, query: &[f32], embeddings: &[f32], node: usize) -> f32 {
        squared_l2_distance(query, self.vector_of(embeddings, node))
    }

    fn greedy_search_layer(
        &self,
        query: &[f32],
        embeddings: &[f32],
        layer: usize,
        start: usize,
    ) -> usize {
        let mut current = start;
        let mut current_distance = self.node_distance(query, embeddings, current);

        loop {
            let mut best = current;
            let mut best_distance = current_distance;
            for &neighbor in &self.layer_links[layer][current] {
                if !self.is_in_layer(neighbor as usize, layer) {
                    continue;
                }
                let distance = self.node_distance(query, embeddings, neighbor as usize);
                if distance < best_distance {
                    best_distance = distance;
                    best = neighbor as usize;
                }
            }
            if best == current {
                return current;
            }
            current = best;
            current_distance = best_distance;
        }
    }

    fn beam_search_layer(
        &self,
        query: &[f32],
        embeddings: &[f32],
        layer: usize,
        entry: usize,
        ef: usize,
    ) -> Vec<u32> {
        let mut visited = vec![false; self.node_count()];
        let mut candidates = Vec::new();
        let mut selected = Vec::new();

        visited[entry] = true;
        let start_distance = self.node_distance(query, embeddings, entry);
        insert_by_distance(&mut candidates, (entry as u32, start_distance));
        insert_by_distance(&mut selected, (entry as u32, start_distance));

        let mut idx = 0usize;
        while idx < candidates.len() {
            let (node, dist) = candidates[idx];
            idx += 1;

            if selected.len() >= ef && dist > selected[selected.len() - 1].1 {
                break;
            }

            for &neighbor in &self.layer_links[layer][node as usize] {
                let neighbor = neighbor as usize;
                if visited[neighbor] || !self.is_in_layer(neighbor, layer) {
                    continue;
                }
                visited[neighbor] = true;

                let distance = self.node_distance(query, embeddings, neighbor);
                insert_by_distance(&mut candidates, (neighbor as u32, distance));
                insert_by_distance(&mut selected, (neighbor as u32, distance));

                if selected.len() > ef {
                    selected.truncate(ef);
                }
            }
        }

        selected
            .into_iter()
            .map(|(node, _)| node)
            .collect()
    }
}

pub(crate) fn build_hnsw_index(
    embeddings: &Embeddings,
    config: HnswBuildConfig,
) -> Result<HnswIndex, Box<dyn std::error::Error>> {
    if embeddings.row_count() == 0 {
        return Err("Cannot build HNSW index with zero vectors".into());
    }
    if config.m == 0 {
        return Err("m must be > 0".into());
    }
    if config.ef_construction == 0 {
        return Err("ef_construction must be > 0".into());
    }
    if config.ef_search == 0 {
        return Err("ef_search must be > 0".into());
    }

    let node_count = embeddings.row_count();
    let dim = embeddings.dim().as_usize();
    let data = embeddings.data();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let level_p = (1.0 / (config.m as f64)).max(0.1).min(0.9);

    let mut node_levels = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        node_levels.push(random_level(&mut rng, level_p) as u32);
    }

    let max_level = node_levels.iter().copied().map(|l| l as usize).max().unwrap_or(0);
    let mut layer_links = vec![vec![Vec::new(); node_count]; max_level + 1];

    for layer in 0..=max_level {
        let active_nodes: Vec<usize> = node_levels
            .iter()
            .enumerate()
            .filter_map(|(idx, level)| {
                if *level as usize >= layer {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        if active_nodes.is_empty() {
            continue;
        }

        for &node in active_nodes.iter() {
            let start = node * dim;
            let end = start + dim;
            let mut distances: Vec<(f32, u32)> = active_nodes
                .iter()
                .filter_map(|&other| {
                    if other == node {
                        return None;
                    }
                    let other_start = other * dim;
                    let other_end = other_start + dim;
                    let dist = squared_l2_distance(&data[start..end], &data[other_start..other_end]);
                    Some((dist, other as u32))
                })
                .collect();

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let neighbor_count = distances.len().min(config.ef_construction).min(config.m);
            layer_links[layer][node] = distances
                .into_iter()
                .take(neighbor_count)
                .map(|(_, idx)| idx)
                .collect();
        }
    }

    for layer in 0..=max_level {
        for node in 0..node_count {
            layer_links[layer][node].sort_unstable();
            layer_links[layer][node].dedup();
            if layer_links[layer][node].len() > config.m {
                layer_links[layer][node].truncate(config.m);
            }
        }
    }

    for layer in 0..=max_level {
        let current_layer = layer_links[layer].clone();
        for (node, neighbors) in current_layer.iter().enumerate() {
            for &neighbor in neighbors.iter() {
                let reverse = &mut layer_links[layer][neighbor as usize];
                if !reverse.contains(&(node as u32)) {
                    reverse.push(node as u32);
                    reverse.sort_unstable();
                    reverse.dedup();
                    if reverse.len() > config.m {
                        reverse.truncate(config.m);
                    }
                }
            }
        }
    }

    let entry_point = if node_count == 0 {
        0
    } else {
        node_levels
            .iter()
            .enumerate()
            .filter_map(|(idx, level)| {
                if *level as usize == max_level {
                    Some(idx)
                } else {
                    None
                }
            })
            .next()
            .unwrap_or(0) as u32
    };

    Ok(HnswIndex {
        dim: embeddings.dim(),
        m: config.m,
        ef_construction: config.ef_construction,
        ef_search: config.ef_search,
        entry_point,
        max_level: max_level as u32,
        node_levels,
        layer_links,
    })
}

fn random_level(rng: &mut rand::rngs::StdRng, level_p: f64) -> usize {
    let mut level = 0usize;
    while rng.gen_range(0.0..1.0) < level_p {
        level += 1;
    }
    level
}

fn insert_by_distance(items: &mut Vec<(u32, f32)>, item: (u32, f32)) {
    let idx = items
        .binary_search_by(|(_, d)| d.partial_cmp(&item.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(|x| x);
    items.insert(idx, item);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ivf::EmbeddingDim;

    #[test]
    fn test_hnsw_index_json_roundtrip() {
        let dim = EmbeddingDim::new(2).unwrap();
        let index = HnswIndex {
            dim,
            m: 2,
            ef_construction: 5,
            ef_search: 4,
            entry_point: 0,
            max_level: 0,
            node_levels: vec![0, 0, 0],
            layer_links: vec![vec![
                vec![1],
                vec![0],
                vec![],
            ]],
        };

        let serialized = index.to_json_bytes().unwrap();
        let restored = HnswIndex::from_json_bytes(&serialized).unwrap();

        assert_eq!(restored.dim(), index.dim());
        assert_eq!(restored.m, index.m);
        assert_eq!(restored.node_levels, index.node_levels);
        assert_eq!(restored.layer_links, index.layer_links);
    }

    #[test]
    fn test_build_and_search_hnsw_index() {
        let data = Embeddings::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 10.0, 10.0],
            EmbeddingDim::new(2).unwrap(),
        )
        .unwrap();
        let config = HnswBuildConfig {
            m: 2,
            ef_construction: 4,
            ef_search: 4,
            seed: 42,
        };
        let index = build_hnsw_index(&data, config).unwrap();
        let candidates = index.candidate_rows(&[0.0, 0.0], data.data(), 4);
        assert!(candidates.contains(&0));
    }
}
