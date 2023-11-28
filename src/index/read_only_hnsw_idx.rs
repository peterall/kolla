use std::collections::{BinaryHeap, HashSet};

use fixedbitset::FixedBitSet;
use rkyv::{Archive, Deserialize, Serialize};

use crate::core::{metrics, neighbor::Neighbor, node::IdxType};

use super::hnsw_idx::HNSWIndex;

#[derive(Archive, Serialize, Deserialize, Default)]
#[archive(check_bytes)]
pub struct ReadOnlyHNSWIndex<T> {
    dimension: usize, // dimension
    n_constructed_items: usize,
    max_item: usize,
    cur_level: usize,                  //current level
    id2neighbor: Vec<Vec<Vec<usize>>>, //neight_id from level 1 to level _max_level
    id2neighbor0: Vec<Vec<usize>>,     //neigh_id at level 0
    nodes: Vec<(Vec<f32>, T)>,         // data saver
    root_id: usize,                    //root of hnsw
    has_removed: bool,
    ef_search: usize,           // num of max candidates when searching
    delete_ids: HashSet<usize>, //save deleted ids
    mt: metrics::Metric,        //compute metrics
}

// convert from HNSWIndex to ReadOnlyHNSWIndex
impl<T: IdxType> From<&HNSWIndex<f32, T>> for ReadOnlyHNSWIndex<T> {
    fn from(hnsw: &HNSWIndex<f32, T>) -> Self {
        let id2neighbor = hnsw
            ._id2neighbor
            .iter()
            .map(|l0| l0.iter().map(|l1| l1.read().unwrap().clone()).collect())
            .collect();

        let id2neighbor0 = hnsw
            ._id2neighbor0
            .iter()
            .map(|l0| l0.read().unwrap().clone())
            .collect();
        let nodes = hnsw
            ._nodes
            .iter()
            .map(|n| {
                (
                    n.as_ref().vectors().clone(),
                    n.as_ref().idx().as_ref().unwrap().clone(),
                )
            })
            .collect();

        ReadOnlyHNSWIndex {
            dimension: hnsw._dimension,
            n_constructed_items: hnsw._n_constructed_items,
            max_item: hnsw._max_item,
            cur_level: hnsw._cur_level,
            id2neighbor,
            id2neighbor0,
            nodes,
            root_id: hnsw._root_id,
            has_removed: hnsw._has_removed,
            ef_search: hnsw._ef_search,
            delete_ids: hnsw._delete_ids.clone(),
            mt: hnsw.mt,
        }
    }
}

impl<T: Archive> ArchivedReadOnlyHNSWIndex<T> {
    pub fn search(&self, item: &[f32], k: u32) -> Vec<(&<T as Archive>::Archived, f32)> {
        if item.len() != self.dimension() {
            panic!("item len not equal to dimension");
        }
        let mut ret: BinaryHeap<Neighbor<f32, usize>> = self.search_knn(item, k).unwrap();
        let mut result = Vec::with_capacity(k as usize);
        let mut result_idx: Vec<(usize, f32)> = Vec::with_capacity(k as usize);
        while !ret.is_empty() {
            let top = ret.peek().unwrap();
            let top_idx = top.idx();
            let top_distance = top.distance();
            ret.pop();
            result_idx.push((top_idx, top_distance))
        }
        for i in 0..result_idx.len() {
            let cur_id = result_idx.len() - i - 1;
            result.push((&self.nodes[result_idx[cur_id].0].1, result_idx[cur_id].1));
        }
        result
    }

    fn search_layer(
        &self,
        root: u32,
        search_data: &[f32],
        level: u32,
        ef: u32,
        has_deletion: bool,
    ) -> BinaryHeap<Neighbor<f32, usize>> {
        let mut visited_id = FixedBitSet::with_capacity(self.nodes.len());
        let mut top_candidates: BinaryHeap<Neighbor<f32, usize>> = BinaryHeap::new();
        let mut candidates: BinaryHeap<Neighbor<f32, usize>> = BinaryHeap::new();
        let mut lower_bound: f32;

        if !has_deletion || !self.is_deleted(root) {
            let dist = self.get_distance_from_vec(self.get_data(root), search_data);
            top_candidates.push(Neighbor::new(root as usize, dist));
            candidates.push(Neighbor::new(root as usize, -dist));
            lower_bound = dist;
        } else {
            lower_bound = f32::MAX; //max dist in top_candidates
            candidates.push(Neighbor::new(root as usize, -lower_bound))
        }
        visited_id.insert(root as usize);

        while !candidates.is_empty() {
            let cur_neigh = candidates.peek().unwrap();
            let cur_dist = -cur_neigh._distance;
            let cur_id = cur_neigh.idx();
            candidates.pop();
            if cur_dist > lower_bound {
                break;
            }
            let cur_neighbors = self.get_neighbor(cur_id as u32, level);
            cur_neighbors.iter().for_each(|neigh| {
                if visited_id.contains(*neigh as usize) {
                    return;
                }
                visited_id.insert(*neigh as usize);
                let dist = self.get_distance_from_vec(self.get_data(*neigh), search_data);
                if top_candidates.len() < ef as usize || dist < lower_bound {
                    candidates.push(Neighbor::new(*neigh as usize, -dist));

                    if !self.is_deleted(*neigh) {
                        top_candidates.push(Neighbor::new(*neigh as usize, dist))
                    }

                    if top_candidates.len() > ef as usize {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap()._distance;
                    }
                }
            });
        }

        top_candidates
    }

    fn search_knn(
        &self,
        search_data: &[f32],
        k: u32,
    ) -> Result<BinaryHeap<Neighbor<f32, usize>>, &'static str> {
        let mut top_candidate: BinaryHeap<Neighbor<f32, usize>> = BinaryHeap::new();
        if self.n_constructed_items == 0 {
            return Ok(top_candidate);
        }
        let mut cur_id = self.root_id;
        let mut cur_dist = self.get_distance_from_vec(self.get_data(cur_id), search_data);
        let mut cur_level = self.cur_level;
        loop {
            let mut changed = true;
            while changed {
                changed = false;
                let cur_neighs = self.get_neighbor(cur_id, cur_level);
                for neigh in cur_neighs.iter() {
                    if *neigh > self.max_item {
                        return Err("cand error");
                    }
                    let dist = self.get_distance_from_vec(self.get_data(cur_id), search_data);
                    if dist < cur_dist {
                        cur_dist = dist;
                        cur_id = *neigh;
                        changed = true;
                    }
                }
            }
            if cur_level == 0 {
                break;
            }
            cur_level -= 1;
        }

        let search_range = if self.ef_search > k {
            self.ef_search
        } else {
            k
        };

        top_candidate = self.search_layer(cur_id, search_data, 0, search_range, self.has_removed);
        while top_candidate.len() > k as usize {
            top_candidate.pop();
        }

        Ok(top_candidate)
    }

    fn dimension(&self) -> usize {
        self.dimension as usize
    }

    fn is_deleted(&self, id: u32) -> bool {
        self.has_removed && self.delete_ids.contains(&id)
    }

    fn get_distance_from_vec(&self, x: &[f32], y: &[f32]) -> f32 {
        metrics::metric(x, y, self.mt).unwrap()
    }

    fn get_data(&self, id: u32) -> &[f32] {
        &self.nodes[id as usize].0[..]
    }

    fn get_neighbor(&self, id: u32, level: u32) -> &[u32] {
        if level == 0 {
            return &self.id2neighbor0[id as usize];
        }
        &self.id2neighbor[id as usize][(level - 1) as usize]
    }
}
