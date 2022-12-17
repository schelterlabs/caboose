use crate::types::SimilarUser;
use crate::row_accumulator::RowAccumulator;

use std::clone::Clone;
use std::ops::Mul;
use std::collections::BinaryHeap;
use sprs::CsMat;


pub(crate) fn build_topk_index<T: Clone + Copy + Default + Mul<Output = f64>>(
    user_representations: &CsMat<T>,
    k: usize
) -> Vec<BinaryHeap<SimilarUser>> {

    let (num_users, num_items) = user_representations.shape();
    let user_representations_transposed: CsMat<_> = user_representations.to_csc();

    let data = user_representations.data();
    let indices = user_representations.indices();
    let indptr = user_representations.indptr();
    let data_t = user_representations_transposed.data();
    let indices_t = user_representations_transposed.indices();
    let indptr_t = user_representations_transposed.indptr();

    let l2norms: Vec<f64> = (0..num_users)
        .map(|user| {
            let mut sum_of_squares: f64 = 0.0;
            for item_index in indptr.outer_inds_sz(user) {
                let value = data[item_index];
                sum_of_squares += value * value;
            }
            sum_of_squares.sqrt()
        })
        .collect();

    let mut accumulator = RowAccumulator::new(num_items.clone(), k);

    let mut index: Vec<BinaryHeap<SimilarUser>> = Vec::with_capacity(num_users);
    for user in 0..num_users {
        for item_index in indptr.outer_inds_sz(user) {
            let value = data[item_index];
            for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                accumulator.add_to(indices_t[user_index], data_t[user_index] * value.clone());
            }
        }

        let topk = accumulator.topk_and_clear(user, &l2norms);

        index.push(topk);
    }

    index
}
