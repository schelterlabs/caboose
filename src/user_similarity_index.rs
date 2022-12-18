use crate::types::SimilarUser;
use crate::row_accumulator::RowAccumulator;

use std::clone::Clone;
use std::collections::BinaryHeap;
use std::collections::binary_heap::Iter;
use std::iter::Skip;
use sprs::CsMat;

use crate::utils::{HeapUpdateResult, update_heap};

pub(crate) struct UserSimilarityIndex {
    user_representations:  CsMat<f64>,
    user_representations_transposed: CsMat<f64>,
    topk_per_user: Vec<BinaryHeap<SimilarUser>>,
    k: usize,
    l2norms: Vec<f64>,
}

impl UserSimilarityIndex {

    pub(crate) fn neighbors(&self, user: usize) -> Skip<Iter<SimilarUser>> {
        let heap = &self.topk_per_user[user];
        let skip_size = 0;
        //let skip_size = if heap.len() > self.k { 1 } else { 0 };
        // TODO this relies on the heap root being in the first position
        heap.iter().skip(skip_size)
    }

    pub(crate) fn new(user_representations: CsMat<f64>, k: usize) -> Self {
        let (num_users, num_items) = user_representations.shape();

        let mut user_representations_transposed: CsMat<f64> = user_representations.to_owned();
        user_representations_transposed.transpose_mut();
        user_representations_transposed = user_representations_transposed.to_csr();

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

        let mut accumulator = RowAccumulator::new(num_items.clone());

        let mut topk_per_user: Vec<BinaryHeap<SimilarUser>> = Vec::with_capacity(num_users);
        for user in 0..num_users {
            for item_index in indptr.outer_inds_sz(user) {
                let value = data[item_index];
                for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                    accumulator.add_to(indices_t[user_index], data_t[user_index] * value.clone());
                }
            }

            let topk = accumulator.topk_and_clear(user, k, &l2norms);

            topk_per_user.push(topk);
        }

        Self {
            user_representations,
            user_representations_transposed,
            topk_per_user,
            k,
            l2norms,
        }
    }

    // fn update_other_users_topk(
    //     &self,
    //     user: usize,
    //     other_user: usize,
    //     similarity: f64,
    //     other_topk: &mut BinaryHeap<SimilarUser>,
    //     users_to_fully_recompute: &mut Vec<usize>
    // ) {
    //     let mut already_in_topk = false;
    //
    //     for other_users_topk_similar in other_topk.iter() {
    //         if other_users_topk_similar.user == user {
    //             already_in_topk = true;
    //             break
    //         }
    //     }
    //
    //     let similar_user_to_update = SimilarUser::new(user, similarity);
    //
    //     if !already_in_topk {
    //
    //         assert!(other_topk.len() >= self.k);
    //
    //         let mut top = other_topk.peek_mut().unwrap();
    //         if similar_user_to_update < *top {
    //             println!("--C1a: Not in topk, in topk after offer");
    //             *top = similar_user_to_update;
    //         } else {
    //             println!("--C1b: Not in topk, not in topk after offer");
    //         }
    //
    //     } else {
    //         if other_topk.len() < self.k {
    //             let update_result = update_heap(other_topk, similar_user_to_update, self.k);
    //
    //             //assert_ne!(matches!(HeapUpdateResult::FullUpdateRequired, update_result));
    //
    //             if let HeapUpdateResult::NewTopK(mut new_topk) = update_result {
    //                 std::mem::swap(other_topk, &mut new_topk);
    //             }
    //
    //             println!("--C2: In topk, updated, not full -> no recomp");
    //         } else {
    //             let update_result = update_heap(other_topk, similar_user_to_update, self.k);
    //             if let HeapUpdateResult::NewTopK(mut new_topk) = update_result {
    //                 std::mem::swap(other_topk, &mut new_topk);
    //                 println!("--C3a: In topk, updated, no recomp");
    //             } else {
    //                 users_to_fully_recompute.push(other_user);
    //                 println!("--C3b: In topk, recomputation required");
    //             }
    //         }
    //     }
    // }

    pub(crate) fn forget(&mut self, user: usize, item: usize) {

        let (_, num_items) = self.user_representations.shape();

        let old_value = self.user_representations.get(user, item).unwrap().clone();

        println!("-Updating user representations");
        // For some reason, the set method in sprs don't work...
        let index = self.user_representations.nnz_index_outer_inner(user, item).unwrap();
        self.user_representations.data_mut()[index.0] = 0.0;
        assert_eq!(*self.user_representations.get(user, item).unwrap(), 0.0_f64);

        let index = self.user_representations_transposed.nnz_index_outer_inner(item, user).unwrap();
        self.user_representations_transposed.data_mut()[index.0] = 0.0;
        assert_eq!(*self.user_representations_transposed.get(item, user).unwrap(), 0.0_f64);

        println!("-Updating norms");
        let old_l2norm = self.l2norms[user];
        self.l2norms[user] = ((old_l2norm * old_l2norm) - (old_value * old_value)).sqrt();

        println!("-Computing new similarities for user {}", user);
        let data = self.user_representations.data();
        let indices = self.user_representations.indices();
        let indptr = self.user_representations.indptr();
        let data_t = self.user_representations_transposed.data();
        let indices_t = self.user_representations_transposed.indices();
        let indptr_t = self.user_representations_transposed.indptr();

        let mut accumulator = RowAccumulator::new(num_items.clone());

        for item_index in indptr.outer_inds_sz(user) {
            let value = data[item_index];

            println!("{:?} {:?}", indices[item_index], value);

            for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                accumulator.add_to(indices_t[user_index], data_t[user_index] * value.clone());
            }
        }


        let updated_similarities = accumulator.collect_all(user, &self.l2norms);

        let mut users_to_fully_recompute = Vec::new();

        for similar_user in updated_similarities {

            assert_ne!(similar_user.user, user);

            let other_user = similar_user.user;
            let similarity = similar_user.similarity;

            println!("Updating topk of user {:?}", other_user);

            let other_topk = &mut self.topk_per_user[other_user];

            /*self.update_other_users_topk(
                user,
                other_user,
                similarity,
                other_topk,
                &mut users_to_fully_recompute
            );*/

            let mut already_in_topk = false;

            for other_users_topk_similar in other_topk.iter() {
                if other_users_topk_similar.user == user {
                    already_in_topk = true;
                    break
                }
            }

            let similar_user_to_update = SimilarUser::new(user, similarity);

            if !already_in_topk {

                assert!(other_topk.len() >= self.k);

                let mut top = other_topk.peek_mut().unwrap();
                if similar_user_to_update < *top {
                    println!("--C1a: Not in topk, in topk after offer");
                    *top = similar_user_to_update;
                } else {
                    println!("--C1b: Not in topk, not in topk after offer");
                }

            } else {
                if other_topk.len() < self.k {
                    let update_result = update_heap(other_topk, similar_user_to_update, self.k);

                    //assert_ne!(matches!(update_result, FullUpdateRequired));

                    if let HeapUpdateResult::NewTopK(mut new_topk) = update_result {
                        std::mem::swap(other_topk, &mut new_topk);
                    }

                    println!("--C2: In topk, updated, not full -> no recomp");
                } else {
                    let update_result = update_heap(other_topk, similar_user_to_update, self.k);
                    if let HeapUpdateResult::NewTopK(mut new_topk) = update_result {
                        std::mem::swap(other_topk, &mut new_topk);
                        println!("--C3a: In topk, updated, no recomp");
                    } else {
                        users_to_fully_recompute.push(other_user);
                        println!("--C3b: In topk, recomputation required");
                    }
                }
            }
        }

        let topk = accumulator.topk_and_clear(user, self.k, &self.l2norms);
        self.topk_per_user[user] = topk;


        for user_to_recompute in users_to_fully_recompute {
            for item_index in indptr.outer_inds_sz(user_to_recompute) {
                let value = data[item_index];
                for user_index in indptr_t.outer_inds_sz(indices[item_index]) {
                    accumulator.add_to(indices_t[user_index], data_t[user_index] * value.clone());
                }
            }

            let topk = accumulator.topk_and_clear(user_to_recompute, self.k, &self.l2norms);

            self.topk_per_user[user_to_recompute] = topk;
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn test_mini_example() {

        /*
        import numpy as np

        A = np.array(
                [[1, 1, 1, 0, 1],
                 [0, 1, 0, 1, 0],
                 [0, 1, 1, 0, 1],
                 [0, 0, 0, 1, 0]])

        similarity = np.dot(A, A.T)
        square_mag = np.diag(similarity)
        inv_square_mag = 1 / square_mag
        inv_square_mag[np.isinf(inv_square_mag)] = 0
        inv_mag = np.sqrt(inv_square_mag)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        print(cosine)

        [[1.         0.35355339 0.8660254  0.        ]
         [0.35355339 1.         0.40824829 0.70710678]
         [0.8660254  0.40824829 1.         0.        ]
         [0.         0.70710678 0.         1.        ]]
        */

        let num_users = 4;
        let num_items = 5;

        let triplets = vec![
            (0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0),
            (1, 1, 1.0), (1, 3, 1.0),
            (2, 1, 1.0), (2, 2, 1.0), (2, 4, 1.0),
            (3, 3, 1.0),
        ];

        let mut input = TriMat::new((num_users, num_items));
        for (row, col, val) in triplets {
            input.add_triplet(row, col, val);
        }

        let user_representations = input.to_csr();
        let index = UserSimilarityIndex::new(user_representations, 2);

        let mut n0: Vec<_> = index.neighbors(0).collect();
        n0.sort();
        assert_eq!(n0.len(), 2);
        check_entry(n0[0], 2, 0.8660254);
        check_entry(n0[1], 1, 0.35355339);

        let mut n1: Vec<_> = index.neighbors(1).collect();
        n1.sort();
        assert_eq!(n1.len(), 2);
        check_entry(n1[0], 3, 0.70710678);
        check_entry(n1[1], 2, 0.40824829);

        let mut n2: Vec<_> = index.neighbors(2).collect();
        n2.sort();
        assert_eq!(n2.len(), 2);
        check_entry(n2[0], 0, 0.8660254);
        check_entry(n2[1], 1, 0.40824829);

        let n3: Vec<_> = index.neighbors(3).collect();
        assert_eq!(n3.len(), 1);
        check_entry(n3[0], 1, 0.70710678);
    }

    fn check_entry(entry: &SimilarUser, expected_user: usize, expected_similarity: f64) {
        assert_eq!(entry.user, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }

}

