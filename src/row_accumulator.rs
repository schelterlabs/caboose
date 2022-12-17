use std::collections::BinaryHeap;

use crate::types::SimilarUser;

pub(crate) struct RowAccumulator {
    sums: Vec<f64>,
    non_zeros: Vec<isize>,
    head: isize,
    k: usize
}

const NONE: f64 = 0.0;
const NOT_OCCUPIED: isize = -1;
const NO_HEAD: isize = -2;

impl RowAccumulator {

    pub(crate) fn new(num_items: usize, k: usize) -> Self {
        RowAccumulator {
            sums: vec![NONE; num_items],
            non_zeros: vec![NOT_OCCUPIED; num_items],
            head: NO_HEAD,
            k,
        }
    }

    pub(crate) fn add_to(&mut self, column: usize, value: f64) {
        self.sums[column] += value;

        if self.non_zeros[column] == NOT_OCCUPIED {
            self.non_zeros[column] = self.head.clone();
            self.head = column as isize;
        }
    }

    pub(crate) fn topk_and_clear(
        &mut self,
        current_user_index: usize,
        l2norms: &Vec<f64>
    ) -> BinaryHeap<SimilarUser> {

        let mut topk_similar_users: BinaryHeap<SimilarUser> = BinaryHeap::with_capacity(self.k);

        while self.head != NO_HEAD {
            let other_user_index = self.head as usize;

            if other_user_index != current_user_index {
                let similarity = self.sums[other_user_index] /
                    (l2norms[current_user_index] * l2norms[other_user_index]);

                let scored_user = SimilarUser::new(other_user_index, similarity);

                if topk_similar_users.len() < self.k {
                    topk_similar_users.push(scored_user);
                } else {
                    let mut top = topk_similar_users.peek_mut().unwrap();
                    if scored_user < *top {
                        *top = scored_user;
                    }
                }
            }

            self.head = self.non_zeros[other_user_index];
            self.sums[other_user_index] = NONE;
            self.non_zeros[other_user_index] = NOT_OCCUPIED;
        }
        self.head = NO_HEAD;

        topk_similar_users
    }

}