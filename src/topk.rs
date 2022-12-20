use std::collections::BinaryHeap;
use crate::similar_user::SimilarUser;
use std::collections::binary_heap::Iter;

pub(crate) struct TopK {
    heap: BinaryHeap<SimilarUser>,
}

impl TopK {

    pub(crate) fn new(heap: BinaryHeap<SimilarUser>) -> Self {
        Self { heap }
    }

    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    pub(crate) fn iter(&self) -> Iter<SimilarUser> {
        self.heap.iter()
    }

    // TODO there must be a better way
    pub(crate) fn contains(&self, user: usize) -> bool {
        for entry in self.heap.iter() {
            if entry.user == user {
                return true;
            }
        }
        false
    }

    pub(crate) fn offer_non_existing_entry(&mut self, offered_entry: SimilarUser) -> bool {
        let mut top = self.heap.peek_mut().unwrap();
        if offered_entry < *top {
            *top = offered_entry;
            return true
        }
        false
    }


    pub(crate) fn update_existing_entry(
        &mut self,
        update: SimilarUser,
        k: usize
    ) -> bool {

        if self.heap.len() == k {
            let old_top = self.heap.peek().unwrap();

            if old_top.user == update.user && old_top.similarity > update.similarity {
                return true
            }
        }

        let mut new_topk = BinaryHeap::with_capacity(k);

        for existing_entry in self.heap.iter() {
            if existing_entry.user != update.user {
                new_topk.push(existing_entry.clone());
            }
        }

        if update.similarity != 0.0 {
            new_topk.push(update);
        }

        self.heap = new_topk;

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_not_smallest() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let mut topk = TopK::new(original_entries);

        let recomputation_required = topk.update_existing_entry(SimilarUser::new(2, 0.7), k);

        assert!(!recomputation_required);
        assert_eq!(topk.len(), 3);

        let n = topk.heap.into_sorted_vec();
        check_entry(&n[0], 1, 1.0);
        check_entry(&n[1], 2, 0.7);
        check_entry(&n[2], 3, 0.5);
    }

    #[test]
    fn test_update_moves() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let mut topk = TopK::new(original_entries);

        let recomputation_required = topk.update_existing_entry(SimilarUser::new(2, 1.5), k);

        assert!(!recomputation_required);
        assert_eq!(topk.len(), 3);

        let n = topk.heap.into_sorted_vec();
        check_entry(&n[0], 2, 1.5);
        check_entry(&n[1], 1, 1.0);
        check_entry(&n[2], 3, 0.5);
    }

    #[test]
    fn test_update_smallest_but_becomes_larger() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let mut topk = TopK::new(original_entries);

        let recomputation_required = topk.update_existing_entry(SimilarUser::new(3, 0.6), k);

        assert!(!recomputation_required);
        assert_eq!(topk.len(), 3);

        let n = topk.heap.into_sorted_vec();
        check_entry(&n[0], 1, 1.0);
        check_entry(&n[1], 2, 0.8);
        check_entry(&n[2], 3, 0.6);
    }

    #[test]
    fn test_update_smallest_becomes_smaller() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(2, 0.8));
        original_entries.push(SimilarUser::new(3, 0.5));

        let mut topk = TopK::new(original_entries);

        let recomputation_required = topk.update_existing_entry(SimilarUser::new(3, 0.4), k);

        assert!(recomputation_required);
    }

    #[test]
    fn test_update_smallest_becomes_smaller_but_not_full() {

        let k = 3;
        let mut original_entries = BinaryHeap::with_capacity(k);
        original_entries.push(SimilarUser::new(1, 1.0));
        original_entries.push(SimilarUser::new(3, 0.5));

        let mut topk = TopK::new(original_entries);

        let recomputation_required = topk.update_existing_entry(SimilarUser::new(3, 0.4), k);

        assert!(!recomputation_required);
        assert_eq!(topk.len(), 2);

        let n = topk.heap.into_sorted_vec();
        check_entry(&n[0], 1, 1.0);
        check_entry(&n[1], 3, 0.4);
    }

    fn check_entry(entry: &SimilarUser, expected_user: usize, expected_similarity: f64) {
        assert_eq!(entry.user, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }
}