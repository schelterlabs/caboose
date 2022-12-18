use std::collections::BinaryHeap;
use crate::types::SimilarUser;

pub(crate) enum HeapUpdateResult {
    NewTopK(BinaryHeap<SimilarUser>),
    FullUpdateRequired,
}

// TODO try to rewrite this to take ownership of the old topk heap
pub(crate) fn update_heap(
    topk: &mut BinaryHeap<SimilarUser>,
    update: SimilarUser,
    k: usize
) -> HeapUpdateResult {

    if topk.len() == k {
        let old_top = topk.peek().unwrap();

        if old_top.user == update.user && old_top.similarity > update.similarity {
            return HeapUpdateResult::FullUpdateRequired
        }
    }

    let mut new_topk = BinaryHeap::with_capacity(k);

    for existing_entry in topk.iter() {
        if existing_entry.user != update.user {
            new_topk.push(existing_entry.clone());
        }
    }
    new_topk.push(update);

    HeapUpdateResult::NewTopK(new_topk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_not_smallest() {

        let k = 3;
        let mut original_topk = BinaryHeap::with_capacity(k);
        original_topk.push(SimilarUser::new(1, 1.0));
        original_topk.push(SimilarUser::new(2, 0.8));
        original_topk.push(SimilarUser::new(3, 0.5));

        let result = update_heap(&mut original_topk, SimilarUser::new(2, 0.7), k);

        assert!(!matches!(result, HeapUpdateResult::FullUpdateRequired));

        if let HeapUpdateResult::NewTopK(new_topk) = result {
            assert_eq!(new_topk.len(), 3);

            let n = new_topk.into_sorted_vec();

            check_entry(&n[0], 1, 1.0);
            check_entry(&n[1], 2, 0.7);
            check_entry(&n[2], 3, 0.5);
        }
    }

    #[test]
    fn test_update_moves() {

        let k = 3;
        let mut original_topk = BinaryHeap::with_capacity(k);
        original_topk.push(SimilarUser::new(1, 1.0));
        original_topk.push(SimilarUser::new(2, 0.8));
        original_topk.push(SimilarUser::new(3, 0.5));

        let result = update_heap(&mut original_topk, SimilarUser::new(2, 1.5), k);

        assert!(!matches!(result, HeapUpdateResult::FullUpdateRequired));

        if let HeapUpdateResult::NewTopK(new_topk) = result {
            assert_eq!(new_topk.len(), 3);

            let n = new_topk.into_sorted_vec();

            check_entry(&n[0], 2, 1.5);
            check_entry(&n[1], 1, 1.0);
            check_entry(&n[2], 3, 0.5);
        }
    }

    #[test]
    fn test_update_smallest_but_becomes_larger() {

        let k = 3;
        let mut original_topk = BinaryHeap::with_capacity(k);
        original_topk.push(SimilarUser::new(1, 1.0));
        original_topk.push(SimilarUser::new(2, 0.8));
        original_topk.push(SimilarUser::new(3, 0.5));

        let result = update_heap(&mut original_topk, SimilarUser::new(3, 0.6), k);

        assert!(!matches!(result, HeapUpdateResult::FullUpdateRequired));

        if let HeapUpdateResult::NewTopK(new_topk) = result {
            assert_eq!(new_topk.len(), 3);

            let n = new_topk.into_sorted_vec();

            check_entry(&n[0], 1, 1.0);
            check_entry(&n[1], 2, 0.8);
            check_entry(&n[2], 3, 0.6);
        }
    }

    #[test]
    fn test_update_smallest_becomes_smaller() {

        let k = 3;
        let mut original_topk = BinaryHeap::with_capacity(k);
        original_topk.push(SimilarUser::new(1, 1.0));
        original_topk.push(SimilarUser::new(2, 0.8));
        original_topk.push(SimilarUser::new(3, 0.5));

        let result = update_heap(&mut original_topk, SimilarUser::new(3, 0.4), k);

        assert!(matches!(result, HeapUpdateResult::FullUpdateRequired));
    }

    #[test]
    fn test_update_smallest_becomes_smaller_but_not_full() {

        let k = 3;
        let mut original_topk = BinaryHeap::with_capacity(k);
        original_topk.push(SimilarUser::new(1, 1.0));
        original_topk.push(SimilarUser::new(3, 0.5));

        let result = update_heap(&mut original_topk, SimilarUser::new(3, 0.4), k);

        assert!(!matches!(result, HeapUpdateResult::FullUpdateRequired));

        if let HeapUpdateResult::NewTopK(new_topk) = result {
            assert_eq!(new_topk.len(), 2);

            let n = new_topk.into_sorted_vec();

            check_entry(&n[0], 1, 1.0);
            check_entry(&n[1], 3, 0.4);
        }
    }

    fn check_entry(entry: &SimilarUser, expected_user: usize, expected_similarity: f64) {
        assert_eq!(entry.user, expected_user);
        assert!((entry.similarity - expected_similarity).abs() < 0.0001);
    }
}