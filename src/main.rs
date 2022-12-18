extern crate sprs;

use sprs::{CsMat, TriMat};

mod types;
mod row_accumulator;
mod user_similarity_index;
mod topk;
mod utils;

use crate::user_similarity_index::UserSimilarityIndex;

fn main() {

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

    let user_representations: CsMat<f64> = input.to_csr();

    let mut index = UserSimilarityIndex::new(user_representations, 1);

    for user in 0..num_users {
        println!("{:?}: {:?}", user, index.neighbors(user));
    }

    index.forget(2, 1);

    println!("\n\nAfter forgetting:");
    for user in 0..num_users {
        println!("{:?}: {:?}", user, index.neighbors(user));
    }
}

