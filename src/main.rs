extern crate sprs;

use sprs::{CsMat, TriMat};

mod types;
mod row_accumulator;
mod index_builder;

use crate::index_builder::build_topk_index;

fn main() {

    let num_users = 4;
    let num_items = 5;

    let mut input = TriMat::new((num_users, num_items));
    input.add_triplet(0, 0, 3.0);
    input.add_triplet(1, 2, 2.0);
    input.add_triplet(1, 3, 3.0);
    input.add_triplet(3, 0, 2.0);

    let user_representations: CsMat<_> = input.to_csr();

    let index = build_topk_index(&user_representations, 2);

    for (user, entry) in index.iter().enumerate() {
        println!("{:?}: {:?}", user, entry);
    }

    

    //let compare = &user_representations * &user_representations.transpose_view();
    //println!("{:?}", compare.to_dense());
}
