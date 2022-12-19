use pyo3::prelude::*;
use numpy::PyArrayDyn;
use sprs::CsMat;
use crate::user_similarity_index::UserSimilarityIndex;

mod user_similarity_index;
mod similar_user;
mod row_accumulator;
mod topk;
mod utils;

#[pyclass]
struct Index {
    similarity_index: UserSimilarityIndex,
}

#[pymethods]
impl Index {

    fn topk(&self, row: usize) -> PyResult<Vec<(usize,f64)>> {
        let topk: Vec<(usize,f64)> = self.similarity_index.neighbors(row)
            .map(|similar_user| (similar_user.user, similar_user.similarity))
            .collect();
        Ok(topk)
    }

    fn forget(&mut self, row: usize, column: usize) -> PyResult<()> {
        self.similarity_index.forget(row, column);
        Ok(())
    }

    #[new]
    fn new(
        num_rows: usize,
        num_cols: usize,
        indptr: &PyArrayDyn<i32>,
        indices: &PyArrayDyn<i32>,
        data: &PyArrayDyn<f64>,
        k: usize
    ) -> Self {

        // TODO this horribly inefficient for now...
        let indices_copy: Vec<usize> = indices.to_vec().unwrap()
            .into_iter().map(|x| x as usize).collect();
        let indptr_copy: Vec<usize> = indptr.to_vec().unwrap()
            .into_iter().map(|x| x as usize).collect();
        let data_copy: Vec<f64> = data.to_vec().unwrap();

        let representations =
            CsMat::new((num_rows, num_cols), indptr_copy, indices_copy, data_copy);

        Self {
            similarity_index: UserSimilarityIndex::new(representations, k),
        }
    }
}



#[pymodule]
fn caboose(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Index>()?;
    Ok(())
}