use pyo3::prelude::*;
use numpy::PyArrayDyn;
use sprs::CsMat;
use caboose_index::sparse_topk_index::SparseTopKIndex;


#[pyclass]
struct Index {
    similarity_index: SparseTopKIndex,
}

#[pymethods]
impl Index {

    fn topk(&self, row: usize) -> PyResult<Vec<(usize,f32)>> {
        let topk: Vec<(usize,f32)> = self.similarity_index.neighbors(row)
            .map(|similar_user| (similar_user.row as usize, similar_user.similarity))
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

        let similarity_index: SparseTopKIndex = SparseTopKIndex::new(representations, k);

        Self { similarity_index }
    }
}



#[pymodule]
fn caboose(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Index>()?;
    Ok(())
}