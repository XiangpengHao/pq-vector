//! Parquet access plan helpers and IVF candidate cursor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::datasource::physical_plan::FileGroup;
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::datasource::physical_plan::parquet::ParquetAccessPlan;
use datafusion::datasource::source::DataSourceExec;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::physical_plan::ExecutionPlan;
use parquet::arrow::arrow_reader::{RowSelection, RowSelector};
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;

#[derive(Clone)]
pub(crate) struct FileEntry {
    pub(crate) object_path: String,
    pub(crate) row_groups: Vec<u64>,
    pub(crate) candidates: Vec<u32>,
}

pub(crate) struct ParquetScanInfo {
    pub(crate) file_groups: Vec<FileGroup>,
    pub(crate) object_store_url: ObjectStoreUrl,
}

pub(crate) fn gather_single_parquet_scan(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<ParquetScanInfo>> {
    let mut scans = Vec::new();
    collect_parquet_scans(plan, &mut scans)?;
    if scans.len() != 1 {
        return Ok(None);
    }
    Ok(Some(scans.remove(0)))
}

fn collect_parquet_scans(
    plan: &Arc<dyn ExecutionPlan>,
    scans: &mut Vec<ParquetScanInfo>,
) -> Result<()> {
    if let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>()
        && let Some((file_scan, _source)) = exec.downcast_to_file_source::<ParquetSource>()
    {
        scans.push(ParquetScanInfo {
            file_groups: file_scan.file_groups.clone(),
            object_store_url: file_scan.object_store_url.clone(),
        });
    }

    for child in plan.children() {
        collect_parquet_scans(child, scans)?;
    }
    Ok(())
}

pub(crate) fn rewrite_with_access_plans(
    plan: Arc<dyn ExecutionPlan>,
    access_plans: &HashMap<String, ParquetAccessPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(exec) = plan.as_any().downcast_ref::<DataSourceExec>()
        && let Some((file_scan, _source)) = exec.downcast_to_file_source::<ParquetSource>()
    {
        let mut new_config = file_scan.clone();
        let mut new_groups = Vec::with_capacity(new_config.file_groups.len());
        for group in &new_config.file_groups {
            let mut new_files = Vec::with_capacity(group.len());
            for file in group.files() {
                let key = file.path().as_ref().to_string();
                let plan = access_plans.get(&key);
                let new_file = if let Some(plan) = plan {
                    file.clone().with_extensions(Arc::new(plan.clone()))
                } else {
                    file.clone()
                };
                new_files.push(new_file);
            }
            new_groups.push(FileGroup::new(new_files));
        }
        new_config.file_groups = new_groups;
        let new_exec = DataSourceExec::new(Arc::new(new_config));
        return Ok(Arc::new(new_exec));
    }

    let mut new_children = Vec::with_capacity(plan.children().len());
    let mut changed = false;
    for child in plan.children() {
        let new_child = rewrite_with_access_plans(child.clone(), access_plans)?;
        changed |= !Arc::ptr_eq(child, &new_child);
        new_children.push(new_child);
    }
    if changed {
        plan.with_new_children(new_children)
    } else {
        Ok(plan)
    }
}

pub(crate) fn build_access_plans(
    files: &[FileEntry],
    selections: &HashMap<String, Vec<u32>>,
) -> Result<HashMap<String, ParquetAccessPlan>> {
    let mut plans = HashMap::new();
    for entry in files {
        let rows = selections
            .get(&entry.object_path)
            .map(|r| r.as_slice())
            .unwrap_or(&[]);
        let plan = access_plan_for_rows(&entry.row_groups, rows)?;
        plans.insert(entry.object_path.clone(), plan);
    }
    Ok(plans)
}

fn access_plan_for_rows(row_groups: &[u64], rows: &[u32]) -> Result<ParquetAccessPlan> {
    if rows.is_empty() {
        return Ok(ParquetAccessPlan::new_none(row_groups.len()));
    }
    let mut plan = ParquetAccessPlan::new_all(row_groups.len());
    let mut starts = Vec::with_capacity(row_groups.len());
    let mut current = 0u64;
    for &count in row_groups {
        starts.push(current);
        current += count;
    }

    let mut rows_by_group: Vec<Vec<u32>> = vec![Vec::new(); row_groups.len()];
    for &row in rows {
        let row = row as u64;
        let idx = match starts.partition_point(|&s| s <= row) {
            0 => 0,
            n => n - 1,
        };
        let start = starts[idx];
        rows_by_group[idx].push((row - start) as u32);
    }

    for (group_idx, mut group_rows) in rows_by_group.into_iter().enumerate() {
        if group_rows.is_empty() {
            plan.skip(group_idx);
            continue;
        }
        group_rows.sort_unstable();
        group_rows.dedup();
        let row_count = row_groups[group_idx] as usize;
        let selection = build_row_selection(row_count, &group_rows)?;
        plan.scan_selection(group_idx, selection);
    }

    Ok(plan)
}

fn build_row_selection(row_count: usize, rows: &[u32]) -> Result<RowSelection> {
    let mut selectors = Vec::new();
    let mut current = 0usize;
    for &row in rows {
        let row = row as usize;
        if row > current {
            selectors.push(RowSelector::skip(row - current));
        }
        selectors.push(RowSelector::select(1));
        current = row + 1;
    }
    if current < row_count {
        selectors.push(RowSelector::skip(row_count - current));
    }
    Ok(RowSelection::from(selectors))
}

pub(crate) fn read_row_group_row_counts(path: &Path) -> Result<Vec<u64>> {
    let file = std::fs::File::open(path).map_err(datafusion::common::DataFusionError::from)?;
    let reader =
        SerializedFileReader::new(file).map_err(datafusion::common::DataFusionError::from)?;
    let metadata = reader.metadata();
    let counts = metadata
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as u64)
        .collect::<Vec<_>>();
    Ok(counts)
}

pub(crate) fn local_path_from_object_store(
    object_store_url: &ObjectStoreUrl,
    path: &str,
) -> Option<PathBuf> {
    if !object_store_url.as_str().starts_with("file://") {
        return None;
    }
    if path.starts_with('/') {
        return Some(PathBuf::from(path));
    }
    let mut full = PathBuf::from("/");
    full.push(path);
    Some(full)
}

pub(crate) struct CandidateCursor {
    candidates: Vec<Vec<u32>>,
    positions: Vec<usize>,
    round_robin: usize,
}

impl CandidateCursor {
    pub(crate) fn new(file_count: usize) -> Self {
        Self {
            candidates: vec![Vec::new(); file_count],
            positions: vec![0; file_count],
            round_robin: 0,
        }
    }

    pub(crate) fn add_candidates(&mut self, idx: usize, candidates: Vec<u32>) {
        if let Some(slot) = self.candidates.get_mut(idx) {
            *slot = candidates;
        }
    }

    pub(crate) fn next_batch(&mut self, batch_size: usize) -> Vec<(usize, u32)> {
        if batch_size == 0 || self.candidates.is_empty() {
            return Vec::new();
        }
        let file_count = self.candidates.len();
        let mut output = Vec::with_capacity(batch_size);
        let mut idx = self.round_robin;
        while output.len() < batch_size {
            let mut progressed = false;
            for _ in 0..file_count {
                let file_idx = idx % file_count;
                idx += 1;
                if self.positions[file_idx] < self.candidates[file_idx].len() {
                    let row = self.candidates[file_idx][self.positions[file_idx]];
                    self.positions[file_idx] += 1;
                    output.push((file_idx, row));
                    progressed = true;
                    if output.len() >= batch_size {
                        break;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        self.round_robin = idx % file_count;
        output
    }
}
