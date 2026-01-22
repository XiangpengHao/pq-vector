//! Expression parsing and literal extraction utilities.

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
};
use datafusion::common::ScalarValue;

pub(crate) fn scalar_to_f32_list(value: &ScalarValue) -> Option<Vec<f32>> {
    match value {
        ScalarValue::List(array) => list_array_to_f32(array),
        ScalarValue::FixedSizeList(array) => fixed_list_array_to_f32(array),
        ScalarValue::LargeList(array) => large_list_array_to_f32(array),
        _ => None,
    }
}

fn list_array_to_f32(array: &ListArray) -> Option<Vec<f32>> {
    if array.len() != 1 {
        return None;
    }
    if array.is_null(0) {
        return None;
    }
    let values = array.value(0);
    float_array_to_vec(&values)
}

fn large_list_array_to_f32(array: &LargeListArray) -> Option<Vec<f32>> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let values = array.value(0);
    float_array_to_vec(&values)
}

fn fixed_list_array_to_f32(array: &FixedSizeListArray) -> Option<Vec<f32>> {
    if array.len() != 1 || array.is_null(0) {
        return None;
    }
    let values = array.value(0);
    float_array_to_vec(&values)
}

fn float_array_to_vec(values: &ArrayRef) -> Option<Vec<f32>> {
    if let Some(arr) = values.as_any().downcast_ref::<Float32Array>() {
        return Some((0..arr.len()).map(|i| arr.value(i)).collect());
    }
    if let Some(arr) = values.as_any().downcast_ref::<Float64Array>() {
        return Some((0..arr.len()).map(|i| arr.value(i) as f32).collect());
    }
    None
}
