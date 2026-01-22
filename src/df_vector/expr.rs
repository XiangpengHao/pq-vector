//! Expression parsing and literal extraction utilities.

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, SortExpr};

/// Normalize a sort expression by stripping aliases.
pub(crate) fn normalize_sort_expr(sort_expr: &SortExpr) -> Expr {
    let expr = sort_expr.expr.clone();
    match expr {
        Expr::Alias(alias) => *alias.expr,
        other => other,
    }
}

/// Extract (column, vector) from array_distance(column, [literal]) expression.
pub(crate) fn extract_array_distance(
    expr: &Expr,
) -> Option<(datafusion::common::Column, Vec<f32>)> {
    let Expr::ScalarFunction(sf) = expr else {
        return None;
    };
    if sf.name() != "array_distance" || sf.args.len() != 2 {
        return None;
    }
    let left = strip_wrappers(&sf.args[0]);
    let right = strip_wrappers(&sf.args[1]);
    let (column, literal) = match (column_from_expr(left), literal_from_expr(right)) {
        (Some(col), Some(lit)) => (col, lit),
        _ => match (column_from_expr(right), literal_from_expr(left)) {
            (Some(col), Some(lit)) => (col, lit),
            _ => return None,
        },
    };
    let vector = scalar_to_f32_list(&literal)?;
    Some((column, vector))
}

fn strip_wrappers<'a>(expr: &'a Expr) -> &'a Expr {
    match expr {
        Expr::Alias(alias) => strip_wrappers(alias.expr.as_ref()),
        Expr::Cast(cast) => strip_wrappers(cast.expr.as_ref()),
        Expr::TryCast(cast) => strip_wrappers(cast.expr.as_ref()),
        _ => expr,
    }
}

fn column_from_expr(expr: &Expr) -> Option<datafusion::common::Column> {
    match expr {
        Expr::Column(column) => Some(column.clone()),
        Expr::Alias(alias) => column_from_expr(alias.expr.as_ref()),
        Expr::Cast(cast) => column_from_expr(cast.expr.as_ref()),
        Expr::TryCast(cast) => column_from_expr(cast.expr.as_ref()),
        _ => None,
    }
}

fn literal_from_expr(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(lit, _) => Some(lit.clone()),
        Expr::Alias(alias) => literal_from_expr(alias.expr.as_ref()),
        Expr::Cast(cast) => literal_from_expr(cast.expr.as_ref()),
        Expr::TryCast(cast) => literal_from_expr(cast.expr.as_ref()),
        _ => None,
    }
}

fn scalar_to_f32_list(value: &ScalarValue) -> Option<Vec<f32>> {
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
