//! Minimal reproducer for DataFusion ARRAY[] type coercion bug in UDTF arguments
//!
//! Run: cargo run --example datafusion_array_bug
//!
//! ## Bug
//!
//! ```sql
//! SELECT ARRAY[0.1, 1, 2]                      -- works
//! SELECT * FROM any_udtf(ARRAY[0.1, 1, 2])     -- panics
//! ```
//!
//! ## Root Cause
//!
//! `get_table_function_source` (session_state.rs:1741-1744) calls
//! `simplifier.simplify(arg)` without calling `simplifier.coerce(arg)` first.
//! Regular SELECT works because TypeCoercion analyzer runs before ConstEvaluator.
//!
//! ## Fix
//!
//! ```rust,ignore
//! // session_state.rs:1741-1744
//! let args = args
//!     .into_iter()
//!     .map(|arg| simplifier.simplify(simplifier.coerce(arg, &dummy_schema)?))
//!     .collect::<datafusion_common::Result<Vec<_>>>()?;
//! ```

use datafusion::catalog::TableFunctionImpl;
use datafusion::logical_expr::Expr;
use datafusion::prelude::*;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = SessionContext::new();
    ctx.register_udtf("f", Arc::new(DummyUdtf));

    println!("SELECT ARRAY[0.1, 1, 2]  -- works:");
    ctx.sql("SELECT ARRAY[0.1, 1, 2]").await?.show().await?;

    println!("\nSELECT * FROM f(ARRAY[0.1, 1, 2])  -- panics:");
    ctx.sql("SELECT * FROM f(ARRAY[0.1, 1, 2])").await?;

    Ok(())
}

#[derive(Debug)]
struct DummyUdtf;

impl TableFunctionImpl for DummyUdtf {
    fn call(&self, _: &[Expr]) -> datafusion::error::Result<Arc<dyn datafusion::catalog::TableProvider>> {
        unimplemented!()
    }
}
