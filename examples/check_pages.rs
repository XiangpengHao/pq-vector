use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::file::serialized_reader::ReadOptionsBuilder;
use std::fs::File;

fn main() {
    let path = "data/vldb_2025_indexed.parquet";
    println!("=== {} ===\n", path);

    // Open with page index enabled
    let file = File::open(path).unwrap();
    let options = ReadOptionsBuilder::new().with_page_index().build();
    let reader = SerializedFileReader::new_with_options(file, options).unwrap();
    let metadata = reader.metadata();

    let rg = metadata.row_group(0);
    println!("Row group 0: {} rows\n", rg.num_rows());

    // Find embedding column
    for i in 0..rg.num_columns() {
        let col = rg.column(i);
        let col_path = col.column_path();
        if col_path.string().contains("embedding") {
            println!("Column {}: {}", i, col_path);
            println!("  Uncompressed size: {} bytes", col.uncompressed_size());

            if let Some(offset) = col.offset_index_offset() {
                println!("  Offset index offset: {}", offset);
                println!("  Offset index length: {:?}", col.offset_index_length());
            } else {
                println!("  No offset index!");
            }

            if let Some(offset) = col.column_index_offset() {
                println!("  Column index offset: {}", offset);
                println!("  Column index length: {:?}", col.column_index_length());
            } else {
                println!("  No column index!");
            }
        }
    }

    // Try to get page locations from offset index
    println!("\n=== Checking Page Index ===");
    if let Some(offset_index) = metadata.offset_index() {
        println!("Offset index available!");
        if let Some(rg_offset) = offset_index.first() {
            // Column 13 is embedding.list.item
            if let Some(col_offset) = rg_offset.get(13) {
                println!("Number of pages: {}", col_offset.page_locations().len());
                for (i, page) in col_offset.page_locations().iter().take(5).enumerate() {
                    println!(
                        "  Page {}: offset={}, size={}, first_row={}",
                        i, page.offset, page.compressed_page_size, page.first_row_index
                    );
                }
                if col_offset.page_locations().len() > 5 {
                    println!(
                        "  ... ({} more pages)",
                        col_offset.page_locations().len() - 5
                    );
                }
            }
        }
    } else {
        println!("No offset index in metadata");
    }
}
