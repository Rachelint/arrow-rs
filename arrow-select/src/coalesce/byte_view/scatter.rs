// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Scatter gathered StringView ranges into multiple in-progress arrays.
//!
//! The caller gathers rows into target order; this module copies contiguous
//! ranges while managing payload buffers and completed output arrays.
//! Entry point: [`scatter_string_view_ranges`].

use crate::coalesce::InProgressArray;
use crate::coalesce::byte_view::InProgressByteViewArray;
use arrow_array::cast::AsArray;
use arrow_array::types::StringViewType;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::Buffer;
use arrow_data::{ByteView, MAX_INLINE_VIEW_LEN};
use arrow_schema::ArrowError;
use std::collections::VecDeque;

/// Mutable destination state for one column in one bucket.
pub struct StringViewScatterTarget<'a> {
    /// Builder receiving the rows assigned to this target.
    pub array: &'a mut InProgressByteViewArray<StringViewType>,
    /// Buffered row count, kept in sync with the builder.
    pub rows: &'a mut usize,
    /// Completed arrays, emitted when the builder reaches its batch size.
    pub finished: &'a mut VecDeque<ArrayRef>,
}

/// Scatter contiguous ranges of a gathered StringView array into all destinations.
///
/// `range_lengths[i]` gives the number of consecutive source rows for target `i`.
/// Ranges partition the source in target order. Each destination uses the same
/// payload copy/GC decision as coalescing its selected range: copy when retained
/// buffers exceed twice the selected payload size. Inline values, nulls, row order
/// and output batch boundaries are preserved.
///
/// # Panics
///
/// Panics if the source is not StringView, lengths do not partition the source,
/// the target count differs, or buffered row counts disagree with their builders.
/// Target batch sizes must be positive, and buffered rows must be below that size.
pub fn scatter_string_view_ranges(
    source: &ArrayRef,
    range_lengths: &[u32],
    targets: &mut [StringViewScatterTarget<'_>],
) -> Result<(), ArrowError> {
    // Step 1: Validate the gathered layout and inspect source metadata once.
    let source = source.as_string_view();
    assert_eq!(range_lengths.len(), targets.len());
    assert_eq!(
        range_lengths.iter().map(|&len| len as usize).sum::<usize>(),
        source.len()
    );
    let buffers = source.data_buffers();
    let actual_bytes: usize = buffers.iter().map(|b| b.capacity()).sum();
    let mut offset = 0;

    // Step 2: Keep one target active while copying its contiguous source range.
    for (&length, target) in range_lengths.iter().zip(targets) {
        let array = &mut *target.array;
        assert_eq!(*target.rows, array.views.len());
        assert!(*target.rows < array.batch_size);
        let end = offset + length as usize;
        if offset == end {
            continue;
        }
        array.set_source(None);
        let mut remaining_bytes: usize = source.views()[offset..end]
            .iter()
            .map(|&view| {
                let length = view as u32;
                if length > MAX_INLINE_VIEW_LEN {
                    length as usize
                } else {
                    0
                }
            })
            .sum();
        let is_inline = remaining_bytes == 0;
        let needs_gc = !is_inline && actual_bytes > 2 * remaining_bytes;
        while offset < end {
            let copied = (end - offset).min(array.batch_size - *target.rows);
            let views = &source.views()[offset..offset + copied];
            array.ensure_capacity();
            if let Some(nulls) = source.nulls() {
                array.nulls.append_buffer(&nulls.slice(offset, copied));
            } else {
                array.nulls.append_n_non_nulls(copied);
            }
            if is_inline {
                array.views.extend_from_slice(views);
            } else if needs_gc {
                copy_string_views(array, views, buffers, &mut remaining_bytes);
            } else {
                // Each segment fills the target or ends this source range, so
                // shared buffers are registered only once per output batch.
                array.append_views_and_update_buffer_index(views, buffers, false);
            }
            *target.rows += copied;
            offset += copied;
            if *target.rows == array.batch_size {
                target.finished.push_back(array.finish()?);
                *target.rows = 0;
            }
        }
    }
    Ok(())
}

fn copy_string_views(
    array: &mut InProgressByteViewArray<StringViewType>,
    views: &[u128],
    buffers: &[Buffer],
    remaining_bytes: &mut usize,
) {
    for &view in views {
        let mut view = ByteView::from(view);
        if view.length > MAX_INLINE_VIEW_LEN {
            let length = view.length as usize;
            if array
                .current
                .as_ref()
                .is_none_or(|b| b.capacity() - b.len() < length)
            {
                array.finish_current();
                array.current = Some(array.buffer_source.next_buffer(*remaining_bytes));
            }
            let dst = array.current.as_mut().unwrap();
            let src = &buffers[view.buffer_index as usize]
                [view.offset as usize..view.offset as usize + length];
            view.offset = dst.len().try_into().expect("view offset fits u32");
            view.buffer_index = array
                .completed
                .len()
                .try_into()
                .expect("buffer index fits u32");
            // SAFETY: capacity was checked above; source and destination buffers are distinct.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr(),
                    dst.as_mut_ptr().add(dst.len()),
                    length,
                );
                dst.set_len(dst.len() + length);
            }
            *remaining_bytes -= length;
        }
        array.views.push(view.as_u128());
    }
}

#[cfg(test)]
mod tests {
    use crate::coalesce::byte_view::scatter::{self, StringViewScatterTarget};
    use crate::coalesce::{InProgressArray, InProgressByteViewArray};
    use crate::take;
    use arrow_array::cast::AsArray;
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, ArrayRef, StringViewArray, UInt32Array};
    use std::collections::VecDeque;
    use std::sync::Arc;

    #[test]
    fn scatter_preserves_values_across_batches() {
        // One bucket exercises buffer sharing; 32 exercise GC. Tiny output
        // batches cover filling, draining and continuing within one source.
        for bucket_count in [1, 32, 512] {
            for inline_only in [false, true] {
                for with_nulls in [false, true] {
                    let mut arrays: Vec<_> = (0..bucket_count)
                        .map(|_| InProgressByteViewArray::<StringViewType>::new(7))
                        .collect();
                    let mut output = vec![VecDeque::new(); bucket_count];
                    let mut rows = vec![0usize; bucket_count];
                    let mut expected: Vec<Vec<Option<String>>> = vec![Vec::new(); bucket_count];
                    for batch in 0..3 {
                        let values: Vec<Option<String>> = (0..301)
                            .map(|row| {
                                if with_nulls && row % 11 == 0 {
                                    None
                                } else if inline_only || row % 5 == 0 {
                                    Some(format!("s{row}"))
                                } else {
                                    Some(format!(
                                        "{batch}:{row}:{}",
                                        "long string content".repeat(1 + row % 30)
                                    ))
                                }
                            })
                            .collect();
                        let source: ArrayRef = Arc::new(StringViewArray::from_iter(
                            values.iter().map(|v| v.as_deref()),
                        ));
                        let ids: Vec<_> = (0..values.len())
                            .map(|row| (row * 17 + batch) % bucket_count)
                            .collect();
                        for (value, &id) in values.into_iter().zip(&ids) {
                            expected[id].push(value);
                        }
                        let mut indices: Vec<u32> = (0..source.len() as u32).collect();
                        indices.sort_by_key(|&row| ids[row as usize]);
                        let source =
                            take::take(source.as_ref(), &UInt32Array::from(indices), None).unwrap();
                        let mut lengths = vec![0u32; bucket_count];
                        for &id in &ids {
                            lengths[id] += 1;
                        }
                        let mut targets: Vec<_> = arrays
                            .iter_mut()
                            .zip(&mut rows)
                            .zip(&mut output)
                            .map(|((array, rows), finished)| StringViewScatterTarget {
                                array,
                                rows,
                                finished,
                            })
                            .collect();
                        scatter::scatter_string_view_ranges(&source, &lengths, &mut targets)
                            .unwrap();
                    }
                    for (id, array) in arrays.iter_mut().enumerate() {
                        if rows[id] != 0 {
                            output[id].push_back(array.finish().unwrap());
                        }
                        let mut actual = Vec::new();
                        for result in &output[id] {
                            assert!(result.len() <= 7);
                            result.to_data().validate_full().unwrap();
                            let strings = result.as_string_view();
                            actual.extend(strings.iter().map(|s| s.map(str::to_string)));
                        }
                        assert_eq!(actual, expected[id]);
                        assert!(array.source.is_none());
                    }
                }
            }
        }
    }
}
