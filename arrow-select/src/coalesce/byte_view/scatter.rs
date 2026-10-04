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

//! Scatter one StringView source into multiple in-progress arrays.
//!
//! The caller assigns rows to targets; this module preserves that assignment
//! and each target's row order while managing payload copies and completed arrays.
//! Entry point: [`scatter_string_views`].

use crate::coalesce::InProgressArray;
use crate::coalesce::byte_view::InProgressByteViewArray;
use arrow_array::cast::AsArray;
use arrow_array::types::StringViewType;
use arrow_array::{Array, ArrayRef, StringViewArray};
use arrow_data::{ByteView, MAX_INLINE_VIEW_LEN};
use arrow_schema::ArrowError;
use std::collections::VecDeque;
use std::sync::Arc;

/// Mutable destination state for one column in one bucket.
pub struct StringViewScatterTarget<'a> {
    /// Builder receiving the rows assigned to this target.
    pub array: &'a mut InProgressByteViewArray<StringViewType>,
    /// Buffered row count, kept in sync with the builder.
    pub rows: &'a mut usize,
    /// Completed arrays, emitted when the builder reaches its batch size.
    pub finished: &'a mut VecDeque<ArrayRef>,
}

/// Scatter one StringView array in source order into all destinations.
///
/// `bucket_ids[row]` selects the target for each source row. Each destination
/// uses the same payload copy/GC decision as coalescing its selected source
/// range: copy when the retained buffers exceed twice the selected payload size.
/// Inline values, nulls, and row order within each target are preserved.
/// Completed arrays are appended to each target's `finished` queue.
///
/// # Panics
///
/// Panics if the source is not StringView, row counts differ, a target index is
/// out of bounds, or a target's buffered row count is inconsistent with its builder.
/// Target batch sizes must be positive, and buffered rows must be below that size.
pub fn scatter_string_views(
    source: &ArrayRef,
    bucket_ids: &[usize],
    targets: &mut [StringViewScatterTarget<'_>],
) -> Result<(), ArrowError> {
    let values = source.as_string_view();
    assert_eq!(values.len(), bucket_ids.len());
    let mut bytes = vec![0usize; targets.len()];
    let mut rows = vec![0usize; targets.len()];
    for (&view, &bucket) in values.views().iter().zip(bucket_ids) {
        let length = view as u32;
        bytes[bucket] += if length > MAX_INLINE_VIEW_LEN {
            length as usize
        } else {
            0
        };
        rows[bucket] += 1;
    }
    let actual_bytes: usize = values.data_buffers().iter().map(|b| b.capacity()).sum();
    let gc: Vec<bool> = bytes
        .iter()
        .map(|&used| used != 0 && actual_bytes > 2 * used)
        .collect();
    for (index, target) in targets.iter_mut().enumerate() {
        assert_eq!(*target.rows, target.array.views.len());
        assert!(*target.rows < target.array.batch_size);
        if rows[index] != 0 {
            target
                .array
                .set_source_with_size(Some(Arc::clone(source)), bytes[index]);
            target.array.ensure_capacity();
        }
    }
    let result = if values.null_count() == 0 {
        scatter_inner::<false>(values, bucket_ids, targets, &gc, &mut bytes, &mut rows)
    } else {
        scatter_inner::<true>(values, bucket_ids, targets, &gc, &mut bytes, &mut rows)
    };
    for target in targets {
        target.array.set_source(None);
    }
    result
}

fn scatter_inner<const HAS_NULLS: bool>(
    source: &StringViewArray,
    bucket_ids: &[usize],
    targets: &mut [StringViewScatterTarget<'_>],
    gc: &[bool],
    bytes: &mut [usize],
    rows: &mut [usize],
) -> Result<(), ArrowError> {
    // A source's shared buffers must be registered once per output batch.
    let mut shared_base = vec![None; targets.len()];
    if !HAS_NULLS {
        for (index, target) in targets.iter_mut().enumerate() {
            target
                .array
                .nulls
                .append_n_non_nulls(rows[index].min(target.array.batch_size - *target.rows));
        }
    }
    for (row, (&view, &bucket)) in source.views().iter().zip(bucket_ids).enumerate() {
        let target = &mut targets[bucket];
        let array = &mut *target.array;
        let mut view = ByteView::from(view);
        if HAS_NULLS {
            array.nulls.append(source.is_valid(row));
        }
        if view.length > MAX_INLINE_VIEW_LEN {
            let length = view.length as usize;
            if gc[bucket] {
                if array
                    .current
                    .as_ref()
                    .is_none_or(|b| b.capacity() - b.len() < length)
                {
                    array.finish_current();
                    array.current = Some(array.buffer_source.next_buffer(bytes[bucket]));
                }
                let dst = array.current.as_mut().unwrap();
                let src = &source.data_buffers()[view.buffer_index as usize]
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
            } else {
                let base = *shared_base[bucket].get_or_insert_with(|| {
                    array.finish_current();
                    let base: u32 = array
                        .completed
                        .len()
                        .try_into()
                        .expect("buffer index fits u32");
                    array.completed.extend_from_slice(source.data_buffers());
                    array.size_of_completed_buffers_from_current_source =
                        source.data_buffers().iter().map(|b| b.capacity()).sum();
                    base
                });
                view.buffer_index = view
                    .buffer_index
                    .checked_add(base)
                    .expect("buffer index fits u32");
            }
            bytes[bucket] -= length;
        }
        array.views.push(view.as_u128());
        *target.rows += 1;
        rows[bucket] -= 1;
        if *target.rows == array.batch_size {
            target.finished.push_back(array.finish()?);
            *target.rows = 0;
            shared_base[bucket] = None;
            if rows[bucket] != 0 {
                array.ensure_capacity();
                if !HAS_NULLS {
                    array
                        .nulls
                        .append_n_non_nulls(rows[bucket].min(array.batch_size));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::coalesce::byte_view::scatter::{self, StringViewScatterTarget};
    use crate::coalesce::{InProgressArray, InProgressByteViewArray};
    use arrow_array::cast::AsArray;
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, ArrayRef, StringViewArray};
    use std::collections::VecDeque;
    use std::sync::Arc;

    #[test]
    fn scatter_preserves_values_across_batches() {
        // One bucket exercises buffer sharing; 32 exercise GC. Tiny output
        // batches cover filling, draining and continuing within one source.
        for bucket_count in [1, 32] {
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
                            } else if row % 5 == 0 {
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
                    scatter::scatter_string_views(&source, &ids, &mut targets).unwrap();
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
