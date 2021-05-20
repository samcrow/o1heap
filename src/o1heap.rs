// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
// OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Copyright (c) 2020 Pavel Kirienko
// Authors: Pavel Kirienko <pavel.kirienko@zubax.com>

#[cfg(test)]
mod tests;

use core::mem::size_of;
use core::ptr;

/// Checks a condition at compile time
macro_rules! const_assert {
    ($x:expr $(,)?) => {
        #[allow(unknown_lints, clippy::eq_op)]
        const _: [(); 0 - !{
            const ASSERT: bool = $x;
            ASSERT
        } as usize] = [];
    };
}

/// The guaranteed alignment depends on the platform pointer width.
pub const O1HEAP_ALIGNMENT: usize = 4 * size_of::<*const ()>();
/// The overhead is at most O1HEAP_ALIGNMENT bytes large,
/// then follows the user data which shall keep the next fragment aligned.
const FRAGMENT_SIZE_MIN: usize = O1HEAP_ALIGNMENT * 2;
/// This is risky, handle with care: if the allocation amount plus per-fragment overhead exceeds 2**(b-1),
/// where b is the pointer bit width, then ceil(log2(amount)) yields b; then 2**b causes an integer overflow.
/// To avoid this, we put a hard limit on fragment size (which is amount + per-fragment overhead): 2**(b-1)
const FRAGMENT_SIZE_MAX: usize = (usize::MAX >> 1) + 1;

/// Normally we should subtract log2(FRAGMENT_SIZE_MIN) but log2 is bulky to compute using the preprocessor only.
/// We will certainly end up with unused bins this way, but it is cheap to ignore.
const NUM_BINS_MAX: usize = size_of::<usize>() * 8;

const_assert!(is_power_of_two(O1HEAP_ALIGNMENT));
const_assert!(is_power_of_two(FRAGMENT_SIZE_MIN));
const_assert!(is_power_of_two(FRAGMENT_SIZE_MAX));

/// The amount of space allocated for the heap instance
///
/// Its size is padded up to O1HEAP_ALIGNMENT to ensure correct alignment of the allocation arena that follows.
const INSTANCE_SIZE_PADDED: usize =
    (size_of::<O1HeapInstance>() + O1HEAP_ALIGNMENT - 1) & !(O1HEAP_ALIGNMENT - 1);

const_assert!(INSTANCE_SIZE_PADDED >= size_of::<O1HeapInstance>());
const_assert!(INSTANCE_SIZE_PADDED % O1HEAP_ALIGNMENT == 0);

#[derive(Debug)]
pub struct O1HeapInstance {
    bins: [*mut Fragment; NUM_BINS_MAX],
    nonempty_bin_mask: usize,
    diagnostics: O1HeapDiagnostics,
}

/// Runtime diagnostic information
///
/// This information can be used to facilitate runtime self-testing,
/// as required by certain safety-critical development guidelines.
/// If assertion checks are not disabled, the library will perform automatic runtime self-diagnostics that trigger
/// an assertion failure if a heap corruption is detected.
/// Health checks and validation can be done with @ref o1heapDoInvariantsHold().
#[derive(Debug, Clone)]
pub struct O1HeapDiagnostics {
    /// The total amount of memory available for serving allocation requests (heap size).
    /// The maximum allocation size is (capacity - O1HEAP_ALIGNMENT).
    /// This parameter does not include the overhead used up by @ref O1HeapInstance and arena alignment.
    /// This parameter is constant.
    pub capacity: usize,
    /// The amount of memory that is currently allocated, including the per-fragment overhead and size alignment.
    /// For example, if the application requested a fragment of size 1 byte, the value reported here may be 32 bytes.
    pub allocated: usize,
    /// The maximum value of 'allocated' seen since initialization. This parameter is never decreased.
    pub peak_allocated: usize,
    /// The largest amount of memory that the allocator has attempted to allocate (perhaps unsuccessfully)
    /// since initialization (not including the rounding and the allocator's own per-fragment overhead,
    /// so the total is larger). This parameter is never decreased. The initial value is zero.
    pub peak_request_size: usize,
    /// The number of times an allocation request could not be completed due to the lack of memory or
    /// excessive fragmentation. OOM stands for "out of memory". This parameter is never decreased.
    pub oom_count: u64,
}

struct Fragment {
    header: FragmentHeader,
    next_free: *mut Fragment,
    prev_free: *mut Fragment,
}
const_assert!(size_of::<Fragment>() <= FRAGMENT_SIZE_MIN);

struct FragmentHeader {
    next: *mut Fragment,
    prev: *mut Fragment,
    size: usize,
    used: bool,
}
const_assert!(size_of::<FragmentHeader>() <= O1HEAP_ALIGNMENT);

/// Returns true if the argument is an integer power of two or zero.
#[inline]
const fn is_power_of_two(x: usize) -> bool {
    x & x.wrapping_sub(1) == 0
}
/// Special case: if the argument is zero, returns zero.
#[inline]
fn log_2_floor(mut x: usize) -> u8 {
    let mut y: u8 = 0;
    // This is currently the only exception to the statement "routines contain neither loops nor recursion".
    // It is unclear if there is a better way to compute the binary logarithm than this.
    while x > 1 {
        x >>= 1;
        y += 1;
    }
    y
}
// / Special case: if the argument is zero, returns zero.
#[inline]
fn log_2_ceil(x: usize) -> u8 {
    log_2_floor(x).wrapping_add(if is_power_of_two(x) { 0 } else { 1 })
}
/// Raise 2 into the specified power
#[inline]
fn pow2(power: u8) -> usize {
    1usize << power
}

/// Links two fragments so that their next/prev pointers point to each other; left goes before right.
#[inline]
unsafe fn interlink(left: *mut Fragment, right: *mut Fragment) {
    if !left.is_null() {
        (*left).header.next = right
    }
    if !right.is_null() {
        (*right).header.prev = left
    };
}
/// Adds a new block into the appropriate bin and updates the lookup mask.
#[inline]
unsafe fn rebin(handle: *mut O1HeapInstance, fragment: *mut Fragment) {
    assert!(!handle.is_null());
    assert!(!fragment.is_null());
    assert!((*fragment).header.size >= FRAGMENT_SIZE_MIN);
    assert_eq!((*fragment).header.size % FRAGMENT_SIZE_MIN, 0);
    let idx: u8 = log_2_floor((*fragment).header.size / FRAGMENT_SIZE_MIN);
    assert!((idx as usize) < NUM_BINS_MAX);
    // Add the new fragment to the beginning of the bin list.
    // I.e., each allocation will be returning the least-recently-used fragment -- good for caching.
    (*fragment).next_free = (*handle).bins[idx as usize];
    (*fragment).prev_free = ptr::null_mut();
    if !(*handle).bins[idx as usize].is_null() {
        (*(*handle).bins[idx as usize]).prev_free = fragment
    }
    (*handle).bins[idx as usize] = fragment;
    (*handle).nonempty_bin_mask |= pow2(idx);
}
/// Removes the specified block from its bin.
#[inline]
unsafe fn unbin(handle: *mut O1HeapInstance, fragment: *const Fragment) {
    assert!(!handle.is_null());
    assert!(!fragment.is_null());
    assert!((*fragment).header.size >= FRAGMENT_SIZE_MIN);
    assert_eq!((*fragment).header.size % FRAGMENT_SIZE_MIN, 0);
    let idx: u8 = log_2_floor((*fragment).header.size / FRAGMENT_SIZE_MIN);
    assert!((idx as usize) < NUM_BINS_MAX);
    // Remove the bin from the free fragment list.
    if !(*fragment).next_free.is_null() {
        (*(*fragment).next_free).prev_free = (*fragment).prev_free
    }
    if !(*fragment).prev_free.is_null() {
        (*(*fragment).prev_free).next_free = (*fragment).next_free
    }
    // Update the bin header.
    if (*handle).bins[idx as usize] as usize == fragment as usize {
        assert!((*fragment).prev_free.is_null());
        (*handle).bins[idx as usize] = (*fragment).next_free;
        if (*handle).bins[idx as usize].is_null() {
            (*handle).nonempty_bin_mask &= !pow2(idx)
        }
    };
}
// ---------------------------------------- PUBLIC API IMPLEMENTATION ----------------------------------------

/// Initializes a heap
///
/// The arena base pointer shall be aligned at @ref O1HEAP_ALIGNMENT, otherwise NULL is returned.
///
/// The total heap capacity cannot exceed approx. (SIZE_MAX/2). If the arena size allows for a larger heap,
/// the excess will be silently truncated away (no error). This is not a realistic use case because a typical
/// application is unlikely to be able to dedicate that much of the address space for the heap.
///
/// The critical section enter/leave callbacks will be invoked when the allocator performs an atomic transaction.
/// There is at most one atomic transaction per allocation/deallocation.
/// Either or both of the callbacks may be NULL if locking is not needed (i.e., the heap is not shared).
/// It is guaranteed that a critical section will never be entered recursively.
/// It is guaranteed that 'enter' is invoked the same number of times as 'leave', unless either of them are NULL.
/// It is guaranteed that 'enter' is invoked before 'leave', unless either of them are NULL.
/// The callbacks are never invoked from the initialization function itself.
///
/// The function initializes a new heap instance allocated in the provided arena, taking some of its space for its
/// own needs (normally about 40..600 bytes depending on the architecture, but this parameter is not characterized).
/// A pointer to the newly initialized instance is returned.
///
/// If the provided space is insufficient, NULL is returned.
///
/// An initialized instance does not hold any resources. Therefore, if the instance is no longer needed,
/// it can be discarded without any de-initialization procedures.
///
/// The time complexity is unspecified.
pub unsafe fn o1heap_init(base: *mut (), size: usize) -> *mut O1HeapInstance {
    let mut out: *mut O1HeapInstance = ptr::null_mut();
    if !base.is_null()
        && (base as usize) % O1HEAP_ALIGNMENT == 0
        && size >= (INSTANCE_SIZE_PADDED + FRAGMENT_SIZE_MIN)
    {
        // Allocate the core heap metadata structure in the beginning of the arena.
        assert_eq!(base as usize % size_of::<*mut O1HeapInstance>(), 0);
        out = base as *mut O1HeapInstance;
        (*out).nonempty_bin_mask = 0;
        for i in 0..NUM_BINS_MAX {
            (*out).bins[i] = ptr::null_mut();
        }

        // Limit and align the capacity.
        let mut capacity: usize = size - INSTANCE_SIZE_PADDED;
        if capacity > FRAGMENT_SIZE_MAX {
            capacity = FRAGMENT_SIZE_MAX;
        }
        while capacity % FRAGMENT_SIZE_MIN != 0 {
            assert!(capacity > 0);
            capacity -= 1;
        }
        assert_eq!(capacity % FRAGMENT_SIZE_MIN, 0);
        assert!((FRAGMENT_SIZE_MIN..=FRAGMENT_SIZE_MAX).contains(&capacity));
        // Initialize the root fragment.
        let frag: *mut Fragment = (base as usize + INSTANCE_SIZE_PADDED) as *mut Fragment;
        assert_eq!(frag as usize % O1HEAP_ALIGNMENT, 0);
        (*frag).header.next = ptr::null_mut();
        (*frag).header.prev = ptr::null_mut();
        (*frag).header.size = capacity;
        (*frag).header.used = false;
        (*frag).next_free = ptr::null_mut();
        (*frag).prev_free = ptr::null_mut();
        rebin(out, frag);
        assert_ne!((*out).nonempty_bin_mask, 0);
        // Initialize the diagnostics.
        (*out).diagnostics.capacity = capacity;
        (*out).diagnostics.allocated = 0;
        (*out).diagnostics.peak_allocated = 0;
        (*out).diagnostics.peak_request_size = 0;
        (*out).diagnostics.oom_count = 0;
    }
    out
}

/// Attempts to allocate memory
///
/// The semantics follows malloc() with additional guarantees the full list of which is provided below.
///
/// If the allocation request is served successfully, a pointer to the newly allocated memory fragment is returned.
/// The returned pointer is guaranteed to be aligned to [`O1HEAP_ALIGNMENT`](#const.O1HEAP_ALIGNMENT).
///
/// If the allocation request cannot be served due to the lack of memory or its excessive fragmentation,
/// a NULL pointer is returned.
///
/// The function is executed in constant time (unless the critical section management hooks are used and are not
/// constant-time). The allocated memory is NOT zero-filled (because zero-filling is a variable-complexity operation).
///
/// The function may invoke critical_section_enter and critical_section_leave at most once each (NULL hooks ignored).
pub unsafe fn o1heap_allocate(handle: *mut O1HeapInstance, amount: usize) -> *mut u8 {
    assert!(!handle.is_null());
    assert!((*handle).diagnostics.capacity <= FRAGMENT_SIZE_MAX);
    let mut out: *mut u8 = ptr::null_mut();
    // If the amount approaches approx. SIZE_MAX/2, an undetected integer overflow may occur.
    // To avoid that, we do not attempt allocation if the amount exceeds the hard limit.
    // We perform multiple redundant checks to account for a possible unaccounted overflow.
    if amount > 0 && amount <= ((*handle).diagnostics.capacity - O1HEAP_ALIGNMENT) {
        // Add the header size and align the allocation size to the power of 2.
        // See "Timing-Predictable Memory Allocation In Hard Real-Time Systems", Herter, page 27.
        let fragment_size: usize = pow2(log_2_ceil(amount + O1HEAP_ALIGNMENT));
        assert!(fragment_size <= FRAGMENT_SIZE_MAX);
        assert!(fragment_size >= FRAGMENT_SIZE_MIN);
        assert!(fragment_size >= amount + O1HEAP_ALIGNMENT);
        assert!(is_power_of_two(fragment_size));
        let optimal_bin_index: u8 = log_2_ceil(fragment_size / FRAGMENT_SIZE_MIN); // Use CEIL when fetching.
        assert!((optimal_bin_index as usize) < NUM_BINS_MAX);
        let candidate_bin_mask: usize = !pow2(optimal_bin_index).wrapping_sub(1);

        // Find the smallest non-empty bin we can use.
        let suitable_bins: usize = (*handle).nonempty_bin_mask & candidate_bin_mask;
        // Clear all bits but the lowest.
        let smallest_bin_mask: usize = suitable_bins & !suitable_bins.wrapping_sub(1);
        if smallest_bin_mask != 0 {
            assert!(is_power_of_two(smallest_bin_mask));
            let bin_index: u8 = log_2_floor(smallest_bin_mask);
            assert!(bin_index >= optimal_bin_index);
            assert!((bin_index as usize) < NUM_BINS_MAX);
            // The bin we found shall not be empty, otherwise it's a state divergence (memory corruption?).
            let frag: *mut Fragment = (*handle).bins[bin_index as usize];
            assert!(!frag.is_null());
            assert!((*frag).header.size >= fragment_size);
            assert_eq!((*frag).header.size % FRAGMENT_SIZE_MIN, 0);
            assert!(!(*frag).header.used);
            unbin(handle, frag);

            // Split the fragment if it is too large.
            let leftover: usize = (*frag).header.size.wrapping_sub(fragment_size); // Overflow check.
            (*frag).header.size = fragment_size; // Alignment check.
            assert!(leftover < (*handle).diagnostics.capacity);
            assert_eq!(leftover % FRAGMENT_SIZE_MIN, 0);
            if leftover >= FRAGMENT_SIZE_MIN {
                let new_frag: *mut Fragment = (frag as usize + fragment_size) as *mut Fragment;
                assert_eq!((new_frag as usize) % O1HEAP_ALIGNMENT, 0);
                (*new_frag).header.size = leftover;
                (*new_frag).header.used = false;
                interlink(new_frag, (*frag).header.next);
                interlink(frag, new_frag);
                rebin(handle, new_frag);
            }
            // Update the diagnostics.
            assert_eq!((*handle).diagnostics.allocated % FRAGMENT_SIZE_MIN, 0);
            (*handle).diagnostics.allocated += fragment_size;
            assert!((*handle).diagnostics.allocated <= (*handle).diagnostics.capacity);
            if (*handle).diagnostics.peak_allocated < (*handle).diagnostics.allocated {
                (*handle).diagnostics.peak_allocated = (*handle).diagnostics.allocated
            }
            // Finalize the fragment we just allocated.
            assert!((*frag).header.size >= amount + O1HEAP_ALIGNMENT);
            (*frag).header.used = true;
            out = (frag as usize + O1HEAP_ALIGNMENT) as *mut u8
        }
    }
    // Update the diagnostics.
    if (*handle).diagnostics.peak_request_size < amount {
        (*handle).diagnostics.peak_request_size = amount
    }
    if out.is_null() && amount > 0 {
        (*handle).diagnostics.oom_count = (*handle).diagnostics.oom_count.wrapping_add(1)
    }
    out
}

/// Frees allocated memory
///
/// The semantics follows free() with additional guarantees the full list of which is provided below.
///
/// If the pointer does not point to a previously allocated block and is not NULL, the behavior is undefined.
/// Builds where assertion checks are enabled may trigger an assertion failure for some invalid inputs.
///
/// The function is executed in constant time (unless the critical section management hooks are used and are not
/// constant-time).
///
/// The function may invoke critical_section_enter and critical_section_leave at most once each (NULL hooks ignored).
pub unsafe fn o1heap_free(handle: *mut O1HeapInstance, pointer: *mut u8) {
    assert!(!handle.is_null());
    assert!((*handle).diagnostics.capacity <= FRAGMENT_SIZE_MAX);
    // NULL pointer is a no-op.
    if !pointer.is_null() {
        let frag: *mut Fragment = (pointer as usize - O1HEAP_ALIGNMENT) as *mut Fragment;
        // Check for heap corruption in debug builds.
        assert_eq!((frag as usize) % size_of::<*mut Fragment>(), 0);
        assert!(frag as usize >= handle as usize + INSTANCE_SIZE_PADDED);
        assert!(
            frag as usize
                <= handle as usize + INSTANCE_SIZE_PADDED + (*handle).diagnostics.capacity
                    - FRAGMENT_SIZE_MIN
        );
        assert!((*frag).header.used); // Catch double-free
        assert_eq!(
            ((*frag).header.next as usize) % size_of::<*mut Fragment>(),
            0
        );
        assert_eq!(
            ((*frag).header.prev as usize) % size_of::<*mut Fragment>(),
            0
        );
        assert!((*frag).header.size >= FRAGMENT_SIZE_MIN);
        assert!((*frag).header.size <= (*handle).diagnostics.capacity);
        assert_eq!((*frag).header.size % FRAGMENT_SIZE_MIN, 0);

        // Even if we're going to drop the fragment later, mark it free anyway to prevent double-free.
        (*frag).header.used = false;
        // Update the diagnostics. It must be done before merging because it invalidates the fragment size information.
        assert!((*handle).diagnostics.allocated >= (*frag).header.size);
        (*handle).diagnostics.allocated -= (*frag).header.size;
        // Merge with siblings and insert the returned fragment into the appropriate bin and update metadata.
        let prev: *mut Fragment = (*frag).header.prev;
        let next: *mut Fragment = (*frag).header.next;
        let join_left: bool = !prev.is_null() && !(*prev).header.used;
        let join_right: bool = !next.is_null() && !(*next).header.used;
        if join_left && join_right {
            // [ prev ][ this ][ next ] => [ ------- prev ------- ]
            unbin(handle, prev); // Invalidate the dropped fragment headers to prevent double-free.
            unbin(handle, next);
            (*prev).header.size += (*frag).header.size + (*next).header.size;
            (*frag).header.size = 0;
            (*next).header.size = 0;
            assert_eq!((*prev).header.size % FRAGMENT_SIZE_MIN, 0);
            interlink(prev, (*next).header.next);
            rebin(handle, prev);
        } else if join_left {
            // [ prev ][ this ][ next ] => [ --- prev --- ][ next ]
            unbin(handle, prev);
            (*prev).header.size += (*frag).header.size;
            (*frag).header.size = 0;
            assert_eq!((*prev).header.size % FRAGMENT_SIZE_MIN, 0);
            interlink(prev, next);
            rebin(handle, prev);
        } else if join_right {
            // [ prev ][ this ][ next ] => [ prev ][ --- this --- ]
            unbin(handle, next);
            (*frag).header.size += (*next).header.size;
            (*next).header.size = 0;
            assert_eq!((*frag).header.size % FRAGMENT_SIZE_MIN, 0);
            interlink(frag, (*next).header.next);
            rebin(handle, frag);
        } else {
            rebin(handle, frag);
        }
    };
}

/// Performs a basic sanity check on the heap
///
/// This function can be used as a weak but fast method of heap corruption detection.
/// It invokes critical_section_enter once (unless NULL) and then critical_section_leave once (unless NULL).
/// If the handle pointer is NULL, the behavior is undefined.
/// The time complexity is constant.
/// The return value is truth if the heap looks valid, falsity otherwise.
pub unsafe fn o1heap_do_invariants_hold(handle: *const O1HeapInstance) -> bool {
    assert!(!handle.is_null());
    let mut valid: bool = true;
    // Check the bin mask consistency.
    for i in 0..NUM_BINS_MAX {
        let mask_bit_set: bool = (*handle).nonempty_bin_mask & pow2(i as u8) != 0;
        let bin_nonempty: bool = !(*handle).bins[i as usize].is_null();
        valid = valid && mask_bit_set == bin_nonempty;
    }

    // Create a local copy of the diagnostics struct to check later and release the critical section early.
    let diag: O1HeapDiagnostics = (*handle).diagnostics.clone();
    // Capacity check.
    valid = valid
        && diag.capacity <= FRAGMENT_SIZE_MAX
        && diag.capacity >= FRAGMENT_SIZE_MIN
        && diag.capacity % FRAGMENT_SIZE_MIN == 0;
    // Allocation info check.
    valid = valid
        && diag.allocated <= diag.capacity
        && diag.allocated % FRAGMENT_SIZE_MIN == 0
        && diag.peak_allocated <= diag.capacity
        && diag.peak_allocated >= diag.allocated
        && diag.peak_allocated % FRAGMENT_SIZE_MIN == 0;
    // Peak request check
    valid = valid && (diag.peak_request_size < diag.capacity || diag.oom_count > 0);
    if diag.peak_request_size == 0 {
        valid = valid && diag.peak_allocated == 0 && diag.allocated == 0 && diag.oom_count == 0
    } else {
        // Overflow on summation is possible but safe to ignore.
        valid = valid
            && (diag.peak_request_size.wrapping_add(O1HEAP_ALIGNMENT) <= diag.peak_allocated
                || diag.oom_count > 0)
    }
    valid
}

/// Samples and returns a copy of the diagnostic information, see @ref O1HeapDiagnostics
///
/// This function merely copies the structure from an internal storage, so it is fast to return.
/// It invokes critical_section_enter once (unless NULL) and then critical_section_leave once (unless NULL).
/// If the handle pointer is NULL, the behavior is undefined.
pub unsafe fn o1heap_get_diagnostics(handle: *const O1HeapInstance) -> O1HeapDiagnostics {
    assert!(!handle.is_null());
    (*handle).diagnostics.clone()
}
