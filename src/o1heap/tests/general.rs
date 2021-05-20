//! General tests that check the heap internals

extern crate alloc;

use core::mem::size_of;
use std::cmp;
use std::ptr;
use std::slice;
use std::vec::Vec;

use crate::o1heap::{
    log_2_floor, o1heap_allocate, o1heap_do_invariants_hold, o1heap_free, o1heap_get_diagnostics,
    o1heap_init, Fragment, O1HeapDiagnostics, O1HeapInstance, FRAGMENT_SIZE_MAX, FRAGMENT_SIZE_MIN,
    O1HEAP_ALIGNMENT,
};
use core::alloc::Layout;
use core::cmp::max;
use core::ptr::NonNull;
use rand::rngs::ThreadRng;
use rand::{thread_rng, Fill, Rng};

const KIBIBYTE: usize = 1024;
const MEBIBYTE: usize = KIBIBYTE * KIBIBYTE;

/// An array of bytes aligned to 128 bytes
struct Arena {
    ptr: NonNull<u8>,
    length: usize,
}

const ARENA_ALIGN: usize = 128;

impl Arena {
    /// Creates an arena with all bytes set to zero
    pub fn zeroed(length: usize) -> Self {
        let layout = Layout::from_size_align(length, ARENA_ALIGN).unwrap();
        unsafe {
            let ptr = alloc::alloc::alloc(layout);
            let non_null_ptr = NonNull::new(ptr).expect("Failed to allocate memory");
            // Initialize everything. This eliminates the possibility of
            // creating a reference or slice to uninitialized memory later.
            ptr::write_bytes(non_null_ptr.as_ptr(), 0u8, length);
            Arena {
                ptr: non_null_ptr,
                length,
            }
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.length) }
    }

    /// Returns a pointer to the beginning of this arena
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.length, ARENA_ALIGN).unwrap();
        unsafe {
            alloc::alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

#[test]
fn general_init() {
    let mut arena = Arena::zeroed(10_000);
    unsafe {
        assert!(init(ptr::null_mut(), 0).is_none());
        assert!(init(arena.as_mut_ptr(), 0).is_none());
        assert!(init(arena.as_mut_ptr(), 99).is_none());

        // Check various offsets and sizes to make sure the initialization is done correctly in all cases.
        for offset in 0..7 {
            for size in (99..5100).step_by(111) {
                let arena_ptr = arena.as_mut_ptr();
                let heap = init(arena_ptr.add(offset), size - offset);
                if let Some(heap) = heap {
                    assert!(size >= size_of::<O1HeapInstance>() + FRAGMENT_SIZE_MIN);
                    assert!(heap.0 as usize >= arena_ptr as usize);
                    assert_eq!(heap.0 as usize % O1HEAP_ALIGNMENT, 0);
                    assert!(heap.do_invariants_hold());
                }
            }
        }
    }
}

#[test]
fn allocate_oom() {
    const MIB_256: usize = 256 * MEBIBYTE;
    const ARENA_SIZE: usize = MIB_256 + MEBIBYTE;
    unsafe {
        let mut arena = Arena::zeroed(ARENA_SIZE);
        let mut heap = init(arena.as_mut_ptr(), ARENA_SIZE).unwrap();
        assert!((*heap.0).diagnostics.capacity > ARENA_SIZE - 1024);
        assert!((*heap.0).diagnostics.capacity < ARENA_SIZE);
        assert_eq!((*heap.0).diagnostics.oom_count, 0);

        assert!(heap.allocate(ARENA_SIZE).is_null());
        assert_eq!((*heap.0).diagnostics.oom_count, 1);

        assert!(heap.allocate(ARENA_SIZE - O1HEAP_ALIGNMENT).is_null());
        assert_eq!((*heap.0).diagnostics.oom_count, 2);

        assert!(heap
            .allocate((*heap.0).diagnostics.capacity - O1HEAP_ALIGNMENT + 1)
            .is_null());
        assert_eq!((*heap.0).diagnostics.oom_count, 3);

        assert!(heap.allocate(ARENA_SIZE * 10).is_null());
        assert_eq!((*heap.0).diagnostics.oom_count, 4);

        assert!(heap.allocate(0).is_null()); // Nothing to allocate
        assert_eq!((*heap.0).diagnostics.oom_count, 4);

        assert_eq!((*heap.0).diagnostics.peak_allocated, 0);
        assert_eq!((*heap.0).diagnostics.allocated, 0);
        assert_eq!((*heap.0).diagnostics.peak_request_size, ARENA_SIZE * 10);

        // Maximum possible allocation
        assert!(!heap.allocate(MIB_256 - O1HEAP_ALIGNMENT).is_null());
        assert_eq!((*heap.0).diagnostics.oom_count, 4);
        assert_eq!((*heap.0).diagnostics.peak_allocated, MIB_256);
        assert_eq!((*heap.0).diagnostics.allocated, MIB_256);
        assert_eq!((*heap.0).diagnostics.peak_request_size, ARENA_SIZE * 10);

        assert!(heap.do_invariants_hold());
    }
}

#[test]
fn allocate_smallest() {
    let arena_size = 300 * MEBIBYTE;
    let mut arena = Arena::zeroed(arena_size);

    unsafe {
        let mut heap = init(arena.as_mut_ptr(), arena_size).unwrap();

        let mem = heap.allocate(1);
        assert!(!mem.is_null());
        assert_eq!((*heap.0).diagnostics.oom_count, 0);
        assert_eq!((*heap.0).diagnostics.peak_allocated, FRAGMENT_SIZE_MIN);
        assert_eq!((*heap.0).diagnostics.allocated, FRAGMENT_SIZE_MIN);
        assert_eq!((*heap.0).diagnostics.peak_request_size, 1);

        let fragment = Fragment::from_allocated_memory(mem as *const u8);
        assert_eq!(fragment.header.size, O1HEAP_ALIGNMENT * 2);
        assert!(!fragment.header.next.is_null());
        assert!(fragment.header.prev.is_null());
        assert!(fragment.header.used);
        assert_eq!(
            (*fragment.header.next).header.size,
            (*heap.0).diagnostics.capacity - fragment.header.size
        );
        assert!(!(*fragment.header.next).header.used);

        heap.free(mem);
        assert!(heap.do_invariants_hold());
    }
}

#[test]
fn allocate_size_overflow() {
    let arena_size = 300 * MEBIBYTE;
    let mut arena = Arena::zeroed(arena_size);
    unsafe {
        let mut heap = init(arena.as_mut_ptr(), arena_size).unwrap();
        assert!((*heap.0).diagnostics.capacity > arena_size - 1024);
        assert!((*heap.0).diagnostics.capacity < arena_size);

        for i in 1..=2 {
            assert!(heap.allocate(usize::MAX / i).is_null());
            assert!(heap.allocate((usize::MAX / i).wrapping_add(1)).is_null());
            assert!(heap.allocate((usize::MAX / i) - 1).is_null());
            assert!(heap
                .allocate(FRAGMENT_SIZE_MAX - O1HEAP_ALIGNMENT + 1)
                .is_null());
        }
    }
    unsafe {
        // Over-commit the arena -- it is SMALLER than the size we're providing; it's an UB but for a test it's acceptable.
        let mut heap = init(arena.as_mut_ptr(), usize::MAX).unwrap();
        assert_eq!((*heap.0).diagnostics.capacity, FRAGMENT_SIZE_MAX);

        for i in 1..=2 {
            assert!(heap.allocate(usize::MAX / i).is_null());
            assert!(heap.allocate((usize::MAX / i).wrapping_add(1)).is_null());
            assert!(heap.allocate((usize::MAX / i) - 1).is_null());
            assert!(heap
                .allocate(FRAGMENT_SIZE_MAX - O1HEAP_ALIGNMENT + 1)
                .is_null());
        }

        // Make sure the max-sized fragments are allocatable
        let mem = heap.allocate(FRAGMENT_SIZE_MAX - O1HEAP_ALIGNMENT);
        assert!(!mem.is_null());
        let fragment = Fragment::from_allocated_memory(mem as *const u8);
        assert_eq!(fragment.header.size, FRAGMENT_SIZE_MAX);
        assert!(fragment.header.prev.is_null());
        assert!(fragment.header.next.is_null());
        assert!(fragment.header.used);

        assert_eq!((*heap.0).diagnostics.peak_allocated, FRAGMENT_SIZE_MAX);
        assert_eq!((*heap.0).diagnostics.allocated, FRAGMENT_SIZE_MAX);

        assert_eq!((*heap.0).nonempty_bin_mask, 0);
        assert!((*heap.0)
            .bins
            .iter()
            .all(|fragment_ptr| fragment_ptr.is_null()));

        assert!(heap.do_invariants_hold());
    }
}

#[test]
fn general_free() {
    let arena_size = 4096 + size_of::<O1HeapInstance>() + O1HEAP_ALIGNMENT - 1;
    let mut arena = Arena::zeroed(arena_size);
    unsafe {
        let mut heap = init(arena.as_mut_ptr(), arena_size).unwrap();

        assert!(heap.allocate(0).is_null());
        assert_eq!((*heap.0).diagnostics.allocated, 0);
        heap.free(ptr::null_mut());
        assert_eq!((*heap.0).diagnostics.peak_allocated, 0);
        assert_eq!((*heap.0).diagnostics.peak_request_size, 0);
        assert_eq!((*heap.0).diagnostics.oom_count, 0);

        struct AllocChecker<'h> {
            heap: &'h mut TestHeap,
            allocated: usize,
            peak_allocated: usize,
            peak_request_size: usize,
        }
        impl<'h> AllocChecker<'h> {
            pub fn new(heap: &'h mut TestHeap) -> Self {
                AllocChecker {
                    heap,
                    allocated: 0,
                    peak_allocated: 0,
                    peak_request_size: 0,
                }
            }
            fn alloc(&mut self, amount: usize, reference: &[(bool, usize)]) -> *mut u8 {
                let p = self.heap.allocate(amount);
                if amount != 0 {
                    assert!(!p.is_null());
                    // Overwrite all to ensure that the allocator does not make implicit assumptions about the memory use
                    {
                        let allocated_slice =
                            unsafe { slice::from_raw_parts_mut(p as *mut u8, amount) };
                        let mut rng = rand::thread_rng();
                        allocated_slice.try_fill(&mut rng).unwrap();
                    }
                    let frag: &'h Fragment =
                        unsafe { Fragment::from_allocated_memory(p as *const u8) };
                    assert!(frag.header.used);
                    assert_eq!(frag.header.size & (frag.header.size - 1), 0);
                    assert!(frag.header.size >= amount + O1HEAP_ALIGNMENT);
                    assert!(frag.header.size <= FRAGMENT_SIZE_MAX);

                    self.allocated += frag.header.size;
                    self.peak_allocated = max(self.peak_allocated, self.allocated);
                    self.peak_request_size = max(self.peak_request_size, amount);
                } else {
                    assert!(p.is_null());
                }

                unsafe {
                    assert_eq!((*self.heap.0).diagnostics.allocated, self.allocated);
                    assert_eq!(
                        (*self.heap.0).diagnostics.peak_allocated,
                        self.peak_allocated
                    );
                    assert_eq!(
                        (*self.heap.0).diagnostics.peak_request_size,
                        self.peak_request_size
                    );
                }
                self.heap.match_fragments(reference);
                assert!(self.heap.do_invariants_hold());
                p
            }
            fn dealloc(&mut self, p: *mut u8, reference: &[(bool, usize)]) {
                if !p.is_null() {
                    // Overwrite some to ensure that the allocator does not make implicit assumptions
                    // about the memory use
                    {
                        let block_slice =
                            unsafe { slice::from_raw_parts_mut(p as *mut u8, O1HEAP_ALIGNMENT) };
                        let mut rng = rand::thread_rng();
                        block_slice.try_fill(&mut rng).unwrap();
                    }
                    let frag: &'h Fragment =
                        unsafe { Fragment::from_allocated_memory(p as *const u8) };
                    assert!(frag.header.used);
                    assert!(self.allocated >= frag.header.size);
                    self.allocated -= frag.header.size;
                }
                self.heap.free(p);

                unsafe {
                    assert_eq!((*self.heap.0).diagnostics.allocated, self.allocated);
                    assert_eq!(
                        (*self.heap.0).diagnostics.peak_allocated,
                        self.peak_allocated
                    );
                    assert_eq!(
                        (*self.heap.0).diagnostics.peak_request_size,
                        self.peak_request_size
                    );
                }
                self.heap.match_fragments(reference);
                assert!(self.heap.do_invariants_hold());
            }
        }

        const X: bool = true; // used
        const O: bool = false; // free
        let mut checker = AllocChecker::new(&mut heap);

        let a = checker.alloc(32, &[(X, 64), (O, 4032)]);
        let b = checker.alloc(32, &[(X, 64), (X, 64), (O, 3968)]);
        let c = checker.alloc(32, &[(X, 64), (X, 64), (X, 64), (O, 3904)]);
        let d = checker.alloc(32, &[(X, 64), (X, 64), (X, 64), (X, 64), (O, 3840)]);
        let e = checker.alloc(
            1024,
            &[(X, 64), (X, 64), (X, 64), (X, 64), (X, 2048), (O, 1792)],
        );
        let f = checker.alloc(
            512,
            &[
                (X, 64),
                (X, 64),
                (X, 64),
                (X, 64),
                (X, 2048),
                (X, 1024),
                (O, 768),
            ],
        );
        checker.dealloc(
            b,
            &[
                (X, 64),
                (O, 64),
                (X, 64),
                (X, 64),
                (X, 2048),
                (X, 1024),
                (O, 768),
            ],
        );
        checker.dealloc(
            a,
            &[(O, 128), (X, 64), (X, 64), (X, 2048), (X, 1024), (O, 768)],
        );
        checker.dealloc(c, &[(O, 192), (X, 64), (X, 2048), (X, 1024), (O, 768)]);
        checker.dealloc(e, &[(O, 192), (X, 64), (O, 2048), (X, 1024), (O, 768)]);
        // The last block will be taken because it is a better fit
        let g = checker.alloc(
            400,
            &[(O, 192), (X, 64), (O, 2048), (X, 1024), (X, 512), (O, 256)],
        );
        checker.dealloc(f, &[(O, 192), (X, 64), (O, 3072), (X, 512), (O, 256)]);
        checker.dealloc(d, &[(O, 3328), (X, 512), (O, 256)]);
        let h = checker.alloc(200, &[(O, 3328), (X, 512), (X, 256)]);
        let i = checker.alloc(32, &[(X, 64), (O, 3264), (X, 512), (X, 256)]);
        checker.dealloc(g, &[(X, 64), (O, 3776), (X, 256)]);
        checker.dealloc(h, &[(X, 64), (O, 4032)]);
        checker.dealloc(i, &[(O, 4096)]);

        assert_eq!((*heap.0).diagnostics.capacity, 4096);
        assert_eq!((*heap.0).diagnostics.allocated, 0);
        assert_eq!((*heap.0).diagnostics.peak_allocated, 3328);
        assert_eq!((*heap.0).diagnostics.peak_request_size, 1024);
        assert_eq!((*heap.0).diagnostics.oom_count, 0);
        assert!(heap.do_invariants_hold());
    }
}

#[test]
fn general_random_a() {
    let arena_size = 300 * MEBIBYTE;
    let mut arena = Arena::zeroed(arena_size);
    // Fill the whole arena with random bytes
    arena.as_mut_slice().try_fill(&mut thread_rng()).unwrap();

    let mut heap = unsafe { init(arena.as_mut_ptr(), arena_size).unwrap() };

    struct RandomChecker<'h> {
        heap: &'h mut TestHeap,
        arena_size: usize,
        pointers: Vec<*mut u8>,
        allocated: usize,
        peak_allocated: usize,
        peak_request_size: usize,
        oom_count: u64,
        rng: ThreadRng,
    }
    impl<'h> RandomChecker<'h> {
        pub fn new(heap: &'h mut TestHeap, arena_size: usize) -> Self {
            RandomChecker {
                heap,
                arena_size,
                pointers: Vec::new(),
                allocated: 0,
                peak_allocated: 0,
                peak_request_size: 0,
                oom_count: 0,
                rng: thread_rng(),
            }
        }

        pub fn allocate(&mut self) {
            assert!(self.heap.do_invariants_hold());
            let size_range = 0..=(self.arena_size / 1000);
            let amount = self.rng.gen_range(size_range);
            let ptr = self.heap.allocate(amount);
            if !ptr.is_null() {
                // Overwrite all to ensure that the allocator does not make implicit assumptions
                // about the memory use
                {
                    let allocated_slice = unsafe { slice::from_raw_parts_mut(ptr, amount) };
                    let mut rng = rand::thread_rng();
                    allocated_slice.try_fill(&mut rng).unwrap();
                }
                self.pointers.push(ptr);
                let frag: &'h Fragment = unsafe { Fragment::from_allocated_memory(ptr) };
                self.allocated += frag.header.size;
                self.peak_allocated = max(self.peak_allocated, self.allocated);
            } else if amount > 0 {
                self.oom_count += 1;
            }
            self.peak_request_size = max(self.peak_request_size, amount);
            assert!(self.heap.do_invariants_hold());
        }
        pub fn deallocate(&mut self) {
            assert!(self.heap.do_invariants_hold());
            if !self.pointers.is_empty() {
                // Choose a random pointer to free
                let index = self.rng.gen_range(0..self.pointers.len());
                let ptr = self.pointers.swap_remove(index);
                if !ptr.is_null() {
                    unsafe {
                        let fragment: &'h Fragment = Fragment::from_allocated_memory(ptr);
                        fragment.validate();
                        assert!(self.allocated >= fragment.header.size);
                        self.allocated -= fragment.header.size;
                    }
                }
                self.heap.free(ptr);
            }
            assert!(self.heap.do_invariants_hold());
        }
    }

    let mut checker = RandomChecker::new(&mut heap, arena_size);
    // The memory use is growing slowly from zero.
    // We stop the test when it's been running near the max heap utilization for long enough.
    while unsafe { (*checker.heap.0).diagnostics.oom_count < 1000 } {
        for _ in 0..100 {
            checker.allocate();
        }
        for _ in 0..50 {
            checker.deallocate();
        }
        unsafe {
            assert_eq!((*checker.heap.0).diagnostics.allocated, checker.allocated);
            assert_eq!(
                (*checker.heap.0).diagnostics.peak_allocated,
                checker.peak_allocated
            );
            assert_eq!(
                (*checker.heap.0).diagnostics.peak_request_size,
                checker.peak_request_size
            );
            assert_eq!((*checker.heap.0).diagnostics.oom_count, checker.oom_count);
        }
        assert!(checker.heap.do_invariants_hold());
    }
}

unsafe fn init(base: *mut u8, size: usize) -> Option<TestHeap> {
    fill_random(base, size);
    let heap = o1heap_init(base as *mut (), size);

    if !heap.is_null() {
        assert_eq!(heap as usize % O1HEAP_ALIGNMENT, 0);
        TestHeap::new(heap).validate();
        assert!((*heap).nonempty_bin_mask > 0);
        assert_eq!(
            (*heap).nonempty_bin_mask & ((*heap).nonempty_bin_mask - 1),
            0
        );
        for (i, bin_ptr) in (*heap).bins.iter().cloned().enumerate() {
            let min = FRAGMENT_SIZE_MIN << i;
            let max = (FRAGMENT_SIZE_MIN << i).saturating_mul(2).saturating_sub(1);
            if ((*heap).nonempty_bin_mask & (1 << i)) == 0 {
                assert!(bin_ptr.is_null());
            } else {
                assert!(!bin_ptr.is_null());
                assert!((*bin_ptr).header.size >= min);
                assert!((*bin_ptr).header.size <= max);
            }
        }

        assert!((*heap).diagnostics.capacity < size);
        assert!((*heap).diagnostics.capacity <= FRAGMENT_SIZE_MAX);
        assert!((*heap).diagnostics.capacity >= FRAGMENT_SIZE_MIN);
        assert_eq!((*heap).diagnostics.allocated, 0);
        assert_eq!((*heap).diagnostics.oom_count, 0);
        assert_eq!((*heap).diagnostics.peak_allocated, 0);
        assert_eq!((*heap).diagnostics.peak_request_size, 0);

        let root_fragment: *mut Fragment =
            (*heap).bins[usize::from(log_2_floor((*heap).nonempty_bin_mask))];
        assert!(!root_fragment.is_null());
        assert!((*root_fragment).next_free.is_null());
        assert!((*root_fragment).prev_free.is_null());
        assert!(!(*root_fragment).header.used);
        assert_eq!((*root_fragment).header.size, (*heap).diagnostics.capacity);
        assert!((*root_fragment).header.next.is_null());
        assert!((*root_fragment).header.prev.is_null());

        Some(TestHeap::new(heap))
    } else {
        None
    }
}

unsafe fn fill_random(base: *mut u8, size: usize) {
    let fill_bytes = cmp::min(MEBIBYTE, size);
    let slice = slice::from_raw_parts_mut(base, fill_bytes);
    let mut rng = rand::thread_rng();
    slice.try_fill(&mut rng).unwrap();
}

/// Heap extension functions used for testing
struct TestHeap(*mut O1HeapInstance);

impl TestHeap {
    unsafe fn new(instance: *mut O1HeapInstance) -> Self {
        assert!(!instance.is_null());
        TestHeap(instance)
    }

    fn allocate(&mut self, amount: usize) -> *mut u8 {
        self.validate();
        let ptr = unsafe { o1heap_allocate(self.0, amount) };
        if !ptr.is_null() {
            unsafe { Fragment::from_allocated_memory(ptr as *const u8).validate() };
        }
        self.validate();
        ptr
    }

    fn free(&mut self, pointer: *mut u8) {
        self.validate();
        unsafe { o1heap_free(self.0, pointer) };
        self.validate();
    }

    fn do_invariants_hold(&self) -> bool {
        unsafe { o1heap_do_invariants_hold(self.0) }
    }

    fn diagnostics(&self) -> O1HeapDiagnostics {
        self.validate();
        let diagnostics = unsafe { o1heap_get_diagnostics(self.0) };
        self.validate();
        diagnostics
    }

    fn first_fragment(&self) -> *const Fragment {
        let mut ptr = self.0 as usize + size_of::<O1HeapInstance>();
        while ptr % O1HEAP_ALIGNMENT != 0 {
            ptr += 1;
        }
        let frag = ptr as *const Fragment;
        unsafe {
            assert!((*frag).header.size >= FRAGMENT_SIZE_MIN);
            assert!((*frag).header.size <= FRAGMENT_SIZE_MAX);
            assert!((*frag).header.size <= (*self.0).diagnostics.capacity);
            assert_eq!((*frag).header.size % FRAGMENT_SIZE_MIN, 0);
            assert!(
                (*frag).header.next.is_null()
                    || (*(*frag).header.next).header.prev as *const Fragment == frag
            );
            // The first fragment has no previous
            assert!((*frag).header.prev.is_null());
        }
        frag
    }

    /// Checks that the sequence of fragments in the heap matches an expected layout
    ///
    /// Each item in references is:
    /// * Allocated: true if the fragment is allocated
    /// * Size: Size of the fragment, including overhead (a value of 0 matches any size)
    fn match_fragments(&self, reference: &[(bool, usize)]) {
        self.validate();
        unsafe {
            let mut frag = self.first_fragment();
            for &(used, size) in reference {
                assert!(!frag.is_null());
                assert_eq!((*frag).header.used, used);
                if size != 0 {
                    assert_eq!((*frag).header.size, size);
                }
                assert!(
                    (*frag).header.next.is_null()
                        || (*(*frag).header.next).header.prev as *const Fragment == frag
                );

                frag = (*frag).header.next;
            }
        }
    }

    fn validate(&self) {
        self.validate_core();
        self.validate_fragment_chain();
        self.validate_segregated_free_lists();
    }

    fn validate_core(&self) {
        unsafe {
            assert!((*self.0).diagnostics.capacity >= FRAGMENT_SIZE_MIN);
            assert!((*self.0).diagnostics.capacity <= FRAGMENT_SIZE_MAX);
            assert_eq!((*self.0).diagnostics.capacity % FRAGMENT_SIZE_MIN, 0);

            assert!((*self.0).diagnostics.allocated <= (*self.0).diagnostics.capacity);
            assert_eq!((*self.0).diagnostics.allocated % FRAGMENT_SIZE_MIN, 0);

            assert!((*self.0).diagnostics.peak_allocated <= (*self.0).diagnostics.capacity);
            assert!((*self.0).diagnostics.peak_allocated >= (*self.0).diagnostics.allocated);
            assert_eq!((*self.0).diagnostics.peak_allocated % FRAGMENT_SIZE_MIN, 0);

            assert!(
                (((*self.0).diagnostics.peak_request_size <= (*self.0).diagnostics.capacity)
                    || ((*self.0).diagnostics.oom_count > 0))
            );
            assert!(
                (((*self.0)
                    .diagnostics
                    .peak_request_size
                    .saturating_add(O1HEAP_ALIGNMENT)
                    <= (*self.0).diagnostics.peak_allocated)
                    || ((*self.0).diagnostics.peak_request_size == 0)
                    || ((*self.0).diagnostics.oom_count > 0))
            );
        }
    }
    fn validate_fragment_chain(&self) {
        unsafe {
            let (_, mut pending_bins) =
                (*self.0)
                    .bins
                    .iter()
                    .fold((0usize, 0usize), |(i, pending_bins), fragment_ptr| {
                        if !fragment_ptr.is_null() {
                            (i + 1, pending_bins | (1usize << i))
                        } else {
                            (i + 1, pending_bins)
                        }
                    });
            assert_eq!(pending_bins, (*self.0).nonempty_bin_mask);

            let mut total_size = 0usize;
            let mut total_allocated = 0usize;
            let mut frag = self.first_fragment();
            while {
                frag.as_ref().unwrap().validate();
                let bin_index = usize::from(frag.as_ref().unwrap().bin_index());
                assert!((*frag).header.size <= (*self.0).diagnostics.capacity);
                // Update and check the totals early
                total_size += (*frag).header.size;
                assert!(total_size <= FRAGMENT_SIZE_MAX);
                assert!(total_size <= (*self.0).diagnostics.capacity);
                assert_eq!(total_size % FRAGMENT_SIZE_MIN, 0);
                if (*frag).header.used {
                    total_allocated += (*frag).header.size;
                    assert!(total_allocated <= total_size);
                    assert_eq!(total_allocated % FRAGMENT_SIZE_MIN, 0);
                    // Ensure no bin links to a used fragment
                    assert_ne!((*self.0).bins[bin_index] as *const Fragment, frag);
                } else {
                    let mask = 1usize << bin_index;
                    assert_ne!((*self.0).nonempty_bin_mask & mask, 0);
                    if (*self.0).bins[bin_index] as *const Fragment == frag {
                        assert_ne!(pending_bins & mask, 0);
                        pending_bins &= !mask;
                    }
                }

                frag = (*frag).header.next;
                !frag.is_null()
            } {}

            // Ensure there were no hanging bin pointers
            assert_eq!(pending_bins, 0);

            // Validate the totals
            assert_eq!(total_size, (*self.0).diagnostics.capacity);
            assert_eq!(total_allocated, (*self.0).diagnostics.allocated);
        }
    }
    fn validate_segregated_free_lists(&self) {
        unsafe {
            let mut total_free = 0usize;
            for (i, frag) in (*self.0).bins.iter().enumerate() {
                let mut frag = *frag;
                let mask = 1usize << i;
                let min = FRAGMENT_SIZE_MIN << i;
                let max = (FRAGMENT_SIZE_MIN << i).wrapping_mul(2).wrapping_sub(1);

                if !frag.is_null() {
                    assert_ne!((*self.0).nonempty_bin_mask & mask, 0);
                    assert!(!(*frag).header.used);
                    assert!((*frag).prev_free.is_null());

                    while {
                        assert!((*frag).header.size >= min);
                        assert!((*frag).header.size <= max);
                        total_free += (*frag).header.size;

                        if !(*frag).next_free.is_null() {
                            assert_eq!((*(*frag).next_free).prev_free, frag);
                            assert!(!(*(*frag).next_free).header.used);
                        }

                        if !(*frag).prev_free.is_null() {
                            assert_eq!((*(*frag).prev_free).next_free, frag);
                            assert!(!(*(*frag).prev_free).header.used);
                        }

                        frag = (*frag).next_free;

                        !frag.is_null()
                    } {}
                } else {
                    assert_eq!((*self.0).nonempty_bin_mask & mask, 0);
                }
            }
            assert_eq!(
                (*self.0).diagnostics.capacity - (*self.0).diagnostics.allocated,
                total_free
            );
        }
    }
}

/// Fragment extension functions used for testing
trait FragmentExt {
    unsafe fn from_allocated_memory<'m>(memory: *const u8) -> &'m Self;
    fn bin_index(&self) -> u8;
    unsafe fn validate(&self);
}

impl FragmentExt for Fragment {
    unsafe fn from_allocated_memory<'m>(memory: *const u8) -> &'m Self {
        assert!(!memory.is_null());
        assert!(memory as usize > O1HEAP_ALIGNMENT);
        assert_eq!(memory as usize % O1HEAP_ALIGNMENT, 0);
        &*((memory as usize - O1HEAP_ALIGNMENT) as *const Self)
    }

    fn bin_index(&self) -> u8 {
        assert_eq!(self.header.size % FRAGMENT_SIZE_MIN, 0);
        assert!(self.header.size >= FRAGMENT_SIZE_MIN);
        (self.header.size as f64 / FRAGMENT_SIZE_MIN as f64)
            .log2()
            .floor() as u8
    }

    unsafe fn validate(&self) {
        let address = self as *const Fragment as usize;
        assert_eq!(address % size_of::<*const ()>(), 0);
        // Size correctness
        assert!(self.header.size >= FRAGMENT_SIZE_MIN);
        assert!(self.header.size <= FRAGMENT_SIZE_MAX);
        assert_eq!(self.header.size % FRAGMENT_SIZE_MIN, 0);
        // Heap fragment interlinking. Free blocks cannot neighbor each other because they are
        // supposed to be merged.
        if !self.header.next.is_null() {
            assert!(self.header.used || (*self.header.next).header.used);
            let next_block_address = self.header.next as usize;
            assert_eq!(next_block_address % size_of::<*const ()>(), 0);
            assert_eq!((*self.header.next).header.prev as usize, address);
            assert!(next_block_address > address);
            assert_eq!((next_block_address - address) % FRAGMENT_SIZE_MIN, 0);
        }
        if !self.header.prev.is_null() {
            assert!(self.header.used || (*self.header.prev).header.used);
            let prev_block_address = self.header.prev as usize;
            assert_eq!(prev_block_address % size_of::<*const ()>(), 0);
            assert_eq!((*self.header.prev).header.next as usize, address);
            assert!(address > prev_block_address);
            assert_eq!((address - prev_block_address) % FRAGMENT_SIZE_MIN, 0);
        }
        // Segregated free list interlinking
        if !self.header.used {
            if !self.next_free.is_null() {
                assert_eq!((*self.next_free).prev_free as usize, address);
                assert!(!(*self.next_free).header.used);
            }
            if !self.prev_free.is_null() {
                assert_eq!((*self.prev_free).next_free as usize, address);
                assert!(!(*self.prev_free).header.used);
            }
        }
    }
}
