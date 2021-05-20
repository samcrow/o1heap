#![cfg_attr(not(test), no_std)]

mod o1heap;

use core::alloc::Layout;
use core::ptr::NonNull;

use crate::o1heap::{
    o1heap_allocate, o1heap_free, o1heap_get_diagnostics, O1HeapInstance, O1HEAP_ALIGNMENT,
};

/// A memory allocator with constant-time allocation and deallocation
pub struct Heap {
    /// Pointer to the beginning of the heap, where the heap header is stored
    instance: Option<NonNull<O1HeapInstance>>,
}

impl Heap {
    /// The minimum alignment required for the address of the beginning of the heap
    pub const ALIGNMENT: usize = o1heap::O1HEAP_ALIGNMENT;

    /// Creates an empty heap
    ///
    /// No other functions on the heap may be called until [`init`](#method.init) is called.
    pub const fn empty() -> Self {
        Heap { instance: None }
    }

    /// Attempts to initialize the heap
    ///
    /// * `start_addr`: The address of the beginning of the heap. This must be a multiple of
    /// [`ALIGNMENT`](#associatedconstant.ALIGNMENT).
    /// * `size`: The number of bytes to use for the heap
    ///
    /// This function returns an error if `size` is too small or `start_addr` is not aligned.
    ///
    /// # Safety
    ///
    /// This function must be called at most one time after creating an empty heap, before calling
    /// any other heap functions.
    ///
    /// The memory in the range `[start_addr, start_addr + size)` must not be used for anything
    /// else during the lifetime of this heap.
    ///
    pub unsafe fn init(&mut self, start_addr: usize, size: usize) -> Result<(), InitError> {
        let instance = o1heap::o1heap_init(start_addr as *mut (), size);
        if instance.is_null() {
            Err(InitError)
        } else {
            self.instance = Some(NonNull::new(instance).unwrap());
            Ok(())
        }
    }

    /// Attempts to allocate a block of memory that satisfies the provided layout
    ///
    /// # Panics
    ///
    /// This function panics if this heap has not been successfully initialized.
    pub fn allocate(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let instance = self.instance.expect("Heap not initialized");
        if layout.size() == 0 {
            // Return a non-null, aligned, non-dereferenceable pointer
            Some(NonNull::new(layout.align() as *mut u8).unwrap())
        } else if layout.align() <= O1HEAP_ALIGNMENT {
            let ptr = unsafe { o1heap_allocate(instance.as_ptr(), layout.size()) };
            NonNull::new(ptr)
        } else {
            // Can't satisfy alignment
            None
        }
    }

    /// Deallocates a block of memory
    ///
    /// # Safety
    ///
    /// `ptr` must be a pointer that was previously returned from a call to
    /// [`allocate`](#method.allocate) with the same layout given as the `layout` parameter.
    ///
    /// # Panics
    ///
    /// This function panics if this heap has not been successfully initialized.
    pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, layout: Layout) {
        let instance = self.instance.expect("Heap not initialized");
        if layout.size() != 0 {
            o1heap_free(instance.as_ptr(), ptr.as_ptr());
        }
    }

    /// Returns diagnostics about the status of the heap
    ///
    /// # Panics
    ///
    /// This function panics if this heap has not been successfully initialized.
    pub fn diagnostics(&self) -> Diagnostics {
        Diagnostics {
            inner: unsafe {
                o1heap_get_diagnostics(self.instance.expect("Heap not initialized").as_ptr())
            },
        }
    }

    /// Performs basic tests on the heap structure and returns true if the heap appears to be
    /// intact
    ///
    /// # Panics
    ///
    /// This function panics if this heap has not been successfully initialized.
    pub fn do_invariants_hold(&self) -> bool {
        let instance = self.instance.expect("Heap not initialized");
        unsafe { o1heap::o1heap_do_invariants_hold(instance.as_ptr()) }
    }
}

/// A heap may be sent between threads
unsafe impl Send for Heap {}

/// Information about the status of the heap
pub struct Diagnostics {
    inner: o1heap::O1HeapDiagnostics,
}

impl Diagnostics {
    /// Returns the total amount of memory available for allocation
    ///
    /// The maximum allocation size is (capacity - O1HEAP_ALIGNMENT).
    /// This parameter does not include the overhead used up by the heap and and arena alignment.
    ///
    /// This parameter is constant.
    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    /// Returns the amount of memory currently allocated
    ///
    /// This value includes the per-fragment overhead and size alignment.
    /// For example, if the application requested a fragment of size 1 byte, the value reported here
    /// may be 32 bytes.
    pub fn allocated(&self) -> usize {
        self.inner.allocated
    }
    /// Returns the maximum value of `allocated` seen since initialization
    ///
    /// This parameter is never decreased.
    pub fn peak_allocated(&self) -> usize {
        self.inner.peak_allocated
    }

    /// Returns the largest requested allocation size
    ///
    /// The return value is the largest amount of memory that the allocator has attempted to
    /// allocate (perhaps unsuccessfully) since initialization (not including the rounding and the
    /// allocator's own per-fragment overhead, so the total is larger).
    ///
    /// This parameter is never decreased. The initial value is zero.
    pub fn peak_request_size(&self) -> usize {
        self.inner.peak_request_size
    }

    /// Returns the number of times an allocation request could not be completed
    pub fn oom_count(&self) -> u64 {
        self.inner.oom_count
    }
}

impl core::fmt::Debug for Diagnostics {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Diagnostics")
            .field("capacity", &self.inner.capacity)
            .field("allocated", &self.inner.allocated)
            .field("peak_allocated", &self.inner.peak_allocated)
            .field("peak_request_size", &self.inner.peak_request_size)
            .field("oom_count", &self.inner.oom_count)
            .finish()
    }
}

/// An error that occurred when initializing the heap
#[derive(Debug)]
pub struct InitError;
