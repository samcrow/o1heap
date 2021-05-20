#![no_std]

extern crate cortex_m;
extern crate o1heap;

use core::alloc::{GlobalAlloc, Layout};
use core::cell::RefCell;
use core::ptr;
use core::ptr::NonNull;
use cortex_m::interrupt::{self, Mutex};
use o1heap::Heap;
pub use o1heap::{Diagnostics, InitError};

/// A global memory allocator that uses interrupt-disabling critical sections to control access
pub struct CortexMHeap {
    heap: Mutex<RefCell<Heap>>,
}

impl CortexMHeap {
    /// The minimum alignment required for the address of the beginning of the heap
    pub const ALIGNMENT: usize = Heap::ALIGNMENT;

    /// Creates an empty heap
    ///
    /// No other functions on the heap may be called until [`init`](#method.init) is called.
    pub const fn empty() -> Self {
        CortexMHeap {
            heap: Mutex::new(RefCell::new(Heap::empty())),
        }
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
        interrupt::free(|cs| self.heap.borrow(cs).borrow_mut().init(start_addr, size))
    }

    /// Returns diagnostics about the status of the heap
    ///
    /// # Panics
    ///
    /// This function panics if this heap has not been successfully initialized.
    pub fn diagnostics(&self) -> Diagnostics {
        interrupt::free(|cs| self.heap.borrow(cs).borrow().diagnostics())
    }

    /// Performs basic tests on the heap structure and returns true if the heap appears to be
    /// intact
    ///
    /// # Panics
    ///
    /// This function panics if this heap has not been successfully initialized.
    pub fn do_invariants_hold(&self) -> bool {
        interrupt::free(|cs| self.heap.borrow(cs).borrow().do_invariants_hold())
    }
}

unsafe impl GlobalAlloc for CortexMHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        interrupt::free(|cs| self.heap.borrow(cs).borrow_mut().allocate(layout))
            .map(NonNull::as_ptr)
            .unwrap_or_else(ptr::null_mut)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(non_null_ptr) = NonNull::new(ptr) {
            interrupt::free(|cs| {
                self.heap
                    .borrow(cs)
                    .borrow_mut()
                    .deallocate(non_null_ptr, layout)
            });
        }
    }
}
