use core::ffi::CStr;
use core::marker::PhantomData;

use encapfn::abi::rv32i_c::Rv32iCABI;
use encapfn::branding::EFID;
use encapfn::rt::EncapfnRt;
use encapfn::types::{AllocScope, AllocTracker};
use encapfn::EFError;

#[derive(Clone, Debug)]
pub struct TockRv32iCRtAllocTrackerCons<'a> {
    pred: &'a TockRv32iCRtAllocTracker<'a>,
    allocation: (*mut u8, usize),
    mutable: bool,
}

#[derive(Clone, Debug)]
pub struct TockRv32iCRtAllocTracker<'a>(Option<TockRv32iCRtAllocTrackerCons<'a>>);

impl TockRv32iCRtAllocTracker<'_> {
    fn is_valid_int(&self, ptr: *mut u8, len: usize, mutable: bool) -> bool {
        let mut cur = self;

        loop {
            if let Some(ref alloc) = cur.0 {
                let (aptr, alen) = alloc.allocation;

                // Make sure that:
                // - start address lies within region,
                // - end address lies within region,
                // - _if_ we require mutability, check that the allocation is
                //   mutable too.
                let matches = (ptr as usize) >= (aptr as usize)
                    && ((ptr as usize)
                        .checked_add(len)
                        .map(|end| end <= (aptr as usize) + alen)
                        .unwrap_or(false))
                    && (!mutable || alloc.mutable);

                if matches {
                    return true;
                } else {
                    cur = alloc.pred;
                }
            } else {
                return false;
            }
        }
    }
}

unsafe impl AllocTracker for TockRv32iCRtAllocTracker<'_> {
    fn is_valid(&self, ptr: *const u8, len: usize) -> bool {
        self.is_valid_int(ptr as *mut u8, len, false)
    }

    fn is_valid_mut(&self, ptr: *mut u8, len: usize) -> bool {
        self.is_valid_int(ptr, len, true)
    }
}

pub struct TockRv32iCRt<ID: EFID> {
    // TODO
    _id: PhantomData<ID>,
}

impl<ID: EFID> TockRv32iCRt<ID> {
    pub unsafe fn new(_branding: ID) -> Self {
        TockRv32iCRt { _id: PhantomData }
    }
}

unsafe impl<ID: EFID> EncapfnRt for TockRv32iCRt<ID> {
    type ID = ID;
    type AllocTracker<'a> = TockRv32iCRtAllocTracker<'a>;
    type ABI = Rv32iCABI;

    // We don't have any symbol table state, as the Tock EF binary
    // already contains a symbol table that we can use.
    type SymbolTableState<const SYMTAB_SIZE: usize, const FIXED_OFFSET_SYMTAB_SIZE: usize> = ();

    fn resolve_symbols<const SYMTAB_SIZE: usize, const FIXED_OFFSET_SYMTAB_SIZE: usize>(
        &self,
        _symbol_table: &'static [&'static CStr; SYMTAB_SIZE],
        _fixed_offset_symbol_table: &'static [Option<&'static CStr>; FIXED_OFFSET_SYMTAB_SIZE],
    ) -> Option<Self::SymbolTableState<SYMTAB_SIZE, FIXED_OFFSET_SYMTAB_SIZE>> {
        // TODO: check whether the binary's symbol table is large enough to
        // contain all symbols that could possbily be referenced by the fixed
        // offset symbol table (i.e., binary symtab size >=
        // FIXED_OFFSET_SYMTAB_SIZE).
        Some(())
    }

    fn lookup_symbol<const SYMTAB_SIZE: usize, const FIXED_OFFSET_SYMTAB_SIZE: usize>(
        &self,
        _index: usize,
        _symtabstate: &Self::SymbolTableState<SYMTAB_SIZE, FIXED_OFFSET_SYMTAB_SIZE>,
    ) -> Option<*const ()> {
        // TODO: actually look up symbol
        None
    }

    // We provide only the required implementations and rely on default
    // implementations for all "convenience" allocation methods. These are as
    // efficient as it gets in our case anyways.
    fn allocate_stacked_untracked_mut<F, R>(
        &self,
        _requested_layout: core::alloc::Layout,
        _fun: F,
    ) -> Result<R, EFError>
    where
        F: FnOnce(*mut u8) -> R,
    {
        unimplemented!()
    }

    fn allocate_stacked_mut<F, R>(
        &self,
        layout: core::alloc::Layout,
        alloc_scope: &mut AllocScope<'_, Self::AllocTracker<'_>, ID>,
        fun: F,
    ) -> Result<R, EFError>
    where
        F: for<'b> FnOnce(*mut u8, &'b mut AllocScope<'_, Self::AllocTracker<'_>, Self::ID>) -> R,
    {
        self.allocate_stacked_untracked_mut(layout, move |ptr| {
            // Create a new AllocScope instance that wraps a new allocation
            // tracker `Cons` list element that points to this allocation, and
            // its predecessors:
            let mut inner_alloc_scope = unsafe {
                AllocScope::new(TockRv32iCRtAllocTracker(Some(
                    TockRv32iCRtAllocTrackerCons {
                        pred: alloc_scope.tracker(),
                        allocation: (ptr, layout.size()),
                        mutable: true,
                    },
                )))
            };

            // Hand a temporary mutable reference to this new scope to the
            // closure.
            //
            // We thus not only allocate, but also track allocations themselves
            // on the stack, and there is nothing to clean up! The new
            // `inner_alloc_scope` will simply go out of scope at the end of
            // this closure.
            fun(ptr, &mut inner_alloc_scope)
        })
    }
}
