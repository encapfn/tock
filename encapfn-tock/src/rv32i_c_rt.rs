use core::cell::UnsafeCell;
use core::ffi::CStr;
use core::marker::PhantomData;

use kernel::platform::mpu::{self, MPU};

use encapfn::abi::rv32i_c::Rv32iCABI;
use encapfn::branding::EFID;
use encapfn::rt::EncapfnRt;
use encapfn::types::{AccessScope, AllocScope, AllocTracker};
use encapfn::EFError;

use crate::binary::EncapfnBinary;

pub const ENCAPFN_HEADER_MAGIC_OFFSET: usize = 0;
pub const ENCAPFN_HEADER_RTHDR_PTR_OFFSET: usize = 4;
pub const ENCAPFN_HEADER_INIT_PTR_OFFSET: usize = 8;
pub const ENCAPFN_HEADER_FNTAB_PTR_OFFSET: usize = 12;
pub const ENCAPFN_HEADER_FNTAB_LEN_OFFSET: usize = 16;
pub const ENCAPFN_HEADER_LEN: usize = 20;
pub const ENCAPFN_HEADER_MAGIC: u32 = 0x454E4350;

#[derive(Clone, Debug)]
pub struct TockRv32iCRtAllocTracker<'a> {
    ram_region_start: *mut (),
    ram_region_length: usize,
    flash_region_start: *mut (),
    flash_region_length: usize,
    _lt: PhantomData<&'a ()>,
}

unsafe impl AllocTracker for TockRv32iCRtAllocTracker<'_> {
    fn is_valid(&self, ptr: *const u8, len: usize) -> bool {
        let is_valid_flash = (ptr as usize) >= (self.flash_region_start as usize)
            && ((ptr as usize)
                .checked_add(len)
                .map(|end| end <= (self.flash_region_start as usize) + self.flash_region_length)
                .unwrap_or(false));

        is_valid_flash || self.is_valid_mut(ptr as *mut u8, len)
    }

    fn is_valid_mut(&self, ptr: *mut u8, len: usize) -> bool {
        (ptr as usize) >= (self.ram_region_start as usize)
            && ((ptr as usize)
                .checked_add(len)
                .map(|end| end <= (self.ram_region_start as usize) + self.ram_region_length)
                .unwrap_or(false))
    }
}

#[repr(C)]
pub struct TockRv32iCRtAsmState {
    // Foreign stack pointer, read by the protection-domain switch assembly
    // and used as a base to copy stacked arguments & continue execution from:
    foreign_stack_ptr: UnsafeCell<*mut ()>,

    // Foreign stack bottom (inclusive). Last usable stack address:
    foreign_stack_bottom: *mut (),

    // TODO: doc
    ram_region_start: *mut (),
    ram_region_length: usize,
}

#[repr(C)]
pub struct TockRv32iCRt<ID: EFID, M: MPU + 'static> {
    // This struct is used both in the protection-domain switch assembly,
    // and in regular Rust code. However, we want to avoid hard-coding offsets
    // into this struct in assembly, and instead use ::core::ptr::offset_of!
    // to resolve offsets of relevant fields at compile. Unfortunately, that is
    // not possible, in general, for a generic type without knowing the generic
    // argument. Instead, we move all assembly-relevant state into a separate
    // struct `TockRv32iCRtAsmState`, which does not have generic parameters.
    // We ensure that this struct is placed at the very beginning of the
    // `TockRv32iCRt` type, for every possible combination of generic
    // parameters, through an assertion in its constructor.
    asm_state: TockRv32iCRtAsmState,

    binary: EncapfnBinary,
    rthdr_addr: *const (),
    init_addr: *const (),
    fntab_addr: *const (),
    fntab_length: usize,

    mpu: &'static M,
    mpu_config: M::MpuConfig,

    _id: PhantomData<ID>,
}

impl<ID: EFID, M: MPU + 'static> TockRv32iCRt<ID, M> {
    pub unsafe fn new(
        mpu: &'static M,
        binary: EncapfnBinary,
        ram_region_start: *mut (),
        ram_region_length: usize,
        _branding: ID,
    ) -> Result<
        (
            Self,
            AllocScope<'static, TockRv32iCRtAllocTracker<'static>, ID>,
            AccessScope<ID>,
        ),
        (),
    > {
        // See the TockRv32iCRt type definition for an explanation of this
        // const assertion. It is required to allow us to index into fields
        // of the nested `TockRv32iCRtAllocTracker` struct from within assembly.
        //
        // Unfortunately, we cannot make this into a const assertion, as
        // constants are instantiated outside of the `impl` block.
        let _: () = assert!(core::mem::offset_of!(Self, asm_state) == 0);

        // Each Encapsulated Functions binary must start with a header indicating
        // relevant data to the loader (Tock). Check that the binary can at
        // least fit the header, ensure that the magic bytes match, and perform
        // other sanity checks.
        //
        // Encapsulated Functions binary header layout:
        //
        // 0             2             4             6             8
        // +---------------------------+---------------------------+
        // | 0x454E4350 (ENCP) MAGIC   | Runtime Header Offset     |
        // +---------------------------+---------------------------+
        // | `init` Function Offset    | Function Table Offset     |
        // +---------------------------+---------------------------+
        // | Function Table Length (in pointers)
        // +---------------------------+
        //
        // We will try to load these sections into the provided RAM region, with
        // a layout as follows:
        //
        // +---------------------------+ <- `ram_region_start`
        // | Loader-initialized data   | -\
        // | - (optional) padding      |  |
        // | - .data                   |  |
        // | - .bss                    |  |
        // +---------------------------+  |
        // | Rust "remote memory"    | |  |
        // | stack allocator         | |  |
        // |                         v |  |- R/W permissions for foreign code
        // +---------------------------+  |
        // | Return trampoline stack   |  |
        // | frame                     |  |
        // +---------------------------+  |
        // | Encapsulated functions  | |  |
        // | service stack           | |  |
        // |                         v | -/
        // +---------------------------+ <- `ram_region_start` + `ram_region_len`
        //
        // The entire Encapsulated Functions binary will further be made
        // available with read-execute permissions.

        // Make sure we have at least enough data to parse the header:
        if binary.binary_length < ENCAPFN_HEADER_LEN {
            return Err(());
        }

        // We generally try to avoid retaining Rust slices to the containerized
        // service binary (to avoid unsoundness, in case this memory should
        // change). However, for parsing the header, we can create an ephemeral
        // slice given that we verified the length:
        let header_slice =
            core::slice::from_raw_parts(binary.binary_start as *const u8, ENCAPFN_HEADER_LEN);

        // We don't rely on the header to be well aligned here. Read fields
        // byte-by-byte. TODO: check that we meet or enforce alignment
        // constraints for other accesses too!
        #[inline(always)]
        fn extract_header_word(header_slice: &[u8], offset: usize) -> u32 {
            let word_slice = &header_slice[offset..offset + core::mem::size_of::<u32>()];
            u32::from_ne_bytes([word_slice[0], word_slice[1], word_slice[2], word_slice[3]])
        }

        // Read the header fields in native endianness. First, check the magic:
        if extract_header_word(header_slice, ENCAPFN_HEADER_MAGIC_OFFSET) != ENCAPFN_HEADER_MAGIC {
            return Err(());
        }

        // Extract the runtime header pointer and ensure that it is fully
        // contained in contained within the binary:
        let rthdr_offset =
            extract_header_word(header_slice, ENCAPFN_HEADER_RTHDR_PTR_OFFSET) as usize;
        if rthdr_offset > binary.binary_length - core::mem::size_of::<u32>() {
            return Err(());
        }
        let rthdr_addr = unsafe { binary.binary_start.byte_add(rthdr_offset) };

        // Extract the init function pointer pointer and ensure that it is fully
        // contained in contained within the binary:
        let init_offset =
            extract_header_word(header_slice, ENCAPFN_HEADER_INIT_PTR_OFFSET) as usize;
        if init_offset > binary.binary_length - core::mem::size_of::<u32>() {
            return Err(());
        }
        let init_addr = unsafe { binary.binary_start.byte_add(init_offset) };

        // Extract the function table pointer and ensure that it is fully
        // contained in contained within the binary:
        let fntab_offset =
            extract_header_word(header_slice, ENCAPFN_HEADER_FNTAB_PTR_OFFSET) as usize;
        let fntab_length =
            extract_header_word(header_slice, ENCAPFN_HEADER_FNTAB_LEN_OFFSET) as usize;
        if fntab_offset + (fntab_length * core::mem::size_of::<*const ()>())
            > binary.binary_length - core::mem::size_of::<u32>()
        {
            return Err(());
        }
        let fntab_addr = unsafe { binary.binary_start.byte_add(fntab_offset) };

        // Create an MPU configuration that sets up appropriate permissions for
        // the Encapsulated Functions binary:
        let mut mpu_config = mpu.new_config().ok_or(())?;

        mpu.allocate_region(
            binary.binary_start as *const u8,
            binary.binary_length,
            binary.binary_length,
            mpu::Permissions::ReadExecuteOnly,
            &mut mpu_config,
        )
        .unwrap();

        mpu.allocate_region(
            ram_region_start as *mut u8 as *const _,
            ram_region_length,
            ram_region_length,
            mpu::Permissions::ReadWriteOnly,
            &mut mpu_config,
        )
        .unwrap();

        // Construct an initial runtime instance. We don't yet know where our
        // `foreign_stack_top` should be placed -- that will depend on how much
        // static data `init` will place at the top of memory. We need to set
        // the stack pointer equal to some valid value though, and thus we --
        // for now -- set it to be the top of memory.
        let rt = TockRv32iCRt {
            asm_state: TockRv32iCRtAsmState {
                foreign_stack_ptr: UnsafeCell::new(ram_region_start),
                ram_region_start,
                ram_region_length,
            },

            binary,
            rthdr_addr,
            init_addr,
            fntab_addr,
            fntab_length,

            mpu,
            mpu_config,

            _id: PhantomData::<ID>,
        };

        unimplemented!()
        // Ok((
        //     encapfn,
        //     unsafe {
        //         AllocScope::new(EncapfnTockRv32iCRtAllocTracker {
        //             ram_region_start: ram_region_start,
        //             ram_region_len: ram_region_len,
        // 	    flash_region_start: flash_region_start,
        // 	    flash_region_end: flash_region_end,
        //         })
        //     },
        //     unsafe { AccessScope::new() },
        // ))
    }

    pub fn init(&self) -> Result<(), ()> {}

    #[naked]
    unsafe extern "C" fn generic_invoke() {
        core::arch::asm!(
            "
                // When entering this symbol, we are supposed to invoke a
                // foreign function in an isolated protection domain (switching
                // to user-mode and thus engaging the PMP).
                //
                // At this stage, the PMP has already been set up (through the
                // call to `execute`). This is an unsafe symbol, and callers
                // of this symbol must transitively guarantee that this function
                // is only invoked in the `execute` closure, and this is the
                // only function in that closure which may ever attempt to
                // modify the PMP state or its enforcement).
                //
                // This symbol is effectively indirectly aliased to a bunch of
                // extern \"C\" functions, which cause the Rust compiler / LLVM
                // to place the function arguments in the correct registers on
                // our Rust stack. We must thus avoid clobbering all such state
                // until we invoke the function.
                //
                // The RT::invoke `#[naked]` wrapper functions also load some
                // const-generic data and other information into a set of
                // well-defined registers; specifically:
                // - t0: &TockRv32iCRtAsmState
                // - t1: function pointer to execute
                // - t2: &mut TockRv32iCRtInvokeResInner
                // - t3: amount of bytes spilled on the stack, to copy
                // - t4: <this function symbol>
                //
                // As this symbol follows the C ABI / calling convention, and we
                // cannot rely on the foreign code to preserve saved registers,
                // we must do so here.
                //
                // Following that, we need to copy the stack-spilled arguments
                // onto the foreign stack and switch to user-mode.
                //
                // The trap handler may return to this code because either
                //
                // - we received an interrupt while executing foreign code.
                //   In this case, disable the interrupt, and resume execution
                //   of foreign code, being careful to not clobber any
                //   registers.
                //
                // - we received a system call (return instruction), or a trap.
                //   It is not possible for us to modify the foreign binary and
                //   insert a system call instruction with execute permissions.
                //   Thus, when memory protection is engaged, we rely on the
                //   the foreign code to attempt to execute a system call inst.,
                //   which then causes a fault at a well-known address. We then
                //   intepret this fault as an attempted system call too.
                //
                //   We define an analog springboard for callbacks, using the
                //   `unimp` mnemonic (to distinguish it from system calls when
                //   not engaging memory protection.
                //
                //   All other traps are faults and should require
                //   re-initialization of the Encapsulated Functions runtime.

                // First, save the current stack pointer in a temporary
                // register. We start copying foreign arguments from this point
                // onward in a bit.
                mv  t5, sp

                // Now, save all callee-saved registers, non-clobberable
                // reigsters (e.g., fp, gp), and other important state on the
                // stack. The stack layout is set up to be compatible with the
                // assumptions of the Tock rv32i trap handler. We also reserve
                // space for caller-saved registers of foreign code, for when we
                // need to disable interrupts with a Rust function (exposing a
                // C ABI). Doing this here, and writing beyond this space
                // prevents stack overflows later on.
                //
                // ```
                //  40*4(sp): <- original stack pointer
                //  39*4(sp):
                //  38*4(sp):
                // ^^^^^^^^^^ Other Interrupt-Saved Registers ^^^^^^^^^^^^^^^^^^
                //  37*4(sp): x4  / tp (we swap to Rust tp)
                //  36*4(sp): x3  / gp (we swap to Rust gp)
                //  35*4(sp): x2  / sp (not caller-saved, but we swap to Rust sp)
                // vvvvvvvvvv Foreign Caller-Saved Registers vvvvvvvvvvvvvvvvvvv
                //  34*4(sp): x31 / t6
                //  33*4(sp): x30 / t5
                //  32*4(sp): x29 / t4
                //  31*4(sp): x28 / t3
                //  30*4(sp): x17 / a7
                //  29*4(sp): x16 / a6
                //  28*4(sp): x15 / a5
                //  27*4(sp): x14 / a4
                //  26*4(sp): x13 / a3
                //  25*4(sp): x12 / a2
                //  24*4(sp): x11 / a1
                //  23*4(sp): x10 / a0
                //  22*4(sp): x7  / t2
                //  21*4(sp): x6  / t1
                //  20*4(sp): x5  / t0
                //  19*4(sp): x1  / ra
                // ^^^^^^^^^^ Foreign Caller-Saved Registers ^^^^^^^^^^^^^^^^^^^
                // vvvvvvvvvv Kernel Callee-Saved Registers vvvvvvvvvvvvvvvvvvvv
                //  18*4(sp): x27 / s11
                //  17*4(sp): x26 / s10
                //  16*4(sp): x25 / s9
                //  15*4(sp): x24 / s8
                //  14*4(sp): x23 / s7
                //  13*4(sp): x22 / s6
                //  12*4(sp): x21 / s5
                //  11*4(sp): x20 / s4
                //  10*4(sp): x19 / s3
                //   9*4(sp): x18 / s2
                //   8*4(sp): x9  / s1
                //   7*4(sp): x8  / s0 / fp
                //   6*4(sp): x4  / tp
                //   5*4(sp): x3  / gp
                // ^^^^^^^^^^ Kernel Callee-Saved Registers ^^^^^^^^^^^^^^^^^^^^
                // vvvvvvvvvv Kernel Caller-Saved Registers vvvvvvvvvvvvvvvvvvvv
                //   4*4(sp): &mut TockRv32iCRtInvokeResInner (x7 / t2)
                //   3*4(sp): &TockRv32iCRtAsmState (x5 / t0)
                //   2*4(sp): x1  / ra (not callee-saved, but we clobber)
                // vvvvvvvvvv Trap Handler Context vvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                //   1*4(sp): custom trap handler address
                //   0*4(sp): scratch space, s1 written to by trap handler
                //            <- new stack pointer
                // ```
                //
                // We don't need to save the stack pointer, as it will be
                // preserved in `mscratch` below.

                addi sp, sp, -40*4  // Move the stack pointer down to make room.

                // Save all registers according to the above memory map:
                sw   x27, 18*4(sp)
                sw   x26, 17*4(sp)
                sw   x25, 16*4(sp)
                sw   x24, 15*4(sp)
                sw   x23, 14*4(sp)
                sw   x22, 13*4(sp)
                sw   x21, 12*4(sp)
                sw   x20, 11*4(sp)
                sw   x19, 10*4(sp)
                sw   x18,  9*4(sp)
                sw    x9,  8*4(sp)
                sw    x8,  7*4(sp)
                sw    x4,  6*4(sp)
                sw    x3,  5*4(sp)
                sw    x7,  4*4(sp)
                sw    x5,  3*4(sp)
                sw    x1,  2*4(sp)

                // At this point, we are free to clobber saved registers. We
                // embrace using registers like `s0` and `s1`, as they fit into
                // compressed loads and stores.

                // Load the address of `_start_ef_trap` into `1*4(sp)`. We swap
                // our stack pointer into the mscratch CSR and the trap handler
                // will load and jump to the address at this offset.
                la    s0, 600f      // s0 = _start_ef_trap
                sw    s0, 1*4(sp)   // 1*4(sp) = s0

                // sw x0, 0*4(sp)   // Reserved as scratch space for trap handler

                // Now, copy the stacked arguments. For this we need to:
                //
                // 1. load the current foreign stack pointer,
                // 2. subtract the amount of bytes occupied by stacked arguments,
                // 3. align the new stack pointer downward to a 16-byte boundary,
                // 4. check whether the new stack pointer would overflow,
                // 5. copy `t3` bytes from our current stack pointer to the
                //    foreign stack.
                //
                // Load the foreign stack pointer (fsp) and the bottom of the
                // stack from our runtime:
                lw    s0, {rtas_foreign_stack_ptr_offset}(t0)
                lw    s1, {rtas_foreign_stack_bottom_offset}(t0)

                // Check if our subtraction would underflow:
                bltu  s0, t3, 200f  // If fsp < stack_spill, overflow!

                // Move the stack downward by `t3` (`stack_spill`) and align.
                // Aligning downward to the next 16-byte boundary cannot
                // possibly underflow:
                sub   s0, s0, t3    // fsp -= stack_spill
                andi  s0, -16       // fsp -= fsp % 16

                // Just because the above operation did not wrap around does not
                // mean that we did not overflow our stack. Check that we're not
                // lower than stack_bottom:
                bge   s0, s1, 300f  // If fsp >= stack_bottom, no overflow!

              200: // _spill_stack_overflow
                unimp               // TODO: error handling!

              300: // _no_spill_stack_overflow
                // The foreign stack is now properly aligned. Copy stack_spill
                // (t3) bytes from our original stack pointer (t5) to fsp (s0).
                //
                // We decrement both the original sp and fsp one word at a time.
                // While we can clobber `t5`, we need to retain fsp (`s0`).
                // Instead, use a copy in `s1`:
                mv    s1, s0        // fsp' = fsp

                // To make sure we don't overshoot our word-copy loop, we round
                // up the stack spill to a multiple of 4 bytes (word size).
                // While this really should always be word-aligned, we're better
                // safe than sorry. In the worst case, we'll copy an extra
                // <= 3 bytes.
                addi  t3, t3, 3
                andi  t3, t3, -8

              400: // _stack_copy
                // Copy the stack, implemented as a while (cond) { copy } loop:
                beq   t3, x0, 500f  // If to copy == 0, jump to _stack_copied
                lw    s2, 0(t5)     // Load a word from our original stack
                sw    s2, 0(s1)     // Store it onto fsp'
                addi  t5, t5, 4     // original sp += 4 (one word)
                addi  s1, s1, 4     // fsp'        += 4 (one word)
                addi  t3, t3, -4    // to copy     -= 4 (one word)
                j     400b          // loop!

              500f: // _stack_copied
                // From here on we can't allow the CPU to take interrupts
                // anymore, as we re-route traps to `_start_ef_trap` below (by
                // writing our stack pointer into the mscratch CSR), and we rely
                // on certain CSRs to not be modified or used in their
                // intermediate states (e.g., mepc).
                //
                // We atomically switch to user-mode and re-enable interrupts
                // using the `mret` instruction below.
                //
                // If interrupts are disabled _after_ setting mscratch, this
                // result in the race condition of
                // [PR 2308](https://github.com/tock/tock/pull/2308)

                // Therefore, clear the following bits in mstatus first:
                //   0x00000008 -> bit 3 -> MIE (disabling interrupts here)
                // + 0x00001800 -> bits 11,12 -> MPP (switch to usermode on mret)
                li    s1, 0x00001808
                csrc  mstatus, s1         // clear bits in mstatus

                // Afterwards, set the following bits in mstatus:
                //   0x00000080 -> bit 7 -> MPIE (enable interrupts on mret)
                li    s1, 0x00000080
                csrs  mstatus, s1         // set bits in mstatus

                // Execute `_start_ef_trap` on a trap by setting the mscratch
                // trap handler address to our current stack pointer. This stack
                // pointer, at `1*4(sp)`, holds the address of `_start_ef_trap`.
                //
                // Upon a trap, the global trap handler (_start_trap) will swap
                // `s0` with the `mscratch` CSR and, if it contains a non-zero
                // address, jump to the address that is now at `1*4(s0)`. This
                // allows us to hook a custom trap handler that saves all
                // userspace state:
                //
                csrw  mscratch, sp        // Store `sp` in mscratch CSR. Discard
                                          // the prior value (was zero)

                // We have to set the mepc CSR with the PC we want the app to
                // start executing at. This has been loaded into register `t1`
                // by the `invoke` wrapper for us:
                csrw  mepc, t1            // Set mepc to the function to run.

                // Switch to the application's stack pointer, which is aligned
                // to a 16-byte boundary (as required by the RISC-V C ABI) and
                // has all spilled arugments copied onto:
                mv    sp, s0

                // All other argument registers have not been clobbered so far.
                // Set our return address to the return springboard:
                la    ra, {ret_springboard)

                // Clear all Rust state that the function should not have access
                // to. This is not strictly necessary under all threat models,
                // but it's a good way to test that we're actually restoring all
                // of them:
                mv    x3, x0        // gp
                mv    x4, x0        // tp
                mv    x5, x0        // t0
                mv    x6, x0        // t1
                mv    x7, x0        // t2
                mv    x8, x0        // s0 / fp
                mv    x9, x0        // s1
                mv   x18, x0        // s2
                mv   x19, x0        // s3
                mv   x20, x0        // s4
                mv   x21, x0        // s5
                mv   x22, x0        // s6
                mv   x23, x0        // s7
                mv   x24, x0        // s8
                mv   x25, x0        // s9
                mv   x26, x0        // s10
                mv   x27, x0        // s11
                mv   x28, x0        // t3
                mv   x29, x0        // t4
                mv   x30, x0        // t5
                mv   x31, x0        // t6

                // Execute the foreign function, re-enabling interrupts.
                mret

                // The global trap handler will jump to this address when
                // catching a trap while the foreign function is executing
                // (address loaded into the mscratch CSR).
                //
                // This custom trap handler is responsible for saving
                // application state, clearing the custom trap handler
                // (mscratch = 0), and restoring the kernel context.

              600: // _start_ef_trap
                // At this point all we know is that we entered the trap handler
                // from an app. We don't know _why_ we got a trap, it could be
                // from an interrupt, syscall, or fault (or maybe something
                // else). Therefore we have to be very careful not to overwrite
                // any registers before we have saved them.
                //
                // The global trap handler has swapped the functions's `s0` into
                // the mscratch CSR, which now contains the address of our stack
                // pointer. The global trap handler further clobbered `s1`,
                // which now contains the address of `_start_ef_trap`. The
                // function's `s1` is saved at `0*4(s0)`.
                //
                // Thus we can clobber `s1` to inspect the source of our trap /
                // interrupt, and branch accordingly.
                csrr  s1, mcause

                // If mcause is greater than or equal to zero this was not an
                // interrupt (i.e. the most significant bit is not 1). In this
                // case, jump to _start_ef_trap_continue.
                bge   s1, x0, 700f

                // This was an interrupt! We save all callee-saved registers
                // and call the function to disable interrupts. We then proceed
                // executing the application.

                // First, save the foreign stack pointer onto our stack:
                sw    x2, 35*4(s0)

                // Now, we can restore the Rust stack pointer, and get an
                // additional register to clobber (s0):
                mv    sp, s0

                // We reset mscratch to 0 (kernel trap handler mode), to ensure
                // that all faults in the interrupt disable function are handled
                // as kernel faults. This also restores the function's s0.
                csrrw s0, mscratch, zero

                // Now, continue to save the remaining registers. Using `sp`
                // will allow all of these to be compressed instructions:
                sw    x4, 37*4(sp) // tp
                sw    x3, 36*4(sp) // gp
                // sw x2, 35*4(sp) // sp, saved above
                sw   x31, 34*4(sp) // t6
                sw   x30, 33*4(sp) // t5
                sw   x29, 32*4(sp) // t4
                sw   x28, 31*4(sp) // t3
                sw   x17, 30*4(sp) // a7
                sw   x16, 29*4(sp) // a6
                sw   x15, 28*4(sp) // a5
                sw   x14, 27*4(sp) // a4
                sw   x13, 26*4(sp) // a3
                sw   x12, 25*4(sp) // a2
                sw   x11, 24*4(sp) // a1
                sw   x10, 23*4(sp) // a0
                sw    x7, 22*4(sp) // t2
                sw    x6, 21*4(sp) // t1
                sw    x5, 20*4(sp) // t0
                sw    x1, 19*4(sp) // ra

                // Restore some important context for Rust, namely the thread-
                // and global-pointers:
                lw    x4,  6*4(sp) // tp
                lw    x3,  5*4(sp) // gp

                // Disable the interrupt. This requires `mcause` (currently in
                // `s1`) to be loaded into `a0`:
                mv   a0, s1     // a0 = s1 (mcause)
                jal  ra, _disable_interrupt_trap_rust_from_app

                // Restore reigsters from the stack:
                lw    x1, 19*4(sp) // ra
                lw    x5, 20*4(sp) // t0
                lw    x6, 21*4(sp) // t1
                lw    x7, 22*4(sp) // t2
                lw   x10, 23*4(sp) // a0
                lw   x11, 24*4(sp) // a1
                lw   x12, 25*4(sp) // a2
                lw   x13, 26*4(sp) // a3
                lw   x14, 27*4(sp) // a4
                lw   x15, 28*4(sp) // a5
                lw   x16, 29*4(sp) // a6
                lw   x17, 30*4(sp) // a7
                lw   x28, 31*4(sp) // t3
                lw   x29, 32*4(sp) // t4
                lw   x30, 33*4(sp) // t5
                lw   x31, 34*4(sp) // t6
                // lw x2, 35*4(sp) // sp, load last as we overwrite our pointer
                lw    x3, 36*4(sp) // gp
                lw    x4, 37*4(sp) // tp

                // Reset the trap handler by switching our kernel stack into
                // `mscratch` again. We discard its current value, which must
                // be zero (kernel trap handler mode).
                csrw  mscratch, sp

                // Restore the function's s1, as it was clobbered by the trap
                // handler:
                lw    x9, 0*4(sp)

                // Finally, load back the functions's stack pointer,
                // the last register:
                lw    x2, 35*4(sp) // sp

                // Return to the function:
                mret

              700: // _start_ef_trap_continue
                // This was not an interrupt. We need to extract all required
                // information, restore kernel state, and then hand off to a
                // final Rust function that encodes the return value.

                // Restore all TODO!

                lw   s1, 2*4(s0)

              // With the per-process stored state address in `t1`, save all
              // non-clobbered registers:
              //
              sw    x1,  0*4(s1)        // ra
              sw    x2,  1*4(s1)        // sp
              sw    x3,  2*4(s1)        // gp
              sw    x4,  3*4(s1)        // tp
              sw    x5,  4*4(s1)        // t0
              sw    x6,  5*4(s1)        // t1
              sw    x7,  6*4(s1)        // t2
              // ------------------------> s0, in mscratch right now
              // ------------------------> s1, stored at 0*4(s0) right now
              sw   x10,  9*4(s1)        // a0
              sw   x11, 10*4(s1)        // a1
              sw   x12, 11*4(s1)        // a2
              sw   x13, 12*4(s1)        // a3
              sw   x14, 13*4(s1)        // a4
              sw   x15, 14*4(s1)        // a5
              sw   x16, 15*4(s1)        // a6
              sw   x17, 16*4(s1)        // a7
              sw   x18, 17*4(s1)        // s2
              sw   x19, 18*4(s1)        // s3
              sw   x20, 19*4(s1)        // s4
              sw   x21, 20*4(s1)        // s5
              sw   x22, 21*4(s1)        // s6
              sw   x23, 22*4(s1)        // s7
              sw   x24, 23*4(s1)        // s8
              sw   x25, 24*4(s1)        // s9
              sw   x26, 25*4(s1)        // s10
              sw   x27, 26*4(s1)        // s11
              sw   x28, 27*4(s1)        // t3
              sw   x29, 28*4(s1)        // t4
              sw   x30, 29*4(s1)        // t5
              sw   x31, 30*4(s1)        // t6

              // At this point, we can restore s0 into our stack pointer:
              mv   sp, s0

              // Now retrieve the original value of s1 and save that as well. We
              // must not clobber s1, our per-process stored state pointer.
              lw   s0,  0*4(sp)         // s0 = app s1 (from trap handler scratch space)
              sw   s0,  8*4(s1)         // Save app s1 to per-process state

              // Retrieve the original value of s0 from the mscratch CSR, save it.
              //
              // This will also restore the kernel trap handler by writing zero to
              // the CSR. `csrrw` allows us to read and write the CSR in a single
              // instruction:
              csrrw s0, mscratch, zero  // s0 <- mscratch[app s0] <- zero
              sw    s0, 7*4(s1)         // Save app s0 to per-process state

              // -------------------------------------------------------------------
              // At this point, the entire app register file is saved. We also
              // restored the kernel trap handler. We have restored the following
              // kernel registers:
              //
              // - sp: kernel stack pointer
              // - s1: per-process stored state pointer
              //
              // We avoid clobbering those registers from this point onward.
              // -------------------------------------------------------------------

              // We also need to store some other information about the trap reason,
              // present in CSRs:
              //
              // - the app's PC (mepc),
              // - the trap reason (mcause),
              // - the trap 'value' (mtval, e.g., faulting address).
              //
              // We need to store mcause because we use that to determine why the
              // app stopped executing and returned to the kernel. We store mepc
              // because it is where we need to return to in the app at some
              // point. We need to store mtval in case the app faulted and we need
              // mtval to help with debugging.
              //
              // We use `s0` as a scratch register, as it fits into the 3-bit
              // register argument of RISC-V compressed loads / stores:

              // Save the PC to the stored state struct. We also load the address
              // of _return_to_kernel into it, as this will be where we jump on
              // the mret instruction, which leaves the trap handler.
              la    s0, 300f            // Load _return_to_kernel into t0.
              csrrw s0, mepc, s0        // s0 <- mepc[app pc] <- _return_to_kernel
              sw    s0, 31*4(s1)        // Store app's pc in stored state struct.

              // Save mtval to the stored state struct
              csrr  s0, mtval
              sw    s0, 33*4(s1)

              // Save mcause and leave it loaded into a0, as we call a function
              // with it below:
              csrr  a0, mcause
              sw    a0, 32*4(s1)

              // Depending on the value of a0, we might be calling into a function
              // while still in the trap handler. The callee may rely on the `gp`,
              // `tp`, and `fp` (s0) registers to be set correctly. Thus we restore
              // them here, as we need to do anyways. They are saved registers,
              // and so we avoid clobbering them beyond this point.
              //
              // We do not restore `s1`, as we need to move it back into `a0`
              // _after_ potentially invoking the _disable_interrupt_... function.
              // LLVM relies on it to not be clobbered internally, but it is not
              // part of the RISC-V C ABI, which we need to follow here.
              //
              lw    x8, 5*4(sp)         // fp/s0: Restore the frame pointer
              lw    x4, 4*4(sp)         // tp: Restore the thread pointer
              lw    x3, 3*4(sp)         // gp: Restore the global pointer

              // --------------------------------------------------------------------
              // From this point onward, avoid clobbering the following registers:
              //
              // - x2 / sp: kernel stack pointer
              // - x3 / gp: kernel global pointer
              // - x4 / tp: kernel thread pointer
              // - x8 / s0 / fp: kernel frame pointer
              // - x9 / s1: per-process stored state pointer
              //
              // --------------------------------------------------------------------

              // Now we need to check if this was an interrupt, and if it was,
              // then we need to disable the interrupt before returning from this
              // trap handler so that it does not fire again.
              //
              // If mcause is greater than or equal to zero this was not an
              // interrupt (i.e. the most significant bit is not 1). In this case,
              // jump to _start_app_trap_continue.
              bge   a0, zero, 200f

              // This was an interrupt. Call the interrupt disable function, with
              // mcause already loaded in a0.
              //
              // This may clobber all caller-saved registers. However, at this
              // stage, we only restored `sp`, `s1`, and the registers above, all of
              // which are saved. Thus we don't have to worry about the function
              // call clobbering these registers.
              //
              jal  ra, _disable_interrupt_trap_rust_from_app

            200: // _start_app_trap_continue

              // Need to set mstatus.MPP to 0b11 so that we stay in machine mode.
              //
              // We use `a0` as a scratch register, as we are allowed to clobber it
              // here, and it fits into a compressed load instruction. We must avoid
              // using restored saved registers like `s0`, etc.
              //
              li    a0, 0x1800          // Load 0b11 to the MPP bits location in a0
              csrs  mstatus, a0         // mstatus |= a0

              // Use mret to exit the trap handler and return to the context
              // switching code. We loaded the address of _return_to_kernel
              // into mepc above.
              mret

              // This is where the trap handler jumps back to after the app stops
              // executing.
            300: // _return_to_kernel

              // We have already stored the app registers in the trap handler. We
              // have further restored `gp`, `tp`, `fp`/`s0` and the stack pointer.
              //
              // The only other non-clobbered registers are `s1` and `a0`, where
              // `a0` needs to hold the per-process state pointer currently stored
              // in `s1`, and the original value of `s1` is saved on the stack.
              // Restore them:
              //
              mv    a0, s1              // a0 = per-process stored state
              lw    s1, 6*4(sp)         // restore s1 (used by LLVM internally)

              // We need thus need to mark all registers as clobbered, except:
              //
              // - x2  (sp)
              // - x3  (gp)
              // - x4  (tp)
              // - x8  (fp)
              // - x9  (s1)
              // - x10 (a0)

              addi sp, sp, 8*4   // Reset kernel stack pointer
            ",
            ret_springboard = sym ef_tock_rv32i_c_rt_ret_springboard,
            cb_springboard = sym ef_tock_rv32i_c_rt_cb_springboard,
            options(noreturn),
        );
    }
}

unsafe impl<ID: EFID, M: MPU + 'static> EncapfnRt for TockRv32iCRt<ID, M> {
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
            // We don't need to create a new `alloc_scope` or modify or current
            // one -- we allocate our stack in the same memory that is usable by
            // the foreign function for any other purpose. Hence, we only check
            // whether this memory actually belongs to the process.
            fun(ptr, alloc_scope)
        })
    }
}

extern "C" {
    fn ef_tock_rv32i_c_rt_ret_springboard();
    fn ef_tock_rv32i_c_rt_cb_springboard();
}

core::arch::global_asm!(
    "
      .global ef_tock_rv32i_c_rt_ret_springboard
      ef_tock_rv32i_c_rt_ret_springboard:
        // Return to machine-mode with an environment call or an instruction
        // access fault from a well-known address:
        ecall

      .global ef_tock_rv32i_c_rt_cb_springboard
      ef_tock_rv32i_c_rt_cb_springboard:
        // Return to kernel with an illegal instruction trap or an instruction
        // access fault from a well-known address:
        unimp
    "
);
