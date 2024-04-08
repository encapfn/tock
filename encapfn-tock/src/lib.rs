#![no_std]
#![feature(
    offset_of,
    naked_functions,
    asm_const,
    maybe_uninit_as_bytes,
    maybe_uninit_write_slice
)]
#![allow(named_asm_labels)]

pub mod binary;
pub mod rv32i_c_rt;
