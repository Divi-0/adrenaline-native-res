//! adrenaline-native-res.suprx: the renderer (`nativeres_vita`) as a taiHEN plugin,
//! without extensions (see `nativeres_vita::ext`). scripts/build-plugin.sh
//! links this into the module.

#![no_std]

use core::ffi::c_void;

/// taiHEN calls this when it loads the plugin.
#[no_mangle]
pub extern "C" fn module_start(_args: u32, _argp: *mut c_void) -> i32 {
    nativeres_vita::plugin::start(None)
}

/// Never unloaded while Adrenaline runs (the hooks stay).
#[no_mangle]
pub extern "C" fn module_stop(_args: u32, _argp: *mut c_void) -> i32 {
    0
}
