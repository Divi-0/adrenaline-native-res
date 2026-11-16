//! taiHEN's function hooks (taihen.h), for the few calls of other modules
//! we watch: our function runs instead of the hooked one and passes the
//! call on to `Hook::next` (taiHEN chains hooks, so others' hooks on
//! the same function keep working).

use core::ffi::c_void;
use core::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use crate::sys::SceUID;

/// tai_hook_args_t.
#[repr(C)]
struct HookArgs {
    size: u32,
    module: *const u8,
    library_nid: u32,
    function_nid: u32,
    hook_function: *const c_void,
}

/// struct _tai_hook_user: one link of a hook chain.
#[repr(C)]
struct HookLink {
    next: usize,
    function: *const c_void,
    original: *const c_void,
}

extern "C" {
    fn taiHookFunctionImportForUser(reference: *mut usize, args: *mut HookArgs) -> SceUID;
}

/// One hook: taiHEN's reference to the chain (tai_hook_ref_t) and the
/// hook's id, or < 0 while not set.
pub struct Hook {
    reference: AtomicUsize,
    id: AtomicI32,
}

impl Hook {
    pub const fn new() -> Self {
        Hook { reference: AtomicUsize::new(0), id: AtomicI32::new(-1) }
    }

    /// The hook's id (< 0: not set, or taiHEN's error).
    pub fn id(&self) -> SceUID {
        self.id.load(Ordering::Relaxed)
    }

    /// Hooks `module`'s import of function `function_nid` from library
    /// `library_nid` (`module` NUL-terminated); once only. Returns the id.
    ///
    /// # Safety
    /// `hook` must have the hooked function's signature and pass calls on
    /// with `call_next`.
    pub unsafe fn hook_import(&self, module: &[u8], library_nid: u32, function_nid: u32, hook: *const c_void) -> SceUID {
        if self.id() >= 0 {
            return self.id();
        }
        let mut args = HookArgs {
            size: core::mem::size_of::<HookArgs>() as u32,
            module: module.as_ptr(),
            library_nid,
            function_nid,
            hook_function: hook,
        };
        let id = taiHookFunctionImportForUser(self.reference.as_ptr(), &mut args);
        self.id.store(id, Ordering::Relaxed);
        id
    }

    /// The function the call goes on to (TAI_CONTINUE): the next hook in the
    /// chain, or the original function. Cast it to the hooked signature.
    ///
    /// # Safety
    /// Only from inside the hook, after `hook_import` succeeded.
    pub unsafe fn next(&self) -> *const c_void {
        let current = self.reference.load(Ordering::Relaxed) as *const HookLink;
        let next = (*current).next as *const HookLink;
        if next.is_null() {
            (*current).original
        } else {
            (*next).function
        }
    }
}
