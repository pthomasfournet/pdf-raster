//! Adapter from `ash`'s `vk::Result` to `BackendError`.

use ash::vk;

use crate::backend::BackendError;

/// Wrap a Vulkan call's `vk::Result` failure into a `BackendError` with
/// the failing call's name baked into the message.
///
/// Use as `unsafe { instance.create_xyz(...) }.map_err(vk_err("vkCreateXyz"))?`.
pub(super) fn vk_err(call: &'static str) -> impl Fn(vk::Result) -> BackendError + use<> {
    move |code| BackendError::msg(format!("{call} failed: {code:?}"))
}
