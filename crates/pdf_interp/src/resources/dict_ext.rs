//! Typed accessor helpers for [`lopdf::Dictionary`].
//!
//! The [`DictExt`] trait adds concise `get_name`, `get_i64`, and `get_bool`
//! methods that collapse the common `.get(key).ok()?.as_TYPE().ok()?` chain
//! into a single call, returning `Option<T>` in all cases.
//!
//! Import with `use crate::resources::dict_ext::DictExt;` (or via the
//! resources prelude once one is added).

use lopdf::{Dictionary, Object};

/// Convenience accessors for [`lopdf::Dictionary`] that return `Option<T>`
/// instead of `lopdf::Error`.
pub trait DictExt {
    /// Return the value at `key` as a byte slice name (`/Name`), or `None`.
    fn get_name<'a>(&'a self, key: &[u8]) -> Option<&'a [u8]>;

    /// Return the value at `key` as an `i64` integer, or `None`.
    fn get_i64(&self, key: &[u8]) -> Option<i64>;

    /// Return the value at `key` as a `bool`, or `None`.
    fn get_bool(&self, key: &[u8]) -> Option<bool>;
}

impl DictExt for Dictionary {
    fn get_name<'a>(&'a self, key: &[u8]) -> Option<&'a [u8]> {
        self.get(key).ok()?.as_name().ok()
    }

    fn get_i64(&self, key: &[u8]) -> Option<i64> {
        self.get(key).ok()?.as_i64().ok()
    }

    fn get_bool(&self, key: &[u8]) -> Option<bool> {
        match self.get(key).ok()? {
            Object::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}
