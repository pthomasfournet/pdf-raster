//! PDF dictionary newtype.
//!
//! Wraps a `HashMap<Vec<u8>, Object>` with lopdf-compatible accessors
//! (`get(&[u8]) -> Option<&Object>`, `set(K, V)`, etc.) so call sites can use
//! byte literals (`dict.get(b"Type")`) without manual `.as_ref()`.

use std::collections::HashMap;

use crate::object::Object;

/// PDF dictionary: an ordered-by-insertion mapping from byte-string keys to
/// [`Object`] values.
///
/// We store entries in a `HashMap` (insertion order is not preserved, matching
/// PDF semantics — the spec gives no order guarantee for dict entries).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dictionary(HashMap<Vec<u8>, Object>);

impl Dictionary {
    /// Return an empty dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Look up `key`. Returns `None` if absent.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<&Object> {
        self.0.get(key)
    }

    /// Mutable lookup.
    pub fn get_mut(&mut self, key: &[u8]) -> Option<&mut Object> {
        self.0.get_mut(key)
    }

    /// `true` if `key` is present.
    #[must_use]
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.0.contains_key(key)
    }

    /// Insert. Returns the previous value if any.
    pub fn insert(&mut self, key: Vec<u8>, value: Object) -> Option<Object> {
        self.0.insert(key, value)
    }

    /// Insert with a string-like key (for ergonomics in tests).
    pub fn set<K: AsRef<[u8]>>(&mut self, key: K, value: Object) {
        self.0.insert(key.as_ref().to_vec(), value);
    }

    /// Remove and return.
    pub fn remove(&mut self, key: &[u8]) -> Option<Object> {
        self.0.remove(key)
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// `true` if no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over `(&key, &value)`.
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, Vec<u8>, Object> {
        self.0.iter()
    }

    /// Iterate over keys.
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, Vec<u8>, Object> {
        self.0.keys()
    }
}

impl<'a> IntoIterator for &'a Dictionary {
    type Item = (&'a Vec<u8>, &'a Object);
    type IntoIter = std::collections::hash_map::Iter<'a, Vec<u8>, Object>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl FromIterator<(Vec<u8>, Object)> for Dictionary {
    fn from_iter<T: IntoIterator<Item = (Vec<u8>, Object)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<HashMap<Vec<u8>, Object>> for Dictionary {
    fn from(map: HashMap<Vec<u8>, Object>) -> Self {
        Self(map)
    }
}

impl From<Dictionary> for HashMap<Vec<u8>, Object> {
    fn from(d: Dictionary) -> Self {
        d.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_with_byte_literal() {
        let mut d = Dictionary::new();
        d.set("Type", Object::Name(b"Page".to_vec()));
        // The whole point: byte literal lookup, no .as_ref() needed.
        assert!(d.get(b"Type").is_some());
        assert!(d.get(b"Missing").is_none());
    }
}
