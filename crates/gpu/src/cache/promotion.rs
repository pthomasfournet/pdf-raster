//! Tier-promotion logic for the image cache.
//!
//! Pulled out of `mod.rs` so the host-tier and disk-tier promote-on-hit
//! paths live in one focused module.  These run on a VRAM miss and lift
//! the entry back into VRAM (one `PCIe` upload) so the renderer never
//! sees a re-decode for an image that's still anywhere in the cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::mapref::entry::Entry;

use super::disk_tier::LookupCallbackError;
use super::host_tier::{HostEntry, HostTier};
use super::{CachedDeviceImage, ContentHash, DeviceImageCache, DocId};

impl DeviceImageCache {
    /// Lift a host-tier entry back into VRAM and install it in the
    /// primary index.  Returns `None` if the host tier doesn't have it
    /// or the upload fails (treated as a cache miss; caller re-decodes).
    pub(super) fn promote_from_host(&self, hash: &ContentHash) -> Option<Arc<CachedDeviceImage>> {
        let host_entry = self.host.lookup(hash, self.next_tick())?;
        let bytes = host_entry.pinned.num_bytes() as u64;
        // Make room in the VRAM tier.  If we can't (every entry pinned
        // by in-flight kernels), fall back to a cache miss; the host
        // entry stays in place for a future retry.
        if self.evict_to_fit(bytes).is_err() {
            return None;
        }
        // Pass the `PinnedHostSlice` directly so cudarc's `HostSlice`
        // impl records the H→D copy against the slice's internal event
        // — without that, dropping `host_entry` while the DMA is still
        // in flight would let `PinnedHostSlice::Drop` free the source
        // buffer mid-copy.  Calling `as_slice()` first would route
        // through the plain `[u8]` impl whose `SyncOnDrop` is a no-op.
        let device_ptr = self
            .stream
            .clone_htod(&host_entry.pinned)
            .map_err(|e| log::warn!("host-tier promotion upload failed: {e}"))
            .ok()?;
        let new_entry = Arc::new(CachedDeviceImage {
            device_ptr,
            width: host_entry.width,
            height: host_entry.height,
            layout: host_entry.layout,
            last_used: AtomicU64::new(self.next_tick()),
        });
        match self.primary.entry(*hash) {
            Entry::Occupied(occ) => {
                // Another thread won the same race — return their entry.
                Some(occ.get().clone())
            }
            Entry::Vacant(vac) => {
                let _ = self
                    .used_bytes
                    .fetch_add(new_entry.vram_bytes(), Ordering::Relaxed);
                let _ = vac.insert(new_entry.clone());
                Some(new_entry)
            }
        }
    }

    /// Lift a disk-tier entry back into VRAM.  Reads the file via
    /// `posix_fadvise(WILLNEED)` + `read_exact` directly into a
    /// fresh pinned host slab (no transient `Vec`), uploads to
    /// VRAM, and installs in the primary index.  The host tier is
    /// also populated as a side effect so a later same-session
    /// lookup that misses VRAM hits host RAM instead of going back
    /// to disk.
    ///
    /// Returns `None` on disk miss, malformed file, allocation
    /// failure, or upload error — caller treats as a cache miss and
    /// re-decodes the source bytes.
    pub(super) fn promote_from_disk(
        &self,
        doc: DocId,
        hash: &ContentHash,
    ) -> Option<Arc<CachedDeviceImage>> {
        let disk = self.disk.as_ref()?;
        // Slab outlives the callback; the closure fills it via
        // `read_exact` and hands ownership back via `pinned`.
        let mut pinned: Option<cudarc::driver::PinnedHostSlice<u8>> = None;
        let info = disk.lookup_into(doc, hash, |info, reader| {
            // Pinned alloc and slab-access failures are caller-side
            // — tag them `Resource` so a transient pinned-pool
            // exhaustion doesn't delete the (perfectly fine) disk
            // entry.
            let mut slab = HostTier::alloc_pinned(self.stream.context(), info.expected_pixel_bytes)
                .map_err(|e| {
                    LookupCallbackError::Resource(std::io::Error::other(format!(
                        "alloc_pinned failed: {e}"
                    )))
                })?;
            let dst = slab.as_mut_slice().map_err(|e| {
                LookupCallbackError::Resource(std::io::Error::other(format!(
                    "pinned slab access failed: {e}"
                )))
            })?;
            reader.read_exact(dst).map_err(LookupCallbackError::Read)?;
            pinned = Some(slab);
            Ok(())
        })?;
        let pinned = pinned?;
        log::debug!(
            "disk-tier: hit for ({doc:?}, {hash:?}) {}×{} {:?}",
            info.width,
            info.height,
            info.layout,
        );
        let host_entry = HostEntry::new(pinned, info.width, info.height, info.layout);
        let _arc = self.host.insert(*hash, host_entry, self.next_tick());
        self.promote_from_host(hash)
    }
}
