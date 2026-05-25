//! Wayland protocol filter for sandboxed desktop apps.
//!
//! Implements an allowlist-based filter for Wayland global interfaces. The
//! filter is applied to `wl_registry.global` events emitted by the compositor:
//! globals not on the allowlist are silently dropped, so sandboxed apps never
//! learn they exist.
//!
//! The full proxy (intercepting actual socket bytes, relaying filtered events,
//! and forwarding all other messages) requires a running compositor.
//!
//! TODO(compositor): wire `WaylandFilter` into a real socket-pair proxy once
//! the kiki compositor is booting. The proxy lives between the app's Wayland
//! client socket and the real compositor socket; it forwards everything except
//! `wl_registry.global` events for denied interfaces. The filter logic
//! implemented here is correct and tested independently.

/// Allowed Wayland global interfaces — all others are filtered out.
///
/// This set grants the primitives needed by well-behaved desktop apps (draw,
/// seat input, outputs, xdg shell surfaces) while hiding the screencopy,
/// data-device, and admin protocols that would let a sandboxed app exfiltrate
/// the screen or intercept global input.
pub const DEFAULT_ALLOWLIST: &[&str] = &[
    "wl_compositor",
    "wl_shm",
    "wl_seat",
    "wl_output",
    "xdg_wm_base",
    "xdg_output_manager_v1",
    "wp_viewporter",
    "wp_fractional_scale_manager_v1",
];

/// A Wayland global descriptor, mirroring the fields of a `wl_registry.global`
/// event: a numeric name (the compositor-assigned object id), the interface
/// name string, and the maximum supported version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaylandGlobal {
    /// The compositor-assigned numeric name for this global.
    pub name:      u32,
    /// The interface name (e.g. `"wl_compositor"`).
    pub interface: String,
    /// Maximum version supported by the compositor for this interface.
    pub version:   u32,
}

/// Filter for `wl_registry.global` events.
///
/// Constructed with an allowlist of interface names. Call
/// [`WaylandFilter::filter_global_event`] for each global the compositor
/// announces; denied globals return `None`.
pub struct WaylandFilter {
    allowed: Vec<String>,
}

impl WaylandFilter {
    /// Construct a filter with the given allowlist.
    pub fn new(allowed: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self { allowed: allowed.into_iter().map(Into::into).collect() }
    }

    /// Construct a filter using [`DEFAULT_ALLOWLIST`].
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_ALLOWLIST.iter().copied())
    }

    /// Returns `true` if the named interface is on the allowlist.
    pub fn is_allowed(&self, interface: &str) -> bool {
        self.allowed.iter().any(|a| a == interface)
    }

    /// Filter a `wl_registry.global` event.
    ///
    /// Returns `Some(WaylandGlobal)` if the interface is allowed, `None` if it
    /// is denied (i.e. the proxy should drop this event silently).
    pub fn filter_global_event(
        &self,
        interface: &str,
        name:      u32,
        version:   u32,
    ) -> Option<WaylandGlobal> {
        if self.is_allowed(interface) {
            Some(WaylandGlobal { name, interface: interface.to_string(), version })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_protocol_passes() {
        let f = WaylandFilter::with_defaults();
        let result = f.filter_global_event("wl_compositor", 1, 6);
        assert!(result.is_some());
        let g = result.unwrap();
        assert_eq!(g.interface, "wl_compositor");
        assert_eq!(g.name, 1);
        assert_eq!(g.version, 6);
    }

    #[test]
    fn denied_protocol_is_filtered() {
        let f = WaylandFilter::with_defaults();
        // screencopy is not in the allowlist — must be denied.
        assert!(f.filter_global_event("zwlr_screencopy_manager_v1", 2, 3).is_none());
        // data-device manager (global clipboard / DnD) is also denied by default.
        assert!(f.filter_global_event("wl_data_device_manager", 3, 3).is_none());
    }

    #[test]
    fn filter_global_event_returns_none_for_screencopy() {
        let f = WaylandFilter::with_defaults();
        assert!(
            f.filter_global_event("zwlr_screencopy_manager_v1", 5, 3).is_none(),
            "screencopy must be denied by the default allowlist"
        );
        // Also test the older v1 screencopy variant.
        assert!(f.filter_global_event("wl_screencopy", 6, 1).is_none());
    }

    #[test]
    fn default_allowlist_includes_xdg_shell() {
        let f = WaylandFilter::with_defaults();
        // xdg_wm_base is the xdg shell global — desktop apps MUST have it.
        assert!(f.is_allowed("xdg_wm_base"), "xdg_wm_base must be in the default allowlist");
        // As well as the fractional scale manager for HiDPI.
        assert!(
            f.is_allowed("wp_fractional_scale_manager_v1"),
            "wp_fractional_scale_manager_v1 must be in the default allowlist"
        );
    }

    #[test]
    fn custom_allowlist_overrides_defaults() {
        // A custom filter that only allows wl_compositor.
        let f = WaylandFilter::new(["wl_compositor"]);
        assert!(f.is_allowed("wl_compositor"));
        // xdg_wm_base would be allowed by defaults but NOT by this custom filter.
        assert!(!f.is_allowed("xdg_wm_base"));
        assert!(f.filter_global_event("xdg_wm_base", 1, 6).is_none());
    }

    #[test]
    fn all_default_allowlist_entries_pass() {
        let f = WaylandFilter::with_defaults();
        for (i, iface) in DEFAULT_ALLOWLIST.iter().enumerate() {
            let result = f.filter_global_event(iface, i as u32 + 1, 1);
            assert!(
                result.is_some(),
                "default allowlist entry '{iface}' should pass the filter"
            );
        }
    }
}
