#pragma once
// Host admission policy, independent from neural-network operators and ESP-NOW.
// A worker holding an output tensor is still BUSY until RELEASE/READY.
namespace noodlepipe {
inline unsigned effective_reserve(unsigned online,unsigned requested) {
    // One-worker bring-up must remain possible; it uses RUN_LOCAL as needed.
    return online>1 ? (requested<online ? requested:online-1) : 0;
}
inline bool host_can_admit(unsigned active_frames,unsigned host_limit,
                           unsigned online_workers,unsigned idle_workers,
                           unsigned requested_reserve) {
    return online_workers>0 && active_frames<host_limit && idle_workers>0 &&
           idle_workers>effective_reserve(online_workers,requested_reserve);
}
}
