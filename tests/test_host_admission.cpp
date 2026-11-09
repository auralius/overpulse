#include <NoodleAdmission.h>
#include <cassert>
#include <cstdio>
int main() {
    using namespace noodlepipe;
    // Five workers; three host-sourced frames; keep spare receiver capacity.
    assert(host_can_admit(0,3,5,5,1));
    assert(host_can_admit(2,3,5,3,1));
    assert(!host_can_admit(3,3,5,2,1)); // admission credit exhausted
    assert(!host_can_admit(2,3,5,1,1)); // reserve final idle worker
    assert(host_can_admit(2,3,5,2,1));
    // Only one worker: don't prevent bring-up or local stages.
    assert(effective_reserve(1,1)==0);
    assert(host_can_admit(0,3,1,1,1));
    assert(!host_can_admit(1,3,1,0,1));
    // No workers or no pipeline budget.
    assert(!host_can_admit(0,3,0,0,1));
    assert(!host_can_admit(0,0,5,5,1));
    // Two workers: one initial stage can run, one spared.
    assert(host_can_admit(0,3,2,2,1));
    assert(!host_can_admit(1,3,2,1,1));
    std::puts("HOST_ADMISSION_POLICY_PASS");
}
