#include "DebugGradient.h"

#ifdef BB_DEBUG_MODE
CheckVar::CheckVar()
{
  BB_DEBUG_PARENT_CHECK=false;
  BB_DEBUG_FEASIBILITY_CHECK=false;
  BB_DEBUG_NO_WARMSTART_CHECK=false;
  BB_DEBUG_INTERNAL_WARMSTART_CHECK=false;
  BB_DEBUG_LEAF_WARMSTART_CHECK=false;
}
void CheckVar::report()
{
  INFOV("BB_DEBUG_MODE: parentCheck=%d feasibilityCheck=%d noWarmStartCheck=%d internalWarmStartCheck=%d leafWarmStartCheck=%d!",
        CheckVar::_instance.BB_DEBUG_PARENT_CHECK?1:0,
        CheckVar::_instance.BB_DEBUG_FEASIBILITY_CHECK?1:0,
        CheckVar::_instance.BB_DEBUG_NO_WARMSTART_CHECK?1:0,
        CheckVar::_instance.BB_DEBUG_INTERNAL_WARMSTART_CHECK?1:0,
        CheckVar::_instance.BB_DEBUG_LEAF_WARMSTART_CHECK?1:0)
}
CheckVar CheckVar::_instance;
#endif
