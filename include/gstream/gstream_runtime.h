#ifndef GSTREAM_RUNTIME_H
#define GSTREAM_RUNTIME_H
#include <gstream/runtime/runtime_defines.h>

namespace gstream_runtime {

GStream* CreateGStream(GStreamConfig const& config, GStreamQuery* query);
void DestroyGStream(GStream const* gstreamInstance);
SuperstepReport BeginSuperstep(GStream const* gstreamInstance);
MemoryAllocationMode StringToMemoryAllocationMode(char const* mode_str);

} // namespace gstream_runtime

#endif // !GSTREAM_RUNTIME_H
