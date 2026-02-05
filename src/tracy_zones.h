#pragma once

// Tracy profiler integration macros
// When TRACY_ENABLE is defined, these expand to Tracy macros
// When disabled, they expand to nothing (zero overhead)

#ifdef TRACY_ENABLE
  #include <tracy/Tracy.hpp>

  // RAII scope guards - automatically end when scope exits
  // Uses __FUNCTION__ for automatic naming
  #define AZ_ZONE_SCOPED ZoneScoped
  #define AZ_ZONE_NAMED(name) ZoneScopedN(name)

  // Frame marking for per-iteration timing
  #define AZ_FRAME_MARK FrameMark

  // Mutex tracking (use TracyLockable instead of std::mutex)
  #define AZ_LOCKABLE(type, var) TracyLockable(type, var)
  #define AZ_LOCKABLE_NAME(var, name) LockableName(var, name, strlen(name))

  // Thread naming
  #define AZ_SET_THREAD_NAME(name) tracy::SetThreadName(name)

#else
  // No-op versions when Tracy is disabled
  #define AZ_ZONE_SCOPED
  #define AZ_ZONE_NAMED(name)
  #define AZ_FRAME_MARK
  #define AZ_LOCKABLE(type, var) type var
  #define AZ_LOCKABLE_NAME(var, name)
  #define AZ_SET_THREAD_NAME(name)

#endif
